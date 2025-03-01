#include "spdlog/spdlog.h"

#include "kmm/worker/scheduler.hpp"

namespace kmm {

static constexpr size_t NUM_DEFAULT_QUEUES = 3;
static constexpr size_t QUEUE_MISC = 0;
static constexpr size_t QUEUE_BUFFERS = 1;
static constexpr size_t QUEUE_HOST = 2;
static constexpr size_t QUEUE_DEVICES = 3;

struct QueueSlot {
    QueueSlot(std::shared_ptr<Task> node) : node(std::move(node)) {}

    std::shared_ptr<Task> node;
    bool scheduled = false;
    bool completed = false;
    std::shared_ptr<QueueSlot> next = nullptr;
};

struct QueueHeadTail {
    std::shared_ptr<QueueSlot> head;
    std::shared_ptr<QueueSlot> tail;
};

struct Scheduler::Queue {
    std::vector<std::deque<std::shared_ptr<Task>>> jobs;
    size_t max_concurrent_jobs = std::numeric_limits<size_t>::max();
    size_t num_jobs_active = 0;
    std::unordered_map<EventId, QueueHeadTail> node_to_slot;
    std::shared_ptr<QueueSlot> head = nullptr;
    std::shared_ptr<QueueSlot> tail = nullptr;

    void push_job(const Task* predecessor, std::shared_ptr<Task> node);
    bool pop_job(std::shared_ptr<Task>& node);
    void scheduled_job(std::shared_ptr<Task> node);
    void completed_job(std::shared_ptr<Task> node);
};

Scheduler::Scheduler(size_t num_devices) {
    m_queues.resize(NUM_DEFAULT_QUEUES + num_devices);

    for (size_t i = 0; i < num_devices; i++) {
        m_queues[QUEUE_DEVICES + i].max_concurrent_jobs = 5;
    }
}

Scheduler::~Scheduler() = default;

void Scheduler::submit(std::shared_ptr<Task> node) {
    KMM_ASSERT(node->status == Task::Status::Init);
    spdlog::debug(
        "submit task {} (command={}, dependencies={})",
        node->event_id,
        node->command,
        node->dependencies
    );

    size_t num_pending = node->dependencies.size();
    DeviceEventSet dependency_events;

    for (EventId dep_id : node->dependencies) {
        auto it = m_events.find(dep_id);

        if (it == m_events.end()) {
            num_pending--;
            continue;
        }

        auto& dep = it->second;
        dep->successors.push_back(node);

        if (dep->status == Task::Status::Executing) {
            num_pending--;
            dependency_events.insert(dep->execution_event);
        }

        if (dep->status == Task::Status::Completed) {
            num_pending--;
        }
    }

    node->status = Task::Status::AwaitingDependencies;
    node->queue_id = determine_queue_id(node->command);
    node->dependencies_pending = num_pending;
    node->dependency_events = std::move(dependency_events);
    enqueue_if_ready(nullptr, node);

    m_events.emplace(node->id(), std::move(node));
}

std::optional<std::shared_ptr<Task>> Scheduler::pop_ready(DeviceEventSet* deps_out) {
    std::shared_ptr<Task> result;

    for (auto& q : m_queues) {
        if (q.pop_job(result)) {
            spdlog::debug(
                "scheduling event {} (command={}, GPU deps={})",
                result->id(),
                result->command,
                result->dependency_events
            );

            KMM_ASSERT(result->status == Task::Status::ReadyToSubmit);
            result->status = Task::Status::Submitted;
            *deps_out = std::move(result->dependency_events);
            return result;
        }
    }

    return std::nullopt;
}

void Scheduler::mark_as_scheduled(EventId id, DeviceEvent event) {
    auto it = m_events.find(id);
    if (it == m_events.end()) {
        return;
    }

    auto node = it->second;

    spdlog::debug(
        "scheduled event {} (command={}, GPU event={})",
        node->id(),
        node->command,
        event
    );

    KMM_ASSERT(node->status == Task::Status::Submitted);
    node->status = Task::Status::Executing;
    node->execution_event = event;

    for (const auto& succ : node->successors) {
        succ->dependency_events.insert(event);
        succ->dependencies_pending -= 1;
        enqueue_if_ready(node.get(), succ);
    }

    m_queues.at(node->queue_id).scheduled_job(node);
}

void Scheduler::mark_as_completed(EventId id) {
    auto it = m_events.find(id);
    if (it == m_events.end()) {
        return;
    }

    auto node = it->second;
    spdlog::debug("completed event {} (command={})", node->id(), node->command);

    if (node->status == Task::Status::Submitted) {
        for (const auto& succ : node->successors) {
            succ->dependencies_pending -= 1;
            enqueue_if_ready(node.get(), succ);
        }

        node->status = Task::Status::Executing;
        m_queues.at(node->queue_id).scheduled_job(node);
    }

    KMM_ASSERT(node->status == Task::Status::Executing);
    node->status = Task::Status::Completed;
    m_events.erase(node->event_id);
    m_queues.at(node->queue_id).completed_job(node);
}
bool Scheduler::is_completed(EventId id) const {
    return m_events.find(id) == m_events.end();
}

bool Scheduler::is_idle() const {
    return m_events.empty();
}

size_t Scheduler::determine_queue_id(const Command& cmd) {
    if (const auto* p = std::get_if<CommandExecute>(&cmd)) {
        if (p->processor_id.is_device()) {
            return QUEUE_DEVICES + p->processor_id.as_device();
        } else {
            return QUEUE_HOST;
        }
    } else if (std::holds_alternative<CommandBufferCreate>(cmd) || std::holds_alternative<CommandBufferDelete>(cmd) || std::holds_alternative<CommandEmpty>(cmd)) {
        return QUEUE_BUFFERS;
    } else {
        return QUEUE_MISC;
    }
}

void Scheduler::enqueue_if_ready(const Task* predecessor, const std::shared_ptr<Task>& node) {
    if (node->status != Task::Status::AwaitingDependencies) {
        return;
    }

    if (node->dependencies_pending > 0) {
        return;
    }

    node->status = Task::Status::ReadyToSubmit;
    m_queues.at(node->queue_id).push_job(predecessor, node);
}

void Scheduler::Queue::push_job(const Task* predecessor, std::shared_ptr<Task> node) {
    auto event_id = node->id();
    auto new_slot = std::make_shared<QueueSlot>(std::move(node));
    node_to_slot.emplace(event_id, QueueHeadTail {new_slot, new_slot});

    if (predecessor != nullptr) {
        auto it = node_to_slot.find(predecessor->id());

        if (it != node_to_slot.end()) {
            KMM_ASSERT(it->second.head->node.get() == predecessor);
            auto current = std::exchange(it->second.tail, new_slot);
            auto next = std::exchange(current->next, new_slot);

            if (next != nullptr) {
                new_slot->next = std::move(next);
            } else {
                tail = new_slot;
            }

            return;
        }
    }

    // Predecessor is not in `node_to_slot`
    auto old_tail = std::exchange(tail, new_slot);

    if (old_tail != nullptr) {
        old_tail->next = new_slot;
    } else {
        head = new_slot;
    }
}

bool Scheduler::Queue::pop_job(std::shared_ptr<Task>& node) {
    if (num_jobs_active >= max_concurrent_jobs) {
        return false;
    }

    auto* p = head.get();

    while (true) {
        if (p == nullptr) {
            return false;
        }

        if (!p->scheduled) {
            break;
        }

        p = p->next.get();
    }

    num_jobs_active++;
    p->scheduled = true;
    node = p->node;
    return true;
}

void Scheduler::Queue::scheduled_job(std::shared_ptr<Task> node) {
    // Nothing to do after scheduling
}

void Scheduler::Queue::completed_job(std::shared_ptr<Task> node) {
    auto it = node_to_slot.find(node->id());
    if (it == node_to_slot.end()) {
        // ???
        return;
    }

    auto slot = it->second.head;
    node_to_slot.erase(it);

    // Set slot to completed
    KMM_ASSERT(slot->node == node);
    num_jobs_active--;
    slot->completed = true;

    // Remove all slots that have been marked as completed
    while (head != nullptr && head->completed) {
        head = head->next;
    }

    if (head == nullptr) {
        tail = nullptr;
    }
}

}  // namespace kmm