#include <queue>

#include "spdlog/spdlog.h"

#include "kmm/runtime/scheduler.hpp"
#include "kmm/runtime/task.hpp"

namespace kmm {

static constexpr size_t NUM_DEFAULT_QUEUES = 3;
static constexpr size_t QUEUE_JOIN = 0;
static constexpr size_t QUEUE_MISC = 1;
static constexpr size_t QUEUE_HOST = 2;
static constexpr size_t QUEUE_DEVICES = 3;

struct QueueSlot {
    std::shared_ptr<TaskRecord> inner;
};

bool operator<(const QueueSlot& lhs, const QueueSlot& rhs) {
    return lhs.inner->id() > rhs.inner->id();
}

struct SchedulerQueue {
    size_t max_concurrent_jobs = std::numeric_limits<size_t>::max();
    size_t num_jobs_active = 0;
    std::priority_queue<QueueSlot> tasks;

    void push_job(const TaskRecord* predecessor, std::shared_ptr<TaskRecord> record);
    std::shared_ptr<TaskRecord> pop_job();
    void scheduled_job(const TaskRecord& record);
    void completed_job(const TaskRecord& record);
};

void SchedulerQueue::push_job(const TaskRecord* predecessor, std::shared_ptr<TaskRecord> record) {
    this->tasks.push(QueueSlot {record});
}

std::shared_ptr<TaskRecord> SchedulerQueue::pop_job() {
    if (num_jobs_active >= max_concurrent_jobs) {
        return nullptr;
    }

    if (tasks.empty()) {
        return nullptr;
    }

    num_jobs_active++;
    auto result = std::move(tasks.top()).inner;
    tasks.pop();
    return result;
}

void SchedulerQueue::scheduled_job(const TaskRecord& record) {
    // Nothing to do after scheduling
}

void SchedulerQueue::completed_job(const TaskRecord& record) {
    num_jobs_active--;
}

Scheduler::Scheduler(
    std::shared_ptr<DeviceResources> device_resources,
    std::shared_ptr<DeviceStreamManager> stream_manager,
    std::shared_ptr<BufferRegistry> buffer_registry,
    bool debug_mode
) :
    m_device_resources(device_resources),
    m_stream_manager(stream_manager),
    m_buffer_registry(buffer_registry),
    m_debug_mode(debug_mode) {
    size_t num_devices = m_device_resources->num_contexts();
    m_ready_queues.resize(NUM_DEFAULT_QUEUES + num_devices);

    for (size_t i = 0; i < num_devices; i++) {
        m_ready_queues[QUEUE_DEVICES + i].max_concurrent_jobs = 5;
    }
}

Scheduler::~Scheduler() {}

void Scheduler::submit(EventId event_id, std::unique_ptr<Task> task, EventList dependencies) {
    auto record = std::make_shared<TaskRecord>(event_id, std::move(task));

    spdlog::debug(
        "submit task {} (command={}, dependencies={})",
        event_id,
        record->task->name(),
        dependencies
    );

    size_t num_pending = 0;
    DeviceEventSet dependency_events;

    for (EventId dep_id : dependencies) {
        auto it = m_tasks.find(dep_id);

        if (it == m_tasks.end()) {
            continue;
        }

        auto& dep = it->second;
        dep->successors.push_back(record);

        if (dep->status == TaskRecord::Status::WaitingForCompletion) {
            dependency_events.insert(dep->output_events);
        } else {
            num_pending++;
        }
    }

    record->status = TaskRecord::Status::AwaitingDependencies;
    record->predecessors = std::move(dependencies);
    record->queue = &m_ready_queues[determine_queue_id(*record->task)];
    record->predecessors_pending = num_pending;
    record->input_events = std::move(dependency_events);
    enqueue_if_ready(nullptr, record);

    m_tasks.emplace(event_id, std::move(record));
}

bool Scheduler::is_completed(EventId event_id) const {
    return m_tasks.find(event_id) == m_tasks.end();
}

bool Scheduler::is_idle() const {
    return m_running_head == nullptr && m_tasks.empty();
}

void Scheduler::make_progress() {
    TaskRecord* prev = nullptr;
    std::shared_ptr<TaskRecord>* current_ptr = &m_running_head;

    while (auto current = *current_ptr) {
        // In debug mode, only poll the head task (prev == nullptr)
        bool should_poll = !m_debug_mode || prev == nullptr;

        if (should_poll && this->poll_completion(*current) == Poll::Ready) {
            *current_ptr = std::move(current->next);
        } else {
            prev = current.get();
            current_ptr = &current->next;
        }
    }

    m_running_tail = prev;

    while (auto record = dequeue_ready_task()) {
        start_task(std::move(record));
    }
}

size_t Scheduler::determine_queue_id(const Task& task) {
    if (dynamic_cast<const JoinTask*>(&task) != nullptr) {
        return QUEUE_JOIN;
    } else if (dynamic_cast<const HostTask*>(&task) != nullptr) {
        return QUEUE_HOST;
    } else if (const auto* p = dynamic_cast<const DeviceTask*>(&task)) {
        return QUEUE_DEVICES + p->resource_id().as_device();
    } else {
        return QUEUE_MISC;
    }
}

void Scheduler::enqueue_if_ready(
    const TaskRecord* predecessor,
    const std::shared_ptr<TaskRecord>& task
) {
    if (task->status != TaskRecord::Status::AwaitingDependencies) {
        return;
    }

    if (task->predecessors_pending > 0) {
        return;
    }

    task->status = TaskRecord::Status::ReadyToStart;
    task->queue->push_job(predecessor, task);
}

std::shared_ptr<TaskRecord> Scheduler::dequeue_ready_task() {
    for (auto& q : m_ready_queues) {
        if (auto result = q.pop_job()) {
            return result;
        }
    }

    return nullptr;
}

void Scheduler::start_task(std::shared_ptr<TaskRecord> record) {
    KMM_ASSERT(record->status == TaskRecord::Status::ReadyToStart);

    if (poll_completion(*record) == Poll::Pending) {
        if (auto* old_tail = std::exchange(m_running_tail, record.get())) {
            old_tail->next = std::move(record);
        } else {
            m_running_head = std::move(record);
        }
    }
}

Poll Scheduler::poll_completion(TaskRecord& record) {
    if (record.status == TaskRecord::Status::ReadyToStart) {
        spdlog::debug(
            "scheduling task {} (command={}, GPU deps={})",
            record.id(),
            record.task->name(),
            record.input_events
        );

        record.status = TaskRecord::Status::Running;
        record.task->start(record.input_events);
    }

    if (record.status == TaskRecord::Status::Running) {
        if (record.task->poll(record, *this, record.output_events) != Poll::Ready) {
            return Poll::Pending;
        }

        spdlog::debug(
            "scheduled task {} (command={}, GPU event={})",
            record.id(),
            record.task->name(),
            record.output_events
        );

        record.queue->scheduled_job(record);
        record.status = TaskRecord::Status::WaitingForCompletion;

        for (const auto& succ : record.successors) {
            succ->input_events.insert(record.output_events);
            succ->predecessors_pending -= 1;
            enqueue_if_ready(&record, succ);
        }
    }

    if (record.status == TaskRecord::Status::WaitingForCompletion) {
        if (!m_stream_manager->is_ready(record.output_events)) {
            return Poll::Pending;
        }

        spdlog::debug("completed task {} (command={})", record.id(), record.task->name());
        m_tasks.erase(record.event_id);
        record.queue->completed_job(record);
        record.status = TaskRecord::Status::Completed;
        record.task = nullptr;
    }

    KMM_ASSERT(record.status == TaskRecord::Status::Completed);
    return Poll::Ready;
}

}  // namespace kmm
