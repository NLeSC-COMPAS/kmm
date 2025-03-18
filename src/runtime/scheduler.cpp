#include <queue>

#include "spdlog/spdlog.h"

#include "kmm/runtime/scheduler.hpp"

namespace kmm {

static constexpr size_t NUM_DEFAULT_QUEUES = 3;
static constexpr size_t QUEUE_MISC = 0;
static constexpr size_t QUEUE_BUFFERS = 1;
static constexpr size_t QUEUE_HOST = 2;
static constexpr size_t QUEUE_DEVICES = 3;

struct QueueSlot {
    TaskHandle inner;
};

bool operator<(const QueueSlot& lhs, const QueueSlot& rhs) {
    return lhs.inner->id() > rhs.inner->id();
}

struct SchedulerQueue {
    size_t max_concurrent_jobs = std::numeric_limits<size_t>::max();
    size_t num_jobs_active = 0;
    std::priority_queue<QueueSlot> tasks;

    void push_job(const Task* predecessor, TaskHandle task);
    bool pop_job(TaskHandle& task_out);
    void scheduled_job(TaskHandle task);
    void completed_job(TaskHandle task);
};

void Scheduler::enqueue_if_ready(const Task* predecessor, const TaskHandle& task) {
    if (task->status != Task::Status::AwaitingDependencies) {
        return;
    }

    if (task->dependencies_pending > 0) {
        return;
    }

    task->status = Task::Status::ReadyToSubmit;
    m_queues.at(task->queue_id).push_job(predecessor, task);
}

void SchedulerQueue::push_job(const Task* predecessor, TaskHandle task) {
    this->tasks.push(QueueSlot {task});
}

bool SchedulerQueue::pop_job(TaskHandle& task_out) {
    if (num_jobs_active >= max_concurrent_jobs) {
        return false;
    }

    if (tasks.empty()) {
        return false;
    }

    num_jobs_active++;
    task_out = std::move(tasks.top()).inner;
    tasks.pop();

    return true;
}

void SchedulerQueue::scheduled_job(TaskHandle task) {
    // Nothing to do after scheduling
}

void SchedulerQueue::completed_job(TaskHandle task) {
    num_jobs_active--;
}

Scheduler::Scheduler(size_t num_devices) {
    m_queues.resize(NUM_DEFAULT_QUEUES + num_devices);

    for (size_t i = 0; i < num_devices; i++) {
        m_queues[QUEUE_DEVICES + i].max_concurrent_jobs = 5;
    }
}

Scheduler::~Scheduler() = default;

void Scheduler::submit(EventId event_id, Command command, EventList dependencies) {
    auto node = std::make_shared<Task>(event_id, std::move(command), std::move(dependencies));

    spdlog::debug(
        "submit task {} (command={}, dependencies={})",
        node->event_id,
        node->command,
        node->dependencies
    );

    size_t num_pending = node->dependencies.size();
    DeviceEventSet dependency_events;

    for (EventId dep_id : node->dependencies) {
        auto it = m_tasks.find(dep_id);

        if (it == m_tasks.end()) {
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

    KMM_ASSERT(node->status == Task::Status::Init);
    node->status = Task::Status::AwaitingDependencies;
    node->queue_id = determine_queue_id(node->command);
    node->dependencies_pending = num_pending;
    node->dependency_events = std::move(dependency_events);
    enqueue_if_ready(nullptr, node);

    m_tasks.emplace(event_id, std::move(node));
}

std::optional<TaskHandle> Scheduler::pop_ready(DeviceEventSet* deps_out) {
    TaskHandle result;

    for (auto& q : m_queues) {
        if (q.pop_job(result)) {
            spdlog::debug(
                "scheduling task {} (command={}, GPU deps={})",
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

void Scheduler::mark_as_scheduled(TaskHandle task, DeviceEvent event) {
    spdlog::debug("scheduled task {} (command={}, GPU event={})", task->id(), task->command, event);

    KMM_ASSERT(task->status == Task::Status::Submitted);
    task->status = Task::Status::Executing;
    task->execution_event = event;

    for (const auto& succ : task->successors) {
        succ->dependency_events.insert(event);
        succ->dependencies_pending -= 1;
        enqueue_if_ready(task.get(), succ);
    }

    m_queues.at(task->queue_id).scheduled_job(task);
}

void Scheduler::mark_as_completed(TaskHandle task) {
    spdlog::debug("completed task {} (command={})", task->id(), task->command);

    if (task->status == Task::Status::Submitted) {
        for (const auto& succ : task->successors) {
            succ->dependencies_pending -= 1;
            enqueue_if_ready(task.get(), succ);
        }

        task->status = Task::Status::Executing;
        m_queues.at(task->queue_id).scheduled_job(task);
    }

    KMM_ASSERT(task->status == Task::Status::Executing);
    task->status = Task::Status::Completed;
    m_tasks.erase(task->event_id);
    m_queues.at(task->queue_id).completed_job(task);
}

bool Scheduler::is_completed(EventId id) const {
    return m_tasks.find(id) == m_tasks.end();
}

bool Scheduler::is_idle() const {
    return m_tasks.empty();
}

size_t Scheduler::determine_queue_id(const Command& cmd) {
    if (const auto* p = std::get_if<CommandExecute>(&cmd)) {
        if (p->processor_id.is_device()) {
            return QUEUE_DEVICES + p->processor_id.as_device();
        } else {
            return QUEUE_HOST;
        }
    } else if (std::holds_alternative<CommandBufferDelete>(cmd) || std::holds_alternative<CommandEmpty>(cmd)) {
        return QUEUE_BUFFERS;
    } else {
        return QUEUE_MISC;
    }
}

Task::Task(EventId event_id, Command&& command, EventList&& dependencies) :
    event_id(event_id),
    command(std::move(command)),
    dependencies(std::move(dependencies)) {}

}  // namespace kmm
