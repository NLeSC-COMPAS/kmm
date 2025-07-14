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

    void push_job(const TaskRecord* predecessor, TaskHandle task);
    bool pop_job(TaskHandle& task_out);
    void scheduled_job(TaskHandle task);
    void completed_job(TaskHandle task);
};

void Scheduler::enqueue_if_ready(const TaskRecord* predecessor, const TaskHandle& task) {
    if (task->status != TaskRecord::Status::AwaitingDependencies) {
        return;
    }

    if (task->dependencies_pending > 0) {
        return;
    }

    task->status = TaskRecord::Status::ReadyToSubmit;
    m_queues.at(task->queue_id).push_job(predecessor, task);
}

void SchedulerQueue::push_job(const TaskRecord* predecessor, TaskHandle task) {
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

void Scheduler::submit(std::shared_ptr<TaskRecord> task, EventId event_id, EventList dependencies) {
    KMM_ASSERT(task->status == TaskRecord::Status::Init);

    spdlog::debug(
        "submit task {} (command={}, dependencies={})",
        event_id,
        task->command,
        dependencies
    );

    size_t num_pending = dependencies.size();
    DeviceEventSet dependency_events;

    for (EventId dep_id : dependencies) {
        auto it = m_tasks.find(dep_id);

        if (it == m_tasks.end()) {
            num_pending--;
            continue;
        }

        auto& dep = it->second;
        dep->successors.push_back(task);

        if (dep->status == TaskRecord::Status::Executing) {
            num_pending--;
            dependency_events.insert(dep->execution_event);
        }

        if (dep->status == TaskRecord::Status::Completed) {
            num_pending--;
        }
    }

    task->status = TaskRecord::Status::AwaitingDependencies;
    task->event_id = event_id;
    task->dependencies = std::move(dependencies);
    task->queue_id = determine_queue_id(task->command);
    task->dependencies_pending = num_pending;
    task->dependency_events = std::move(dependency_events);
    enqueue_if_ready(nullptr, task);

    m_tasks.emplace(event_id, std::move(task));
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

            KMM_ASSERT(result->status == TaskRecord::Status::ReadyToSubmit);
            result->status = TaskRecord::Status::Submitted;
            *deps_out = std::move(result->dependency_events);
            return result;
        }
    }

    return std::nullopt;
}

void Scheduler::mark_as_scheduled(TaskHandle task, DeviceEvent event) {
    spdlog::debug("scheduled task {} (command={}, GPU event={})", task->id(), task->command, event);

    KMM_ASSERT(task->status == TaskRecord::Status::Submitted);
    task->status = TaskRecord::Status::Executing;
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

    if (task->status == TaskRecord::Status::Submitted) {
        for (const auto& succ : task->successors) {
            succ->dependencies_pending -= 1;
            enqueue_if_ready(task.get(), succ);
        }

        task->status = TaskRecord::Status::Executing;
        m_queues.at(task->queue_id).scheduled_job(task);
    }

    KMM_ASSERT(task->status == TaskRecord::Status::Executing);
    task->status = TaskRecord::Status::Completed;
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

TaskRecord::TaskRecord(Command&& command) : command(std::move(command)) {}

}  // namespace kmm
