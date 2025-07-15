#include <queue>

#include "spdlog/spdlog.h"

#include "kmm/runtime/executor.hpp"
#include "kmm/runtime/task.hpp"

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

Executor::Executor(
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
    m_queues.resize(NUM_DEFAULT_QUEUES + num_devices);

    for (size_t i = 0; i < num_devices; i++) {
        m_queues[QUEUE_DEVICES + i].max_concurrent_jobs = 5;
    }
}

Executor::~Executor() {}

void Executor::submit(EventId event_id, Command&& command, EventList dependencies) {
    auto task = std::make_shared<TaskRecord>(std::move(command));

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
            dependency_events.insert(dep->output_event);
        }

        if (dep->status == TaskRecord::Status::Completed) {
            num_pending--;
        }
    }

    task->status = TaskRecord::Status::AwaitingDependencies;
    task->event_id = event_id;
    task->predecessors = std::move(dependencies);
    task->queue_id = determine_queue_id(task->command);
    task->predecessors_pending = num_pending;
    task->input_events = std::move(dependency_events);
    enqueue_if_ready(nullptr, task);

    m_tasks.emplace(event_id, std::move(task));
}

bool Executor::is_completed(EventId event_id) const {
    return m_tasks.find(event_id) == m_tasks.end();
}

bool Executor::is_idle() const {
    return m_jobs_head == nullptr && m_tasks.empty();
}

void Executor::make_progress() {
    Task* prev = nullptr;
    std::unique_ptr<Task>* current_ptr = &m_jobs_head;

    while (auto* current = current_ptr->get()) {
        // In debug mode, only poll the head task (prev == nullptr)
        bool should_poll = !m_debug_mode || prev == nullptr;

        if (should_poll && current->poll(*this) == Poll::Ready) {
            this->mark_as_completed(current->m_task);
            *current_ptr = std::move(current->next);
        } else {
            prev = current;
            current_ptr = &current->next;
        }
    }

    m_jobs_tail = prev;

    bool job_found = true;

    while (job_found) {
        TaskHandle result;
        job_found = false;

        for (auto& q : m_queues) {
            if (q.pop_job(result)) {
                execute_task(result, std::move(result->input_events));
                job_found = true;
                break;
            }
        }
    }
}

void Executor::enqueue_if_ready(const TaskRecord* predecessor, const TaskHandle& task) {
    if (task->status != TaskRecord::Status::AwaitingDependencies) {
        return;
    }

    if (task->predecessors_pending > 0) {
        return;
    }

    task->status = TaskRecord::Status::ReadyToSubmit;
    m_queues.at(task->queue_id).push_job(predecessor, task);
}

void Executor::mark_as_scheduled(TaskHandle task, DeviceEvent event) {
    spdlog::debug("scheduled task {} (command={}, GPU event={})", task->id(), task->command, event);

    KMM_ASSERT(task->status == TaskRecord::Status::Submitted);
    task->status = TaskRecord::Status::Executing;
    task->output_event = event;

    for (const auto& succ : task->successors) {
        succ->input_events.insert(event);
        succ->predecessors_pending -= 1;
        enqueue_if_ready(task.get(), succ);
    }

    m_queues.at(task->queue_id).scheduled_job(task);
}

void Executor::mark_as_completed(TaskHandle task) {
    spdlog::debug("completed task {} (command={})", task->id(), task->command);

    if (task->status == TaskRecord::Status::Submitted) {
        for (const auto& succ : task->successors) {
            succ->predecessors_pending -= 1;
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

size_t Executor::determine_queue_id(const Command& cmd) {
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

void Executor::execute_task(TaskHandle task, DeviceEventSet dependencies) {
    KMM_ASSERT(task->status == TaskRecord::Status::ReadyToSubmit);
    task->status = TaskRecord::Status::Submitted;

    spdlog::debug(
        "scheduling task {} (command={}, GPU deps={})",
        task->id(),
        task->command,
        task->input_events
    );

    const Command& command = task->get_command();

    // For the case of an empty task, we provide a fast path if all device events are done
    if (m_stream_manager->is_ready(dependencies) && std::holds_alternative<CommandEmpty>(command)) {
        this->mark_as_completed(task);
        return;
    }

    auto job = build_job_for_command(task, command, dependencies);

    if (auto* old_tail = std::exchange(m_jobs_tail, job.get())) {
        old_tail->next = std::move(job);
    } else {
        m_jobs_head = std::move(job);
    }
}

TaskRecord::TaskRecord(Command&& command) : command(std::move(command)) {}

}  // namespace kmm
