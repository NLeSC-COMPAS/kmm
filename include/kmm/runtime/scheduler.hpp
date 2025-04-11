#pragma once

#include <deque>
#include <unordered_map>

#include "kmm/core/commands.hpp"
#include "kmm/runtime/stream_manager.hpp"

namespace kmm {

class Task;
using TaskHandle = std::shared_ptr<Task>;
class Scheduler;
struct SchedulerQueue;

class Scheduler {
    KMM_NOT_COPYABLE_OR_MOVABLE(Scheduler)

  public:
    Scheduler(size_t num_devices);
    ~Scheduler();

    void submit(EventId event_id, Command&& command, EventList dependencies);
    std::optional<TaskHandle> pop_ready(DeviceEventSet* deps_out);

    void mark_as_scheduled(TaskHandle task, DeviceEvent event);
    void mark_as_completed(TaskHandle task);

    bool is_completed(EventId id) const;
    bool is_idle() const;

  private:
    static size_t determine_queue_id(const Command& cmd);
    void enqueue_if_ready(const Task* predecessor, const TaskHandle& task);

    std::vector<SchedulerQueue> m_queues;
    std::unordered_map<EventId, TaskHandle> m_tasks;
};

class Task {
    KMM_NOT_COPYABLE_OR_MOVABLE(Task)

    friend class Scheduler;
    enum Status { Init, AwaitingDependencies, ReadyToSubmit, Submitted, Executing, Completed };

  public:
    Task(EventId event_id, Command&& command, EventList&& dependencies);

    EventId id() const {
        return event_id;
    }

    const Command& get_command() const {
        return command;
    }

  private:
    Status status = Status::Init;
    EventId event_id;
    Command command;
    size_t queue_id = 0;
    DeviceEvent execution_event;
    small_vector<TaskHandle, 4> successors;

    EventList dependencies;
    size_t dependencies_pending = 0;
    DeviceEventSet dependency_events;
};

}  // namespace kmm
