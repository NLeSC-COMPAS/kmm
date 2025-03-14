#pragma once

#include <deque>
#include <unordered_map>

#include "kmm/dag/commands.hpp"
#include "kmm/runtime/stream_manager.hpp"

namespace kmm {

class Scheduler;

struct Task {
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
    small_vector<std::shared_ptr<Task>, 4> successors;

    EventList dependencies;
    size_t dependencies_pending = 0;
    DeviceEventSet dependency_events;
};

class Scheduler {
    KMM_NOT_COPYABLE_OR_MOVABLE(Scheduler)

  public:
    Scheduler(size_t num_devices);
    ~Scheduler();

    void submit(EventId event_id, Command command, EventList dependencies);
    std::optional<std::shared_ptr<Task>> pop_ready(DeviceEventSet* deps_out);

    void mark_as_scheduled(std::shared_ptr<Task> task, DeviceEvent event);
    void mark_as_completed(std::shared_ptr<Task> task);

    bool is_completed(EventId id) const;
    bool is_idle() const;

  private:
    size_t determine_queue_id(const Command& cmd);
    void enqueue_if_ready(const Task* predecessor, const std::shared_ptr<Task>& node);

    struct Queue;
    std::vector<Queue> m_queues;
    std::unordered_map<EventId, std::shared_ptr<Task>> m_tasks;
};

}  // namespace kmm
