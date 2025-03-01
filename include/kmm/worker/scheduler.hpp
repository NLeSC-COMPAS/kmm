#pragma once

#include <deque>
#include <unordered_map>

#include "kmm/dag/commands.hpp"
#include "kmm/worker/stream_manager.hpp"

namespace kmm {

class Scheduler;

struct Task {
    friend class Scheduler;
    enum Status { Init, AwaitingDependencies, ReadyToSubmit, Submitted, Executing, Completed };

  public:
    Task(EventId event_id, Command&& command, EventList&& dependencies) :
        event_id(event_id),
        command(std::move(command)),
        dependencies(std::move(dependencies)) {}

    EventId id() const {
        return event_id;
    }

    const Command& get_command() const {
        return command;
    }

  private:
    EventId event_id;
    Status status = Status::Init;
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
    struct Queue;

    Scheduler(size_t num_devices);
    ~Scheduler();

    void submit(std::shared_ptr<Task> task);
    std::optional<std::shared_ptr<Task>> pop_ready(DeviceEventSet* deps_out);

    void mark_as_scheduled(EventId id, DeviceEvent event);
    void mark_as_completed(EventId id);

    bool is_completed(EventId id) const;
    bool is_idle() const;

  private:
    size_t determine_queue_id(const Command& cmd);
    void enqueue_if_ready(const Task* predecessor, const std::shared_ptr<Task>& node);

    std::vector<Queue> m_queues;
    std::unordered_map<EventId, std::shared_ptr<Task>> m_events;
};

}  // namespace kmm