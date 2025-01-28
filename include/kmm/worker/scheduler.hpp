#pragma once

#include <deque>
#include <unordered_map>

#include "kmm/worker/commands.hpp"
#include "kmm/worker/device_stream_manager.hpp"

namespace kmm {

class Scheduler {
    KMM_NOT_COPYABLE_OR_MOVABLE(Scheduler)

  public:
    struct Queue;
    struct TaskNode {
        friend class Scheduler;

        enum Status { AwaitingDependencies, ReadyToSubmit, Submitted, Executing, Completed };

        TaskNode(EventId event_id, Command&& command) :
            event_id(event_id),
            command(std::move(command)) {}

        EventId id() const {
            return event_id;
        }

        const Command& get_command() const {
            return command;
        }

      private:
        EventId event_id;
        Status status = Status::AwaitingDependencies;
        Command command;
        size_t queue_id = 0;
        DeviceEvent execution_event;
        small_vector<std::shared_ptr<TaskNode>, 4> successors;
        DeviceEventSet dependency_events;
        size_t dependencies_pending = 0;
    };

    Scheduler(size_t num_devices);
    ~Scheduler();

    void submit(std::vector<CommandNode> commands);
    void submit(EventId event_id, Command command, const EventList& deps = {});
    std::optional<std::shared_ptr<TaskNode>> pop_ready(DeviceEventSet* deps_out);

    void mark_as_scheduled(EventId id, DeviceEvent event);
    void mark_as_completed(EventId id);

    bool is_completed(EventId id) const;
    bool is_idle() const;

  private:
    size_t determine_queue_id(const Command& cmd);
    void enqueue_if_ready(const TaskNode* predecessor, const std::shared_ptr<TaskNode>& node);

    std::vector<Queue> m_queues;
    std::unordered_map<EventId, std::shared_ptr<TaskNode>> m_events;
};

using SchedulerNode = std::shared_ptr<Scheduler::TaskNode>;

}  // namespace kmm