#pragma once

#include <future>

#include "kmm/core/commands.hpp"
#include "kmm/core/resource.hpp"
#include "kmm/runtime/buffer_registry.hpp"
#include "kmm/runtime/device_resources.hpp"
#include "kmm/runtime/memory_manager.hpp"
#include "kmm/runtime/stream_manager.hpp"
#include "kmm/utils/poll.hpp"

namespace kmm {

struct SchedulerQueue;
class Executor;

class TaskRecord {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskRecord)

    friend class Executor;
    enum Status { Init, AwaitingDependencies, ReadyToSubmit, Submitted, Executing, Completed };

  public:
    TaskRecord(Command&& command);

    EventId id() const {
        return event_id;
    }

    const Command& get_command() const {
        return command;
    }

  private:
    Status status = Status::Init;
    EventId event_id = EventId::invalid();
    Command command;
    size_t queue_id = 0;
    DeviceEvent output_event;
    small_vector<std::shared_ptr<TaskRecord>, 4> successors;

    EventList predecessors;
    size_t predecessors_pending = 0;
    DeviceEventSet input_events;
};

using TaskHandle = std::shared_ptr<TaskRecord>;

class Task {
    KMM_NOT_COPYABLE_OR_MOVABLE(Task)

  public:
    Task(TaskHandle task, DeviceEventSet dependencies) :
        m_task(std::move(task)),
        m_dependencies(std::move(dependencies)) {}
    virtual ~Task() = default;
    virtual Poll poll(Executor& executor) = 0;

    //      private:
    TaskHandle m_task;
    DeviceEventSet m_dependencies;
    std::unique_ptr<Task> next = nullptr;
};

class Executor {
    KMM_NOT_COPYABLE_OR_MOVABLE(Executor)

  public:
    Executor(
        std::shared_ptr<DeviceResources> device_resources,
        std::shared_ptr<DeviceStreamManager> stream_manager,
        std::shared_ptr<BufferRegistry> buffer_registry,
        bool debug_mode
    );

    ~Executor();

    void submit(EventId event_id, Command&& command, EventList dependencies);
    bool is_completed(EventId event_id) const;

    bool is_idle() const;
    void make_progress();

    void mark_as_scheduled(TaskHandle task, DeviceEvent event);

    DeviceResources& devices() {
        return *m_device_resources;
    }

    DeviceStreamManager& streams() {
        return *m_stream_manager;
    }

    BufferRegistry& buffers() {
        return *m_buffer_registry;
    }

  private:
    void execute_task(TaskHandle task, DeviceEventSet dependencies);
    static size_t determine_queue_id(const Command& cmd);
    void enqueue_if_ready(const TaskRecord* predecessor, const TaskHandle& task);
    void mark_as_completed(TaskHandle task);

    std::shared_ptr<DeviceResources> m_device_resources;
    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    std::shared_ptr<BufferRegistry> m_buffer_registry;

    std::vector<SchedulerQueue> m_queues;
    std::unordered_map<EventId, TaskHandle> m_tasks;
    std::unique_ptr<Task> m_jobs_head = nullptr;
    Task* m_jobs_tail = nullptr;
    bool m_debug_mode = false;
};

}  // namespace kmm
