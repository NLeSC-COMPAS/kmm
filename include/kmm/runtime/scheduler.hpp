#pragma once

#include <future>

#include "kmm/core/commands.hpp"
#include "kmm/core/resource.hpp"
#include "kmm/runtime/buffer_registry.hpp"
#include "kmm/runtime/device_resources.hpp"
#include "kmm/runtime/memory_manager.hpp"
#include "kmm/runtime/stream_manager.hpp"
#include "kmm/runtime/task.hpp"
#include "kmm/utils/poll.hpp"

namespace kmm {

struct SchedulerQueue;

class TaskRecord {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskRecord)

    friend class Scheduler;
    enum struct Status {  //
        Init,
        AwaitingDependencies,
        ReadyToStart,
        Running,
        WaitingForCompletion,
        Completed
    };

  public:
    TaskRecord(EventId event_id, std::unique_ptr<Task> task) :
        event_id(event_id),
        task(std::move(task)) {}

    EventId id() const {
        return event_id;
    }

  private:
    EventId event_id;
    Status status = Status::Init;
    SchedulerQueue* queue = nullptr;

    EventList predecessors;
    size_t predecessors_pending = 0;
    DeviceEventSet input_events;

    small_vector<std::shared_ptr<TaskRecord>, 4> successors;
    DeviceEventSet output_events;

    std::shared_ptr<TaskRecord> next = nullptr;
    std::unique_ptr<Task> task = nullptr;
};

class Scheduler {
    KMM_NOT_COPYABLE_OR_MOVABLE(Scheduler)

  public:
    Scheduler(
        std::shared_ptr<DeviceResources> device_resources,
        std::shared_ptr<DeviceStreamManager> stream_manager,
        std::shared_ptr<BufferRegistry> buffer_registry,
        bool debug_mode
    );

    ~Scheduler();

    void submit(EventId event_id, std::unique_ptr<Task> task, EventList dependencies);
    bool is_completed(EventId event_id) const;
    bool is_idle() const;
    void make_progress();

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
    static size_t determine_queue_id(const Task&);
    void enqueue_if_ready(const TaskRecord* predecessor, const std::shared_ptr<TaskRecord>& task);
    std::shared_ptr<TaskRecord> dequeue_ready_task();
    void start_task(std::shared_ptr<TaskRecord> record);
    Poll poll_completion(TaskRecord& record);

    std::shared_ptr<DeviceResources> m_device_resources;
    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    std::shared_ptr<BufferRegistry> m_buffer_registry;

    std::vector<SchedulerQueue> m_ready_queues;
    std::unordered_map<EventId, std::shared_ptr<TaskRecord>> m_tasks;
    std::shared_ptr<TaskRecord> m_running_head = nullptr;
    TaskRecord* m_running_tail = nullptr;
    bool m_debug_mode = false;
};

}  // namespace kmm
