#pragma once

#include <future>

#include "kmm/core/commands.hpp"
#include "kmm/core/resource.hpp"
#include "kmm/runtime/buffer_registry.hpp"
#include "kmm/runtime/device_resources.hpp"
#include "kmm/runtime/memory_manager.hpp"
#include "kmm/runtime/scheduler.hpp"
#include "kmm/runtime/stream_manager.hpp"
#include "kmm/runtime/task.hpp"
#include "kmm/utils/poll.hpp"

namespace kmm {

class Executor {
    KMM_NOT_COPYABLE_OR_MOVABLE(Executor)

  public:
    Executor(
        std::shared_ptr<DeviceResources> device_resources,
        std::shared_ptr<DeviceStreamManager> stream_manager,
        std::shared_ptr<BufferRegistry> buffer_registry,
        std::shared_ptr<Scheduler> scheduler,
        bool debug_mode
    );

    ~Executor();

    void submit(EventId event_id, Command&& command, EventList dependencies);
    bool is_completed(EventId event_id);

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

    Scheduler& scheduler() {
        return *m_scheduler;
    }

  private:
    void execute_task(TaskHandle task, DeviceEventSet dependencies);

    std::shared_ptr<DeviceResources> m_device_resources;
    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    std::shared_ptr<BufferRegistry> m_buffer_registry;
    std::shared_ptr<Scheduler> m_scheduler;

    std::unique_ptr<Task> m_jobs_head = nullptr;
    Task* m_jobs_tail = nullptr;
    bool m_debug_mode = false;
};

}  // namespace kmm
