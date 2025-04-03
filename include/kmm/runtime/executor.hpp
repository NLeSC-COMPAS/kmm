#pragma once

#include <future>

#include "kmm/core/commands.hpp"
#include "kmm/core/resource.hpp"
#include "kmm/runtime/buffer_registry.hpp"
#include "kmm/runtime/device_resources.hpp"
#include "kmm/runtime/memory_manager.hpp"
#include "kmm/runtime/scheduler.hpp"
#include "kmm/runtime/stream_manager.hpp"
#include "kmm/utils/poll.hpp"

namespace kmm {

class Executor {
    KMM_NOT_COPYABLE_OR_MOVABLE(Executor)

  public:
    class Job {
        KMM_NOT_COPYABLE_OR_MOVABLE(Job)

      public:
        Job(TaskHandle task) : m_task(task) {}
        virtual ~Job() = default;
        virtual Poll poll(Executor& executor) = 0;

        //      private:
        TaskHandle m_task;
        std::unique_ptr<Job> next = nullptr;
    };

    Executor(
        std::shared_ptr<DeviceResources> m_device_manager,
        std::shared_ptr<DeviceStreamManager> stream_manager,
        std::shared_ptr<BufferRegistry> buffer_registry,
        std::shared_ptr<Scheduler> scheduler,
        bool debug_mode
    );

    ~Executor();

    void execute_task(TaskHandle task, DeviceEventSet dependencies);
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
    void insert_job(std::unique_ptr<Job> job);
    void execute_task(TaskHandle task, const CommandEmpty& command, DeviceEventSet dependencies);

    void execute_task(TaskHandle task, const CommandExecute& command, DeviceEventSet dependencies);

    void execute_task(TaskHandle task, const CommandCopy& command, DeviceEventSet dependencies);

    void execute_task(
        TaskHandle task,
        const CommandReduction& command,
        DeviceEventSet dependencies
    );

    void execute_task(TaskHandle task, const CommandFill& command, DeviceEventSet dependencies);

    std::shared_ptr<DeviceResources> m_device_resources;
    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    std::shared_ptr<BufferRegistry> m_buffer_registry;
    std::shared_ptr<Scheduler> m_scheduler;

    std::unique_ptr<Job> m_jobs_head = nullptr;
    Job* m_jobs_tail = nullptr;
    bool m_debug_mode = false;
};

}  // namespace kmm
