#pragma once

#include <future>

#include "kmm/core/resource.hpp"
#include "kmm/dag/commands.hpp"
#include "kmm/utils/error_ptr.hpp"
#include "kmm/utils/poll.hpp"
#include "kmm/worker/buffer_registry.hpp"
#include "kmm/worker/memory_manager.hpp"
#include "kmm/worker/resource_manager.hpp"
#include "kmm/worker/scheduler.hpp"
#include "kmm/worker/stream_manager.hpp"

namespace kmm {

class Executor {
    KMM_NOT_COPYABLE_OR_MOVABLE(Executor)

  public:
    class Job {
        KMM_NOT_COPYABLE_OR_MOVABLE(Job)

      public:
        Job() = default;
        virtual ~Job() = default;
        virtual Poll poll(Executor& executor) = 0;

        //      private:
        std::unique_ptr<Job> next = nullptr;
    };

    Executor(
        std::shared_ptr<DeviceResourceManager> m_device_manager,
        std::shared_ptr<DeviceStreamManager> stream_manager,
        std::shared_ptr<BufferRegistry> buffer_registry,
        std::shared_ptr<Scheduler> scheduler,
        bool debug_mode
    );

    ~Executor();

    void execute_command(EventId id, const Command& command, DeviceEventSet dependencies);
    bool is_idle() const;
    void make_progress();

    DeviceResourceManager& devices() {
        return *m_device_manager;
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
    void execute_command(EventId id, const CommandExecute& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandCopy& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandReduction& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandFill& command, DeviceEventSet dependencies);

    std::shared_ptr<DeviceResourceManager> m_device_manager;
    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    std::shared_ptr<BufferRegistry> m_buffer_registry;
    std::shared_ptr<Scheduler> m_scheduler;

    std::unique_ptr<Job> m_jobs_head = nullptr;
    Job* m_jobs_tail = nullptr;
    bool m_debug_mode = false;
};

}  // namespace kmm