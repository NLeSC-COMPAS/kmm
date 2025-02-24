#pragma once

#include <future>

#include "worker_state.hpp"

#include "kmm/core/resource.hpp"
#include "kmm/utils/error_ptr.hpp"
#include "kmm/utils/poll.hpp"
#include "kmm/worker/buffer_registry.hpp"
#include "kmm/worker/memory_manager.hpp"
#include "kmm/worker/scheduler.hpp"
#include "kmm/worker/stream_manager.hpp"

namespace kmm {

struct DeviceState {
    KMM_NOT_COPYABLE_OR_MOVABLE(DeviceState)

  public:
    GPUContextHandle context;
    DeviceStream stream;
    DeviceEvent last_event;
    DeviceResource device;

    DeviceState(DeviceId id, GPUContextHandle context, DeviceStreamManager& stream_manager) :
        context(context),
        stream(stream_manager.create_stream(context)),
        device(DeviceInfo(id, context), context, stream_manager.get(stream)) {}
};

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
        std::vector<GPUContextHandle> contexts,
        std::shared_ptr<DeviceStreamManager> stream_manager,
        std::shared_ptr<BufferRegistry> buffer_registry,
        std::shared_ptr<Scheduler> scheduler,
        bool debug_mode
    );

    ~Executor();

    bool is_idle() const;
    void make_progress(WorkerState& worker);

    DeviceState& device_state(DeviceId id, const DeviceEventSet& hint_deps = {});

    void execute_command(EventId id, const Command& command, DeviceEventSet dependencies);

    Scheduler& scheduler() {
        return *m_scheduler;
    }

    BufferRegistry& buffer_registry() {
        return *m_buffer_registry;
    }

    DeviceStreamManager& stream_manager() {
        return *m_stream_manager;
    }

  private:
    void insert_job(std::unique_ptr<Job> job);
    void execute_command(EventId id, const CommandExecute& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandCopy& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandReduction& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandFill& command, DeviceEventSet dependencies);

    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    std::shared_ptr<BufferRegistry> m_buffer_registry;
    std::shared_ptr<Scheduler> m_scheduler;

    std::unique_ptr<Job> m_jobs_head = nullptr;
    Job* m_jobs_tail = nullptr;
    std::vector<std::unique_ptr<DeviceState>> m_devices;
    bool m_debug_mode = false;
};

}  // namespace kmm