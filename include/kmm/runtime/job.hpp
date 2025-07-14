#pragma once

#include <future>

#include "kmm/runtime/buffer_registry.hpp"
#include "kmm/runtime/device_resources.hpp"
#include "kmm/runtime/scheduler.hpp"
#include "kmm/utils/poll.hpp"

namespace kmm {

class Executor;

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

std::unique_ptr<Job> build_job_for_command(
    TaskHandle task,
    const Command& command,
    DeviceEventSet dependencies
);

class MergeJob: public Job {
  public:
    MergeJob(TaskHandle task) : Job(std::move(task)) {}

    MergeJob(TaskHandle task, DeviceEventSet dependencies) :
        Job(task),
        m_dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) final;

  private:
    DeviceEventSet m_dependencies;
};

class DeleteBufferJob: public Job {
  public:
    DeleteBufferJob(TaskHandle task, BufferId buffer_id, DeviceEventSet dependencies) :
        Job(task),
        m_buffer_id(buffer_id),
        m_dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) final;

  private:
    BufferId m_buffer_id;
    DeviceEventSet m_dependencies;
};

class HostJob: public Job {
  public:
    HostJob(TaskHandle task, std::vector<BufferRequirement> buffers, DeviceEventSet dependencies) :
        Job(task),
        m_buffers(std::move(buffers)),
        m_dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) final;

  protected:
    virtual std::future<void> submit(Executor& executor, std::vector<BufferAccessor> accessors) = 0;

  private:
    enum struct Status {
        Init,
        PollingBuffers,
        PollingDependencies,
        Running,
        Completing,
        Completed
    };

    Status m_status = Status::Init;
    std::future<void> m_future;
    std::vector<BufferRequirement> m_buffers;
    BufferRequestList m_requests;
    DeviceEventSet m_dependencies;
};

class ExecuteHostJob: public HostJob {
  public:
    ExecuteHostJob(
        TaskHandle task,
        ComputeTask* compute_task,
        std::vector<BufferRequirement> buffers,
        DeviceEventSet dependencies
    ) :
        HostJob(task, std::move(buffers), std::move(dependencies)),
        m_task(compute_task) {}

    std::future<void> submit(Executor& executor, std::vector<BufferAccessor> accessors) override;

  private:
    ComputeTask* m_task;
};

class CopyHostJob: public HostJob {
  public:
    CopyHostJob(
        TaskHandle task,
        BufferId src_buffer,
        BufferId dst_buffer,
        CopyDef definition,
        DeviceEventSet dependencies
    ) :
        HostJob(
            task,
            {BufferRequirement {src_buffer, MemoryId::host(), AccessMode::Read},
             BufferRequirement {dst_buffer, MemoryId::host(), AccessMode::ReadWrite}},
            std::move(dependencies)
        ),
        m_copy(definition) {}

    std::future<void> submit(Executor& executor, std::vector<BufferAccessor> accessors) override;

  private:
    CopyDef m_copy;
};

class ReductionHostJob: public HostJob {
  public:
    ReductionHostJob(
        TaskHandle task,
        BufferId src_buffer,
        BufferId dst_buffer,
        ReductionDef definition,
        DeviceEventSet dependencies
    ) :
        HostJob(
            task,
            {BufferRequirement {src_buffer, MemoryId::host(), AccessMode::Read},
             BufferRequirement {dst_buffer, MemoryId::host(), AccessMode::ReadWrite}},
            std::move(dependencies)
        ),
        m_reduction(definition) {}

    std::future<void> submit(Executor& executor, std::vector<BufferAccessor> accessors) override;

  private:
    ReductionDef m_reduction;
};

class FillHostJob: public HostJob {
  public:
    FillHostJob(
        TaskHandle task,
        BufferId dst_buffer,
        FillDef definition,
        DeviceEventSet dependencies
    ) :
        HostJob(
            task,
            {BufferRequirement {dst_buffer, MemoryId::host(), AccessMode::ReadWrite}},
            std::move(dependencies)
        ),
        m_fill(definition) {}

    std::future<void> submit(Executor& executor, std::vector<BufferAccessor> accessors) override;

  private:
    FillDef m_fill;
};

class DeviceJob: public Job, public DeviceResourceOperation {
  public:
    DeviceJob(
        TaskHandle task,
        ResourceId resource_id,
        std::vector<BufferRequirement> buffers,
        DeviceEventSet dependencies
    ) :
        Job(task),
        m_resource(resource_id),
        m_buffers(std::move(buffers)),
        m_dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) final;

  private:
    enum struct Status {
        Init,
        PollingBuffers,
        PollingDependencies,
        Running,
        Completing,
        Completed
    };

    Status m_status = Status::Init;
    ResourceId m_resource;
    std::vector<BufferRequirement> m_buffers;
    BufferRequestList m_requests;
    DeviceEvent m_execution_event;
    DeviceEventSet m_dependencies;
    DeviceEventSet m_local_dependencies;
};

class ExecuteDeviceJob: public DeviceJob {
  public:
    ExecuteDeviceJob(
        TaskHandle task,
        ResourceId device_id,
        ComputeTask* compute_task,
        std::vector<BufferRequirement> buffers,
        DeviceEventSet dependencies
    ) :
        DeviceJob(task, device_id, std::move(buffers), std::move(dependencies)),
        m_task(compute_task) {}

    void execute(DeviceResource& device, std::vector<BufferAccessor> accessors) final;

  private:
    ComputeTask* m_task;
};

class CopyDeviceJob: public DeviceJob {
  public:
    CopyDeviceJob(
        TaskHandle id,
        DeviceId device_id,
        BufferId src_buffer,
        BufferId dst_buffer,
        CopyDef definition,
        DeviceEventSet dependencies
    ) :
        DeviceJob(
            id,
            device_id,
            {BufferRequirement {src_buffer, device_id, AccessMode::Read},
             BufferRequirement {dst_buffer, device_id, AccessMode::ReadWrite}},
            std::move(dependencies)
        ),
        m_copy(definition) {}

    void execute(DeviceResource& device, std::vector<BufferAccessor> accessors) final;

  private:
    CopyDef m_copy;
};

class ReductionDeviceJob: public DeviceJob {
  public:
    ReductionDeviceJob(
        TaskHandle id,
        DeviceId device_id,
        BufferId src_buffer,
        BufferId dst_buffer,
        ReductionDef definition,
        DeviceEventSet dependencies
    ) :
        DeviceJob(
            id,
            device_id,
            {BufferRequirement {src_buffer, device_id, AccessMode::Read},
             BufferRequirement {dst_buffer, device_id, AccessMode::ReadWrite}},
            std::move(dependencies)
        ),
        m_reduction(std::move(definition)) {}

    void execute(DeviceResource& device, std::vector<BufferAccessor> accessors) final;

  private:
    ReductionDef m_reduction;
};

class FillDeviceJob: public DeviceJob {
  public:
    FillDeviceJob(
        TaskHandle id,
        DeviceId device_id,
        BufferId dst_buffer,
        FillDef definition,
        DeviceEventSet dependencies
    ) :
        DeviceJob(
            id,
            device_id,
            {BufferRequirement {dst_buffer, device_id, AccessMode::ReadWrite}},
            std::move(dependencies)
        ),
        m_fill(std::move(definition)) {}

    void execute(DeviceResource& device, std::vector<BufferAccessor> accessors) final;

  private:
    FillDef m_fill;
};

class PrefetchJob: public Job {
  public:
    PrefetchJob(
        TaskHandle task,
        BufferId buffer_id,
        MemoryId memory_id,
        DeviceEventSet dependencies
    ) :
        Job(task),
        m_buffers {{buffer_id, memory_id, AccessMode::Read}},
        m_dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) final;

  private:
    enum struct Status { Init, Polling, Completing, Completed };

    Status m_status = Status::Init;
    std::vector<BufferRequirement> m_buffers;
    BufferRequestList m_requests;
    DeviceEventSet m_dependencies;
};

}  // namespace kmm