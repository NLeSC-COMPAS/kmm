#pragma once

#include <future>

#include "kmm/runtime/buffer_registry.hpp"
#include "kmm/runtime/device_resources.hpp"
#include "kmm/runtime/executor.hpp"
#include "kmm/utils/poll.hpp"

namespace kmm {

class MergeTask: public Task {
  public:
    MergeTask(TaskHandle task) : Task(std::move(task), {}) {}

    MergeTask(TaskHandle task, DeviceEventSet dependencies) : Task(task, std::move(dependencies)) {}

    Poll poll(Executor& executor) final;

  private:
    //DeviceEventSet m_dependencies;
};

class DeleteBufferTask: public Task {
  public:
    DeleteBufferTask(TaskHandle task, BufferId buffer_id, DeviceEventSet dependencies) :
        Task(task, std::move(dependencies)),
        m_buffer_id(buffer_id) {}

    Poll poll(Executor& executor) final;

  private:
    BufferId m_buffer_id;
    //DeviceEventSet m_dependencies;
};

class HostTask: public Task {
  public:
    HostTask(TaskHandle task, std::vector<BufferRequirement> buffers, DeviceEventSet dependencies) :
        Task(task, std::move(dependencies)),
        m_buffers(std::move(buffers)) {}

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
    //DeviceEventSet m_dependencies;
};

class ExecuteHostTask: public HostTask {
  public:
    ExecuteHostTask(
        TaskHandle task,
        ComputeTask* compute_task,
        std::vector<BufferRequirement> buffers,
        DeviceEventSet dependencies
    ) :
        HostTask(task, std::move(buffers), std::move(dependencies)),
        m_task(compute_task) {}

    std::future<void> submit(Executor& executor, std::vector<BufferAccessor> accessors) override;

  private:
    ComputeTask* m_task;
};

class CopyHostTask: public HostTask {
  public:
    CopyHostTask(
        TaskHandle task,
        BufferId src_buffer,
        BufferId dst_buffer,
        CopyDef definition,
        DeviceEventSet dependencies
    ) :
        HostTask(
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

class ReductionHostTask: public HostTask {
  public:
    ReductionHostTask(
        TaskHandle task,
        BufferId src_buffer,
        BufferId dst_buffer,
        ReductionDef definition,
        DeviceEventSet dependencies
    ) :
        HostTask(
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

class FillHostTask: public HostTask {
  public:
    FillHostTask(
        TaskHandle task,
        BufferId dst_buffer,
        FillDef definition,
        DeviceEventSet dependencies
    ) :
        HostTask(
            task,
            {BufferRequirement {dst_buffer, MemoryId::host(), AccessMode::ReadWrite}},
            std::move(dependencies)
        ),
        m_fill(definition) {}

    std::future<void> submit(Executor& executor, std::vector<BufferAccessor> accessors) override;

  private:
    FillDef m_fill;
};

class DeviceTask: public Task, public DeviceResourceOperation {
  public:
    DeviceTask(
        TaskHandle task,
        ResourceId resource_id,
        std::vector<BufferRequirement> buffers,
        DeviceEventSet dependencies
    ) :
        Task(task, std::move(dependencies)),
        m_resource(resource_id),
        m_buffers(std::move(buffers)) {}

    Poll poll(Executor& executor) final;

  private:
    enum struct Status {
        Init,
        PollingBuffers,
        PollingDependencies,
        Running,
        WaitingForDeviceEvent,
        Completing,
        Completed
    };

    Status m_status = Status::Init;
    ResourceId m_resource;
    std::vector<BufferRequirement> m_buffers;
    BufferRequestList m_requests;
    DeviceEvent m_execution_event;
    //DeviceEventSet m_dependencies;
    DeviceEventSet m_local_dependencies;
};

class ExecuteDeviceTask: public DeviceTask {
  public:
    ExecuteDeviceTask(
        TaskHandle task,
        ResourceId device_id,
        ComputeTask* compute_task,
        std::vector<BufferRequirement> buffers,
        DeviceEventSet dependencies
    ) :
        DeviceTask(task, device_id, std::move(buffers), std::move(dependencies)),
        m_task(compute_task) {}

    void execute(DeviceResource& device, std::vector<BufferAccessor> accessors) final;

  private:
    ComputeTask* m_task;
};

class CopyDeviceTask: public DeviceTask {
  public:
    CopyDeviceTask(
        TaskHandle id,
        DeviceId device_id,
        BufferId src_buffer,
        BufferId dst_buffer,
        CopyDef definition,
        DeviceEventSet dependencies
    ) :
        DeviceTask(
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

class ReductionDeviceTask: public DeviceTask {
  public:
    ReductionDeviceTask(
        TaskHandle id,
        DeviceId device_id,
        BufferId src_buffer,
        BufferId dst_buffer,
        ReductionDef definition,
        DeviceEventSet dependencies
    ) :
        DeviceTask(
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

class FillDeviceTask: public DeviceTask {
  public:
    FillDeviceTask(
        TaskHandle id,
        DeviceId device_id,
        BufferId dst_buffer,
        FillDef definition,
        DeviceEventSet dependencies
    ) :
        DeviceTask(
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

class PrefetchTask: public Task {
  public:
    PrefetchTask(
        TaskHandle task,
        BufferId buffer_id,
        MemoryId memory_id,
        DeviceEventSet dependencies
    ) :
        Task(task, std::move(dependencies)),
        m_buffers {{buffer_id, memory_id, AccessMode::Read}} {}

    Poll poll(Executor& executor) final;

  private:
    enum struct Status { Init, Polling, Completing, Completed };

    Status m_status = Status::Init;
    std::vector<BufferRequirement> m_buffers;
    BufferRequestList m_requests;
    //DeviceEventSet m_dependencies;
};

std::unique_ptr<Task> build_job_for_command(
    TaskHandle task,
    const Command& command,
    DeviceEventSet dependencies
);

}  // namespace kmm