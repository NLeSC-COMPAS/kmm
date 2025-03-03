#include "spdlog/spdlog.h"

#include "kmm/memops/gpu_copy.hpp"
#include "kmm/memops/gpu_fill.hpp"
#include "kmm/memops/gpu_reduction.hpp"
#include "kmm/memops/host_copy.hpp"
#include "kmm/memops/host_fill.hpp"
#include "kmm/memops/host_reduction.hpp"
#include "kmm/worker/executor.hpp"

namespace kmm {

class MergeJob: public Executor::Job {
  public:
    MergeJob(std::shared_ptr<Task> task) : m_task(std::move(task)) {}

    MergeJob(std::shared_ptr<Task> task, DeviceEventSet dependencies) :
        m_task(task),
        m_dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) final {
        if (!executor.streams().is_ready(m_dependencies)) {
            return Poll::Pending;
        }

        executor.scheduler().mark_as_completed(m_task);
        return Poll::Ready;
    }

  private:
    std::shared_ptr<Task> m_task;
    DeviceEventSet m_dependencies;
};

class HostJob: public Executor::Job {
  public:
    HostJob(
        std::shared_ptr<Task> task,
        std::vector<BufferRequirement> buffers,
        DeviceEventSet dependencies
    ) :
        m_task(task),
        m_buffers(std::move(buffers)),
        m_dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) final {
        if (m_status == Status::Init) {
            try {
                m_requests = executor.buffers().create_requests(m_buffers);
                m_status = Status::Waiting;
            } catch (const std::exception& e) {
                executor.buffers().poison_all(m_buffers, PoisonException(m_task->id(), e));
                m_status = Status::Completing;
            }
        }

        if (m_status == Status::Waiting) {
            try {
                if (executor.buffers().poll_requests(m_requests, m_dependencies) == Poll::Pending) {
                    return Poll::Pending;
                }

                if (!executor.streams().is_ready(m_dependencies)) {
                    return Poll::Pending;
                }

                m_future = submit(executor, executor.buffers().access_requests(m_requests));
                m_status = Status::Running;
            } catch (const std::exception& e) {
                executor.buffers().poison_all(m_buffers, PoisonException(m_task->id(), e));
                m_status = Status::Completing;
            }
        }

        if (m_status == Status::Running) {
            try {
                if (m_future.wait_for(std::chrono::seconds(0)) == std::future_status::timeout) {
                    return Poll::Pending;
                }

                m_status = Status::Completing;
            } catch (const std::exception& e) {
                executor.buffers().poison_all(m_buffers, PoisonException(m_task->id(), e));
                m_status = Status::Completing;
            }
        }

        if (m_status == Status::Completing) {
            executor.buffers().release_requests(m_requests);
            executor.scheduler().mark_as_completed(m_task);
            m_status = Status::Completed;
        }

        return Poll::Ready;
    }

  protected:
    virtual std::future<void> submit(Executor& executor, std::vector<BufferAccessor> accessors) = 0;

  private:
    enum struct Status { Init, Waiting, Running, Completing, Completed };

    Status m_status = Status::Init;
    std::shared_ptr<Task> m_task;
    std::future<void> m_future;
    std::vector<BufferRequirement> m_buffers;
    BufferRequestList m_requests;
    DeviceEventSet m_dependencies;
};

class DeviceJob: public Executor::Job, public DeviceResourceOperation {
  public:
    DeviceJob(
        std::shared_ptr<Task> id,
        DeviceId device_id,
        std::vector<BufferRequirement> buffers,
        DeviceEventSet dependencies
    ) :
        m_task(id),
        m_device_id(device_id),
        m_buffers(std::move(buffers)),
        m_dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) final {
        if (m_status == Status::Init) {
            try {
                m_requests = executor.buffers().create_requests(m_buffers);
                m_status = Status::Pending;
            } catch (const std::exception& e) {
                executor.buffers().poison_all(m_buffers, PoisonException(m_task->id(), e));
                m_status = Status::Completing;
            }
        }

        if (m_status == Status::Pending) {
            try {
                if (executor.buffers().poll_requests(m_requests, m_dependencies) == Poll::Pending) {
                    return Poll::Pending;
                }

                executor.devices().submit(
                    m_device_id,
                    m_dependencies,
                    *this,
                    executor.buffers().access_requests(m_requests)
                );

                executor.scheduler().mark_as_scheduled(m_task, m_event);
                executor.buffers().release_requests(m_requests, m_event);
                m_status = Status::Running;
            } catch (const std::exception& e) {
                executor.buffers().poison_all(m_buffers, PoisonException(m_task->id(), e));
                executor.buffers().release_requests(m_requests);
                m_status = Status::Completing;
            }
        }

        if (m_status == Status::Running) {
            if (!executor.streams().is_ready(m_event)) {
                return Poll::Pending;
            }

            m_status = Status::Completing;
        }

        if (m_status == Status::Completing) {
            executor.scheduler().mark_as_completed(m_task);
            m_status = Status::Completed;
        }

        return Poll::Ready;
    }

  private:
    enum struct Status { Init, Pending, Running, Completing, Completed };

    Status m_status = Status::Init;
    std::shared_ptr<Task> m_task;
    DeviceId m_device_id;
    std::vector<BufferRequirement> m_buffers;
    BufferRequestList m_requests;
    DeviceEvent m_event;
    DeviceEventSet m_dependencies;
};

class ExecuteHostJob: public HostJob {
  public:
    ExecuteHostJob(
        std::shared_ptr<Task> id,
        std::shared_ptr<ComputeTask> task,
        std::vector<BufferRequirement> buffers,
        DeviceEventSet dependencies
    ) :
        HostJob(id, std::move(buffers), std::move(dependencies)),
        m_task(std::move(task)) {}

    std::future<void> submit(Executor& executor, std::vector<BufferAccessor> accessors) override {
        return std::async(std::launch::async, [=] {
            auto host = HostResource {};
            auto context = TaskContext {std::move(accessors)};
            m_task->execute(host, context);
        });
    }

  private:
    std::shared_ptr<ComputeTask> m_task;
};

class CopyHostJob: public HostJob {
  public:
    CopyHostJob(
        std::shared_ptr<Task> id,
        BufferId src_buffer,
        BufferId dst_buffer,
        CopyDef definition,
        DeviceEventSet dependencies
    ) :
        HostJob(
            id,
            {BufferRequirement {src_buffer, MemoryId::host(), AccessMode::Read},
             BufferRequirement {dst_buffer, MemoryId::host(), AccessMode::ReadWrite}},
            std::move(dependencies)
        ),
        m_copy(definition) {}

    std::future<void> submit(Executor& executor, std::vector<BufferAccessor> accessors) override {
        KMM_ASSERT(accessors[0].layout.size_in_bytes >= m_copy.minimum_source_bytes_needed());
        KMM_ASSERT(accessors[1].layout.size_in_bytes >= m_copy.minimum_destination_bytes_needed());
        KMM_ASSERT(accessors[1].is_writable);

        return std::async(std::launch::async, [=] {
            execute_copy(accessors[0].address, accessors[1].address, m_copy);
        });
    }

  private:
    CopyDef m_copy;
};

class ReductionHostJob: public HostJob {
  public:
    ReductionHostJob(
        std::shared_ptr<Task> id,
        BufferId src_buffer,
        BufferId dst_buffer,
        ReductionDef definition,
        DeviceEventSet dependencies
    ) :
        HostJob(
            id,
            {BufferRequirement {src_buffer, MemoryId::host(), AccessMode::Read},
             BufferRequirement {dst_buffer, MemoryId::host(), AccessMode::ReadWrite}},
            std::move(dependencies)
        ),
        m_reduction(definition) {}

    std::future<void> submit(Executor& executor, std::vector<BufferAccessor> accessors) override {
        return std::async(std::launch::async, [=] {
            execute_reduction(accessors[0].address, accessors[1].address, m_reduction);
        });
    }

  private:
    ReductionDef m_reduction;
};

class FillHostJob: public HostJob {
  public:
    FillHostJob(
        std::shared_ptr<Task> id,
        BufferId dst_buffer,
        FillDef definition,
        DeviceEventSet dependencies
    ) :
        HostJob(
            id,
            {BufferRequirement {dst_buffer, MemoryId::host(), AccessMode::ReadWrite}},
            std::move(dependencies)
        ),
        m_fill(definition) {}

    std::future<void> submit(Executor& executor, std::vector<BufferAccessor> accessors) override {
        return std::async(std::launch::async, [=] { execute_fill(accessors[0].address, m_fill); });
    }

  private:
    FillDef m_fill;
};

class ExecuteDeviceJob: public DeviceJob {
  public:
    ExecuteDeviceJob(
        std::shared_ptr<Task> id,
        DeviceId device_id,
        std::shared_ptr<ComputeTask> task,
        std::vector<BufferRequirement> buffers,
        DeviceEventSet dependencies
    ) :
        DeviceJob(id, device_id, std::move(buffers), std::move(dependencies)),
        m_task(std::move(task)) {}

    void execute(DeviceResource& device, std::vector<BufferAccessor> accessors) final {
        auto context = TaskContext {std::move(accessors)};
        m_task->execute(device, context);
    }

  private:
    std::shared_ptr<ComputeTask> m_task;
};

class CopyDeviceJob: public DeviceJob {
  public:
    CopyDeviceJob(
        std::shared_ptr<Task> id,
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

    void execute(DeviceResource& device, std::vector<BufferAccessor> accessors) final {
        KMM_ASSERT(accessors[0].layout.size_in_bytes >= m_copy.minimum_source_bytes_needed());
        KMM_ASSERT(accessors[1].layout.size_in_bytes >= m_copy.minimum_destination_bytes_needed());
        KMM_ASSERT(accessors[1].is_writable);

        execute_gpu_d2d_copy_async(
            device,
            reinterpret_cast<GPUdeviceptr>(accessors[0].address),
            reinterpret_cast<GPUdeviceptr>(accessors[1].address),
            m_copy
        );
    }

  private:
    CopyDef m_copy;
};

class ReductionDeviceJob: public DeviceJob {
  public:
    ReductionDeviceJob(
        std::shared_ptr<Task> id,
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

    void execute(DeviceResource& device, std::vector<BufferAccessor> accessors) final {
        execute_gpu_reduction_async(
            device,
            reinterpret_cast<GPUdeviceptr>(accessors[0].address),
            reinterpret_cast<GPUdeviceptr>(accessors[1].address),
            m_reduction
        );
    }

  private:
    ReductionDef m_reduction;
};

class FillDeviceJob: public DeviceJob {
  public:
    FillDeviceJob(
        std::shared_ptr<Task> id,
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

    void execute(DeviceResource& device, std::vector<BufferAccessor> accessors) final {
        execute_gpu_fill_async(
            device,
            reinterpret_cast<GPUdeviceptr>(accessors[0].address),
            m_fill
        );
    }

  private:
    FillDef m_fill;
};

class PrefetchJob: public Executor::Job {
  public:
    PrefetchJob(
        std::shared_ptr<Task> task,
        BufferId buffer_id,
        MemoryId memory_id,
        DeviceEventSet dependencies
    ) :
        m_task(task),
        m_buffers {{buffer_id, memory_id, AccessMode::Read}},
        m_dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) final {
        if (m_status == Status::Init) {
            m_requests = executor.buffers().create_requests(m_buffers);
            m_status = Status::Polling;
        }

        if (m_status == Status::Polling) {
            if (executor.buffers().poll_requests(m_requests, m_dependencies) == Poll::Pending) {
                return Poll::Pending;
            }

            executor.buffers().release_requests(m_requests);
            m_status = Status::Completing;
        }

        if (m_status == Status::Completing) {
            if (!executor.streams().is_ready(m_dependencies)) {
                return Poll::Pending;
            }

            executor.scheduler().mark_as_completed(m_task);
            m_status = Status::Completed;
        }

        return Poll::Ready;
    }

  private:
    enum struct Status { Init, Polling, Completing, Completed };

    Status m_status = Status::Init;
    std::shared_ptr<Task> m_task;
    std::vector<BufferRequirement> m_buffers;
    BufferRequestList m_requests;
    DeviceEventSet m_dependencies;
};

Executor::Executor(
    std::shared_ptr<DeviceResourceManager> device_manager,
    std::shared_ptr<DeviceStreamManager> stream_manager,
    std::shared_ptr<BufferRegistry> buffer_registry,
    std::shared_ptr<Scheduler> scheduler,
    bool debug_mode
) :
    m_device_manager(device_manager),
    m_stream_manager(stream_manager),
    m_buffer_registry(buffer_registry),
    m_scheduler(scheduler),
    m_debug_mode(debug_mode) {}

Executor::~Executor() {}

bool Executor::is_idle() const {
    return m_jobs_head == nullptr;
}

void Executor::make_progress() {
    Job* prev = nullptr;
    std::unique_ptr<Job>* current_ptr = &m_jobs_head;

    while (auto* current = current_ptr->get()) {
        // In debug mode, only poll the head task (prev == nullptr)
        bool should_poll = !m_debug_mode || prev == nullptr;

        if (should_poll && current->poll(*this) == Poll::Ready) {
            *current_ptr = std::move(current->next);
        } else {
            prev = current;
            current_ptr = &current->next;
        }
    }

    m_jobs_tail = prev;
}

void Executor::insert_job(std::unique_ptr<Job> job) {
    if (auto* old_tail = std::exchange(m_jobs_tail, job.get())) {
        old_tail->next = std::move(job);
    } else {
        m_jobs_head = std::move(job);
    }
}

void Executor::execute_task(std::shared_ptr<Task> task, DeviceEventSet dependencies) {
    const Command& command = task->get_command();

    if (std::get_if<CommandEmpty>(&command) != nullptr) {
        execute_task(task, CommandEmpty {}, std::move(dependencies));

    } else if (const auto* e = std::get_if<CommandBufferCreate>(&command)) {
        m_buffer_registry->add(e->id, e->layout);
        execute_task(task, CommandEmpty {}, std::move(dependencies));

    } else if (const auto* e = std::get_if<CommandBufferDelete>(&command)) {
        m_buffer_registry->remove(e->id);
        execute_task(task, CommandEmpty {}, std::move(dependencies));

    } else if (const auto* e = std::get_if<CommandPrefetch>(&command)) {
        auto job = std::make_unique<PrefetchJob>(
            task,
            e->buffer_id,
            e->memory_id,
            std::move(dependencies)
        );

        insert_job(std::move(job));

    } else if (const auto* e = std::get_if<CommandExecute>(&command)) {
        execute_task(task, *e, std::move(dependencies));

    } else if (const auto* e = std::get_if<CommandCopy>(&command)) {
        execute_task(task, *e, std::move(dependencies));

    } else if (const auto* e = std::get_if<CommandReduction>(&command)) {
        execute_task(task, *e, std::move(dependencies));

    } else if (const auto* e = std::get_if<CommandFill>(&command)) {
        execute_task(task, *e, std::move(dependencies));

    } else {
        KMM_PANIC("could not handle unknown command: ", command);
    }
}

void Executor::execute_task(
    std::shared_ptr<Task> task,
    const CommandEmpty& command,
    DeviceEventSet dependencies
) {
    if (m_stream_manager->is_ready(dependencies)) {
        m_scheduler->mark_as_completed(task);
    } else {
        insert_job(std::make_unique<MergeJob>(task, std::move(dependencies)));
    }
}

void Executor::execute_task(
    std::shared_ptr<Task> task,
    const CommandExecute& command,
    DeviceEventSet dependencies
) {
    auto proc = command.processor_id;

    if (proc.is_device()) {
        insert_job(std::make_unique<ExecuteDeviceJob>(
            task,
            proc.as_device(),
            command.task,
            command.buffers,
            std::move(dependencies)
        ));
    } else {
        insert_job(std::make_unique<ExecuteHostJob>(
            task,
            command.task,
            command.buffers,
            std::move(dependencies)
        ));
    }
}

void Executor::execute_task(
    std::shared_ptr<Task> id,
    const CommandCopy& command,
    DeviceEventSet dependencies
) {
    auto src_mem = command.src_memory;
    auto dst_mem = command.dst_memory;

    if (src_mem.is_host() && dst_mem.is_host()) {
        insert_job(std::make_unique<CopyHostJob>(
            id,
            command.src_buffer,
            command.dst_buffer,
            command.definition,
            std::move(dependencies)
        ));
    } else if (dst_mem.is_device()) {
        insert_job(std::make_unique<CopyDeviceJob>(
            id,
            dst_mem.as_device(),
            command.src_buffer,
            command.dst_buffer,
            command.definition,
            std::move(dependencies)
        ));
    } else if (src_mem.is_device()) {
        KMM_TODO();
    }
}

void Executor::execute_task(
    std::shared_ptr<Task> id,
    const CommandReduction& command,
    DeviceEventSet dependencies
) {
    auto memory_id = command.memory_id;

    if (memory_id.is_device()) {
        insert_job(std::make_unique<ReductionDeviceJob>(
            id,
            memory_id.as_device(),
            command.src_buffer,
            command.dst_buffer,
            std::move(command.definition),
            std::move(dependencies)
        ));
    } else {
        insert_job(std::make_unique<ReductionHostJob>(
            id,
            command.src_buffer,
            command.dst_buffer,
            std::move(command.definition),
            std::move(dependencies)
        ));
    }
}

void Executor::execute_task(
    std::shared_ptr<Task> id,
    const CommandFill& command,
    DeviceEventSet dependencies
) {
    auto memory_id = command.memory_id;

    if (memory_id.is_device()) {
        insert_job(std::make_unique<FillDeviceJob>(
            id,
            memory_id.as_device(),
            command.dst_buffer,
            std::move(command.definition),
            std::move(dependencies)
        ));
    } else {
        insert_job(std::make_unique<FillHostJob>(
            id,
            command.dst_buffer,
            std::move(command.definition),
            std::move(dependencies)
        ));
    }
}

}  // namespace kmm