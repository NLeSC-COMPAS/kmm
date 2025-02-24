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
    MergeJob(EventId id) : m_id(id) {}

    MergeJob(EventId id, DeviceEventSet dependencies) :
        m_id(id),
        m_dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) final {
        if (!executor.stream_manager().is_ready(m_dependencies)) {
            return Poll::Pending;
        }

        executor.scheduler().mark_as_completed(m_id);
        return Poll::Ready;
    }

  private:
    EventId m_id;
    DeviceEventSet m_dependencies;
};

class HostJob: public Executor::Job {
  public:
    HostJob(EventId id, std::vector<BufferRequirement> buffers, DeviceEventSet dependencies) :
        m_id(id),
        m_buffers(std::move(buffers)),
        m_dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) final {
        if (m_status == Status::Init) {
            try {
                m_requests = executor.buffer_registry().create_requests(m_buffers);
                m_status = Status::Waiting;
            } catch (const std::exception& e) {
                executor.buffer_registry().poison_all(m_buffers, PoisonException(m_id, e));
                m_status = Status::Completing;
            }
        }

        if (m_status == Status::Waiting) {
            try {
                if (executor.buffer_registry().poll_requests(m_requests, &m_dependencies)
                    == Poll::Pending) {
                    return Poll::Pending;
                }

                if (!executor.stream_manager().is_ready(m_dependencies)) {
                    return Poll::Pending;
                }

                m_future = submit(executor, executor.buffer_registry().access_requests(m_requests));
                m_status = Status::Running;
            } catch (const std::exception& e) {
                executor.buffer_registry().poison_all(m_buffers, PoisonException(m_id, e));
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
                executor.buffer_registry().poison_all(m_buffers, PoisonException(m_id, e));
                m_status = Status::Completing;
            }
        }

        if (m_status == Status::Completing) {
            executor.buffer_registry().release_requests(m_requests);
            executor.scheduler().mark_as_completed(m_id);
            m_status = Status::Completed;
        }

        return Poll::Ready;
    }

  protected:
    virtual std::future<void> submit(Executor& executor, std::vector<BufferAccessor> accessors) = 0;

  private:
    enum struct Status { Init, Waiting, Running, Completing, Completed };

    Status m_status = Status::Init;
    EventId m_id;
    std::future<void> m_future;
    std::vector<BufferRequirement> m_buffers;
    BufferRequestList m_requests;
    DeviceEventSet m_dependencies;
};

class DeviceJob: public Executor::Job {
  public:
    DeviceJob(
        EventId id,
        DeviceId device_id,
        std::vector<BufferRequirement> buffers,
        DeviceEventSet dependencies
    ) :
        m_id(id),
        m_device_id(device_id),
        m_buffers(std::move(buffers)),
        m_dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) final {
        if (m_status == Status::Init) {
            try {
                m_requests = executor.buffer_registry().create_requests(m_buffers);
                m_status = Status::Pending;
            } catch (const std::exception& e) {
                executor.buffer_registry().poison_all(m_buffers, PoisonException(m_id, e));
                m_status = Status::Completing;
            }
        }

        if (m_status == Status::Pending) {
            try {
                if (executor.buffer_registry().poll_requests(m_requests, &m_dependencies)
                    == Poll::Pending) {
                    return Poll::Pending;
                }

                auto& state = executor.device_state(m_device_id, m_dependencies);

                try {
                    GPUContextGuard guard {state.context};
                    executor.stream_manager().wait_for_events(state.stream, m_dependencies);
                    submit(state.device, executor.buffer_registry().access_requests(m_requests));
                    m_event = executor.stream_manager().record_event(state.stream);
                } catch (const std::exception& e) {
                    try {
                        executor.stream_manager().wait_until_ready(state.stream);
                    } catch (...) {
                        KMM_PANIC("fatal error: ", e.what());
                    }

                    throw;
                }

                state.last_event = m_event;
                executor.scheduler().mark_as_scheduled(m_id, m_event);
                executor.buffer_registry().release_requests(m_requests, m_event);
                m_status = Status::Running;
            } catch (const std::exception& e) {
                executor.buffer_registry().poison_all(m_buffers, PoisonException(m_id, e));
                executor.buffer_registry().release_requests(m_requests);
                m_status = Status::Completing;
            }
        }

        if (m_status == Status::Running) {
            if (!executor.stream_manager().is_ready(m_event)) {
                return Poll::Pending;
            }

            m_status = Status::Completing;
        }

        if (m_status == Status::Completing) {
            executor.scheduler().mark_as_completed(m_id);
            m_status = Status::Completed;
        }

        return Poll::Ready;
    }

  protected:
    virtual void submit(DeviceResource& device, std::vector<BufferAccessor> accessors) = 0;

  private:
    enum struct Status { Init, Pending, Running, Completing, Completed };

    Status m_status = Status::Init;
    EventId m_id;
    DeviceId m_device_id;
    std::vector<BufferRequirement> m_buffers;
    BufferRequestList m_requests;
    DeviceEvent m_event;
    DeviceEventSet m_dependencies;
};

class ExecuteHostJob: public HostJob {
  public:
    ExecuteHostJob(
        EventId id,
        std::shared_ptr<Task> task,
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
    std::shared_ptr<Task> m_task;
};

class CopyHostJob: public HostJob {
  public:
    CopyHostJob(
        EventId id,
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
        EventId id,
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
    FillHostJob(EventId id, BufferId dst_buffer, FillDef definition, DeviceEventSet dependencies) :
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
        EventId id,
        DeviceId device_id,
        std::shared_ptr<Task> task,
        std::vector<BufferRequirement> buffers,
        DeviceEventSet dependencies
    ) :
        DeviceJob(id, device_id, std::move(buffers), std::move(dependencies)),
        m_task(std::move(task)) {}

    void submit(DeviceResource& device, std::vector<BufferAccessor> accessors) {
        auto context = TaskContext {std::move(accessors)};
        m_task->execute(device, context);
    }

  private:
    std::shared_ptr<Task> m_task;
};

class CopyDeviceJob: public DeviceJob {
  public:
    CopyDeviceJob(
        EventId id,
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

    void submit(DeviceResource& device, std::vector<BufferAccessor> accessors) {
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
        EventId id,
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

    void submit(DeviceResource& device, std::vector<BufferAccessor> accessors) {
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
        EventId id,
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

    void submit(DeviceResource& device, std::vector<BufferAccessor> accessors) {
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
    PrefetchJob(EventId id, BufferId buffer_id, MemoryId memory_id, DeviceEventSet dependencies) :
        m_id(id),
        m_buffers {{buffer_id, memory_id, AccessMode::Read}},
        m_dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) final {
        if (m_status == Status::Init) {
            m_requests = executor.buffer_registry().create_requests(m_buffers);
            m_status = Status::Waiting;
        }

        if (m_status == Status::Waiting) {
            if (executor.buffer_registry().poll_requests(m_requests, &m_dependencies)
                == Poll::Pending) {
                return Poll::Pending;
            }

            if (!executor.stream_manager().is_ready(m_dependencies)) {
                return Poll::Pending;
            }

            executor.buffer_registry().release_requests(m_requests);
            executor.scheduler().mark_as_completed(m_id);
            m_status = Status::Completed;
        }

        return Poll::Ready;
    }

  private:
    enum struct Status { Init, Waiting, Completed };

    Status m_status = Status::Init;
    EventId m_id;
    std::vector<BufferRequirement> m_buffers;
    BufferRequestList m_requests;
    DeviceEventSet m_dependencies;
};

Executor::Executor(
    std::vector<GPUContextHandle> contexts,
    std::shared_ptr<DeviceStreamManager> stream_manager,
    std::shared_ptr<BufferRegistry> buffer_registry,
    std::shared_ptr<Scheduler> scheduler,
    bool debug_mode
) :
    m_stream_manager(stream_manager),
    m_buffer_registry(buffer_registry),
    m_scheduler(scheduler),
    m_debug_mode(debug_mode) {
    for (size_t i = 0; i < contexts.size(); i++) {
        m_devices.emplace_back(
            std::make_unique<DeviceState>(DeviceId(i), contexts[i], *stream_manager)
        );
    }
}

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

DeviceState& Executor::device_state(DeviceId id, const DeviceEventSet& hint_deps) {
    KMM_ASSERT(id < m_devices.size());
    return *m_devices.at(id);
}

void Executor::insert_job(std::unique_ptr<Job> job) {
    if (auto* old_tail = std::exchange(m_jobs_tail, job.get())) {
        old_tail->next = std::move(job);
    } else {
        m_jobs_head = std::move(job);
    }
}

void Executor::execute_command(EventId id, const Command& command, DeviceEventSet dependencies) {
    if (std::get_if<CommandEmpty>(&command) != nullptr) {
        insert_job(std::make_unique<MergeJob>(id, std::move(dependencies)));

    } else if (const auto* e = std::get_if<CommandBufferCreate>(&command)) {
        m_buffer_registry->add(e->id, e->layout);
        execute_command(id, CommandEmpty {}, std::move(dependencies));

    } else if (const auto* e = std::get_if<CommandBufferDelete>(&command)) {
        m_buffer_registry->remove(e->id);
        execute_command(id, CommandEmpty {}, std::move(dependencies));

    } else if (std::get_if<CommandPrefetch>(&command) != nullptr) {
        execute_command(id, CommandEmpty {}, std::move(dependencies));

    } else if (const auto* e = std::get_if<CommandExecute>(&command)) {
        execute_command(id, *e, std::move(dependencies));

    } else if (const auto* e = std::get_if<CommandCopy>(&command)) {
        execute_command(id, *e, std::move(dependencies));

    } else if (const auto* e = std::get_if<CommandReduction>(&command)) {
        execute_command(id, *e, std::move(dependencies));

    } else if (const auto* e = std::get_if<CommandFill>(&command)) {
        execute_command(id, *e, std::move(dependencies));

    } else {
        KMM_PANIC("could not handle unknown command: ", command);
    }
}

void Executor::execute_command(
    EventId id,
    const CommandExecute& command,
    DeviceEventSet dependencies
) {
    auto proc = command.processor_id;

    if (proc.is_device()) {
        insert_job(std::make_unique<ExecuteDeviceJob>(
            id,
            proc.as_device(),
            command.task,
            command.buffers,
            std::move(dependencies)
        ));
    } else {
        insert_job(std::make_unique<ExecuteHostJob>(
            id,
            command.task,
            command.buffers,
            std::move(dependencies)
        ));
    }
}

void Executor::execute_command(
    EventId id,
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

void Executor::execute_command(
    EventId id,
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

void Executor::execute_command(
    EventId id,
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