#include "kmm/memops/gpu_copy.hpp"
#include "kmm/memops/gpu_fill.hpp"
#include "kmm/memops/gpu_reduction.hpp"
#include "kmm/memops/host_copy.hpp"
#include "kmm/memops/host_fill.hpp"
#include "kmm/memops/host_reduction.hpp"
#include "kmm/runtime/executor.hpp"
#include "kmm/runtime/job.hpp"

namespace kmm {

static PoisonException make_poison_exception(TaskHandle task, const std::exception& error) {
    if (const auto* reason = dynamic_cast<const PoisonException*>(&error)) {
        return *reason;
    }

    return fmt::format("task {} failed due to error: {}", task->id(), error.what());
}

Poll MergeJob::poll(Executor& executor) {
    if (!executor.streams().is_ready(m_dependencies)) {
        return Poll::Pending;
    }

    executor.scheduler().mark_as_completed(m_task);
    return Poll::Ready;
}

Poll DeleteBufferJob::poll(Executor& executor) {
    if (!executor.streams().is_ready(m_dependencies)) {
        return Poll::Pending;
    }

    executor.buffers().remove(m_buffer_id);
    executor.scheduler().mark_as_completed(m_task);
    return Poll::Ready;
}

Poll HostJob::poll(Executor& executor) {
    if (m_status == Status::Init) {
        try {
            m_requests = executor.buffers().create_requests(m_buffers);
            m_status = Status::PollingBuffers;
        } catch (const std::exception& e) {
            executor.buffers().poison_all(m_buffers, make_poison_exception(m_task, e));
            m_status = Status::Completing;
        }
    }

    if (m_status == Status::PollingBuffers) {
        try {
            if (executor.buffers().poll_requests(m_requests, m_dependencies) == Poll::Pending) {
                return Poll::Pending;
            }

            m_status = Status::PollingDependencies;
        } catch (const std::exception& e) {
            executor.buffers().poison_all(m_buffers, make_poison_exception(m_task, e));
            m_status = Status::Completing;
        }
    }

    if (m_status == Status::PollingDependencies) {
        try {
            if (!executor.streams().is_ready(m_dependencies)) {
                return Poll::Pending;
            }

            m_future = submit(executor, executor.buffers().access_requests(m_requests));
            m_status = Status::Running;
        } catch (const std::exception& e) {
            executor.buffers().poison_all(m_buffers, make_poison_exception(m_task, e));
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
            executor.buffers().poison_all(m_buffers, make_poison_exception(m_task, e));
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

std::future<void> ExecuteHostJob::submit(
    Executor& executor,
    std::vector<BufferAccessor> accessors
) {
    auto* task = m_task;

    return std::async(std::launch::async, [=] {
        auto host = HostResource {};
        auto context = TaskContext {std::move(accessors)};
        task->execute(host, context);
    });
}

std::future<void> CopyHostJob::submit(Executor& executor, std::vector<BufferAccessor> accessors) {
    KMM_ASSERT(accessors[0].layout.size_in_bytes >= m_copy.minimum_source_bytes_needed());
    KMM_ASSERT(accessors[1].layout.size_in_bytes >= m_copy.minimum_destination_bytes_needed());
    KMM_ASSERT(accessors[1].is_writable);

    return std::async(std::launch::async, [=] {
        execute_copy(accessors[0].address, accessors[1].address, m_copy);
    });
}

std::future<void> ReductionHostJob::submit(
    Executor& executor,
    std::vector<BufferAccessor> accessors
) {
    return std::async(std::launch::async, [=] {
        execute_reduction(accessors[0].address, accessors[1].address, m_reduction);
    });
}

std::future<void> FillHostJob::submit(Executor& executor, std::vector<BufferAccessor> accessors) {
    return std::async(std::launch::async, [=] { execute_fill(accessors[0].address, m_fill); });
}

Poll DeviceJob::poll(Executor& executor) {
    if (m_status == Status::Init) {
        try {
            m_requests = executor.buffers().create_requests(m_buffers);
            m_status = Status::PollingBuffers;
        } catch (const std::exception& e) {
            executor.buffers().poison_all(m_buffers, make_poison_exception(m_task, e));
            m_status = Status::Completing;
        }
    }

    if (m_status == Status::PollingBuffers) {
        try {
            if (executor.buffers().poll_requests(m_requests, m_dependencies) == Poll::Pending) {
                return Poll::Pending;
            }

            // Remove the `local` events from the list of dependencies. These events
            // have the same context as the current device, and thus can be directly put
            // as dependencies on the current device stream.
            m_local_dependencies = m_dependencies.extract_events_for_context(
                executor.streams(),
                executor.devices().context(m_resource.as_device())
            );

            m_status = Status::PollingDependencies;
        } catch (const std::exception& e) {
            executor.buffers().poison_all(m_buffers, make_poison_exception(m_task, e));
            m_status = Status::Completing;
        }
    }

    if (m_status == Status::PollingDependencies) {
        if (!executor.streams().is_ready(m_dependencies)) {
            return Poll::Pending;
        }

        try {
            m_execution_event = executor.devices().submit(
                m_resource.as_device(),
                m_resource.stream_affinity(),
                m_local_dependencies,
                *this,
                executor.buffers().access_requests(m_requests)
            );

            executor.scheduler().mark_as_scheduled(m_task, m_execution_event);
            m_status = Status::Running;
        } catch (const std::exception& e) {
            executor.buffers().poison_all(m_buffers, make_poison_exception(m_task, e));
            m_status = Status::Completing;
        }

        executor.buffers().release_requests(m_requests, m_execution_event);
    }

    if (m_status == Status::Running) {
        if (!executor.streams().is_ready(m_execution_event)) {
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

void ExecuteDeviceJob::execute(DeviceResource& device, std::vector<BufferAccessor> accessors) {
    auto context = TaskContext {std::move(accessors)};
    m_task->execute(device, context);
}

void CopyDeviceJob::execute(DeviceResource& device, std::vector<BufferAccessor> accessors) {
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

void ReductionDeviceJob::execute(DeviceResource& device, std::vector<BufferAccessor> accessors) {
    execute_gpu_reduction_async(
        device,
        reinterpret_cast<GPUdeviceptr>(accessors[0].address),
        reinterpret_cast<GPUdeviceptr>(accessors[1].address),
        m_reduction
    );
}

void FillDeviceJob::execute(DeviceResource& device, std::vector<BufferAccessor> accessors) {
    execute_gpu_fill_async(device, reinterpret_cast<GPUdeviceptr>(accessors[0].address), m_fill);
}

Poll PrefetchJob::poll(Executor& executor) {
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

std::unique_ptr<Job> build_job_for_command(
    TaskHandle task,
    const Command& command,
    DeviceEventSet dependencies
) {
    if (std::get_if<CommandEmpty>(&command) != nullptr) {
        return std::make_unique<MergeJob>(task, std::move(dependencies));

    } else if (const auto* e = std::get_if<CommandBufferDelete>(&command)) {
        return std::make_unique<DeleteBufferJob>(task, e->id, std::move(dependencies));

    } else if (const auto* e = std::get_if<CommandPrefetch>(&command)) {
        return std::make_unique<PrefetchJob>(
            task,
            e->buffer_id,
            e->memory_id,
            std::move(dependencies)
        );

    } else if (const auto* e = std::get_if<CommandExecute>(&command)) {
        auto proc = e->processor_id;

        if (proc.is_device()) {
            return std::make_unique<ExecuteDeviceJob>(
                task,
                proc,
                e->task.get(),
                e->buffers,
                std::move(dependencies)
            );
        } else {
            return std::make_unique<ExecuteHostJob>(
                task,
                e->task.get(),
                e->buffers,
                std::move(dependencies)
            );
        }

    } else if (const auto* e = std::get_if<CommandCopy>(&command)) {
        auto src_mem = e->src_memory;
        auto dst_mem = e->dst_memory;

        if (src_mem.is_host() && dst_mem.is_host()) {
            return std::make_unique<CopyHostJob>(
                task,
                e->src_buffer,
                e->dst_buffer,
                e->definition,
                std::move(dependencies)
            );
        } else if (dst_mem.is_device()) {
            return std::make_unique<CopyDeviceJob>(
                task,
                dst_mem.as_device(),
                e->src_buffer,
                e->dst_buffer,
                e->definition,
                std::move(dependencies)
            );
        } else if (src_mem.is_device()) {
            return std::make_unique<CopyDeviceJob>(
                task,
                src_mem.as_device(),
                e->src_buffer,
                e->dst_buffer,
                e->definition,
                std::move(dependencies)
            );
        } else {
            KMM_PANIC("unsupported copy");
        }

    } else if (const auto* e = std::get_if<CommandReduction>(&command)) {
        auto memory_id = e->memory_id;

        if (memory_id.is_device()) {
            return std::make_unique<ReductionDeviceJob>(
                task,
                memory_id.as_device(),
                e->src_buffer,
                e->dst_buffer,
                std::move(e->definition),
                std::move(dependencies)
            );
        } else {
            return std::make_unique<ReductionHostJob>(
                task,
                e->src_buffer,
                e->dst_buffer,
                std::move(e->definition),
                std::move(dependencies)
            );
        }

    } else if (const auto* e = std::get_if<CommandFill>(&command)) {
        auto memory_id = e->memory_id;

        if (memory_id.is_device()) {
            return std::make_unique<FillDeviceJob>(
                task,
                memory_id.as_device(),
                e->dst_buffer,
                std::move(e->definition),
                std::move(dependencies)
            );
        } else {
            return std::make_unique<FillHostJob>(
                task,
                e->dst_buffer,
                std::move(e->definition),
                std::move(dependencies)
            );
        }

    } else {
        KMM_PANIC_FMT("could not handle unknown command: {}", command);
    }
}

}  // namespace kmm