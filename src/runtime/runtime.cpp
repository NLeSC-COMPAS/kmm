#include <thread>

#include "spdlog/spdlog.h"

#include "kmm/runtime/allocators/block.hpp"
#include "kmm/runtime/allocators/device.hpp"
#include "kmm/runtime/allocators/system.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

static SystemInfo make_system_info(const std::vector<GPUContextHandle>& contexts) {
    spdlog::info("detected {} GPU device(s):", contexts.size());
    std::vector<DeviceInfo> device_infos;

    for (size_t i = 0; i < contexts.size(); i++) {
        auto info = DeviceInfo(DeviceId(i), contexts[i]);
        auto memory_gb = static_cast<double>(info.total_memory_size()) / 1e9;

        spdlog::info(" - {} ({:.2} GB)", info.name(), memory_gb);
        device_infos.push_back(info);
    }

    return device_infos;
}

Runtime::Runtime(
    std::vector<GPUContextHandle> contexts,
    std::shared_ptr<DeviceStreamManager> stream_manager,
    std::shared_ptr<MemorySystem> memory_system,
    const RuntimeConfig& config
) :
    m_memory_system(memory_system),
    m_memory_manager(std::make_shared<MemoryManager>(memory_system)),
    m_buffer_registry(std::make_shared<BufferRegistry>(m_memory_manager)),
    m_stream_manager(stream_manager),
    m_devices(std::make_shared<DeviceResourceManager>(contexts, m_stream_manager)),
    m_scheduler(std::make_shared<Scheduler>(contexts.size())),
    m_info(make_system_info(contexts)),
    m_executor(m_devices, stream_manager, m_buffer_registry, m_scheduler, config.debug_mode) {}

Runtime::~Runtime() {
    shutdown();
}

BufferId Runtime::create_buffer(BufferLayout layout) {
    TaskGraphStage stage(&m_graph);
    return stage.create_buffer(layout);
}

void Runtime::delete_buffer(BufferId id) {
    TaskGraphStage stage(&m_graph);
    stage.delete_buffer(id);
}

bool Runtime::query_event(EventId event_id, std::chrono::system_clock::time_point deadline) {
    static constexpr auto TIMEOUT = std::chrono::microseconds {100};

    std::unique_lock guard {m_mutex};
    flush_events_impl();

    if (m_scheduler->is_completed(event_id)) {
        return true;
    }

    auto next_update = std::chrono::system_clock::now();

    while (true) {
        make_progress_impl();

        if (m_scheduler->is_completed(event_id)) {
            return true;
        }

        auto now = std::chrono::system_clock::now();
        next_update += TIMEOUT;

        if (next_update < now) {
            next_update = now;
        }

        if (next_update > deadline) {
            return false;
        }

        guard.unlock();
        std::this_thread::sleep_until(next_update);
        guard.lock();
    }
}

bool Runtime::is_idle() {
    std::lock_guard guard {m_mutex};
    flush_events_impl();
    return is_idle_impl();
}

void Runtime::trim_memory() {
    std::lock_guard guard {m_mutex};
    m_memory_system->trim_host();
    m_memory_system->trim_device();
}

void Runtime::make_progress() {
    std::lock_guard guard {m_mutex};
    flush_events_impl();
    make_progress_impl();
}

void Runtime::shutdown() {
    std::unique_lock guard {m_mutex};
    if (m_has_shutdown) {
        return;
    }

    m_has_shutdown = true;
    flush_events_impl();

    while (!is_idle_impl()) {
        make_progress_impl();

        guard.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds {10});
        guard.lock();
    }

    m_stream_manager->wait_until_idle();
}

void Runtime::flush_events_impl() {
    for (auto&& [id, layout] : m_graph.flush_buffers()) {
        m_buffer_registry->add(id, layout);
    }

    // Flush all events from the DAG builder to the scheduler
    for (auto&& e : m_graph.flush_tasks()) {
        m_scheduler->submit(e.id, std::move(e.command), std::move(e.dependencies));
    }
}

void Runtime::make_progress_impl() {
    m_stream_manager->make_progress();
    m_memory_system->make_progress();
    m_executor.make_progress();

    DeviceEventSet deps;
    while (auto task = m_scheduler->pop_ready(&deps)) {
        m_executor.execute_task(std::move(*task), std::move(deps));
    }
}

bool Runtime::is_idle_impl() {
    return m_stream_manager->is_idle() && m_executor.is_idle() && m_scheduler->is_idle()
        && m_memory_manager->is_idle(*m_stream_manager);
}

std::unique_ptr<AsyncAllocator> create_device_allocator(
    const RuntimeConfig& config,
    GPUContextHandle context,
    std::shared_ptr<DeviceStreamManager> stream_manager
) {
    switch (config.device_memory_kind) {
        case DeviceMemoryKind::NoPool:
            return std::make_unique<DeviceMemoryAllocator>(
                context,
                stream_manager,
                config.device_memory_limit
            );

        case DeviceMemoryKind::DefaultPool:
            return std::make_unique<DevicePoolAllocator>(
                context,
                stream_manager,
                DevicePoolKind::Default,
                config.device_memory_limit
            );

        case DeviceMemoryKind::PrivatePool:
            return std::make_unique<DevicePoolAllocator>(
                context,
                stream_manager,
                DevicePoolKind::Create,
                config.device_memory_limit
            );

        default:
            KMM_PANIC("invalid memory kind");
    }
}

std::shared_ptr<Runtime> make_worker(const RuntimeConfig& config) {
    std::unique_ptr<AsyncAllocator> host_mem;
    std::vector<std::unique_ptr<AsyncAllocator>> device_mems;

    auto stream_manager = std::make_shared<DeviceStreamManager>();
    auto contexts = std::vector<GPUContextHandle>();
    auto devices = get_gpu_devices();

    if (devices.empty()) {
        host_mem = std::make_unique<SystemAllocator>(stream_manager, config.host_memory_limit);
    } else if (devices.size() > MAX_DEVICES) {
        throw std::runtime_error(fmt::format("cannot support more than {} GPU(s)", MAX_DEVICES));
    } else {
        for (const auto& device : devices) {
            auto context = GPUContextHandle::retain_primary_context_for_device(device);
            contexts.push_back(context);
            device_mems.push_back(create_device_allocator(config, context, stream_manager));
        }

        host_mem = std::make_unique<PinnedMemoryAllocator>(
            contexts.at(0),
            stream_manager,
            config.host_memory_limit
        );
    }

    if (config.host_memory_block_size > 0) {
        host_mem = std::make_unique<BlockAllocator>(  //
            std::move(host_mem),
            config.host_memory_block_size
        );
    }

    if (config.device_memory_block_size > 0) {
        for (size_t i = 0; i < devices.size(); i++) {
            device_mems[i] = std::make_unique<BlockAllocator>(
                std::move(device_mems[i]),
                config.device_memory_block_size
            );
        }
    }

    auto memory_system = std::make_shared<MemorySystemImpl>(
        stream_manager,
        contexts,
        std::move(host_mem),
        std::move(device_mems)
    );

    return std::make_shared<Runtime>(contexts, stream_manager, memory_system, config);
}
}  // namespace kmm
