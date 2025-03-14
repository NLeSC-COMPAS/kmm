#pragma once

#include <mutex>

#include "executor.hpp"
#include "scheduler.hpp"

#include "kmm/core/config.hpp"
#include "kmm/core/system_info.hpp"
#include "kmm/dag/task_graph.hpp"
#include "kmm/runtime/memory_system.hpp"

namespace kmm {

class Runtime: public std::enable_shared_from_this<Runtime> {
    KMM_NOT_COPYABLE_OR_MOVABLE(Runtime)

  public:
    Runtime(
        std::vector<GPUContextHandle> contexts,
        std::shared_ptr<DeviceStreamManager> stream_manager,
        std::shared_ptr<MemorySystem> memory_system,
        const RuntimeConfig& config
    );
    ~Runtime();

    bool query_event(EventId event_id, std::chrono::system_clock::time_point deadline);
    bool is_idle();
    void trim_memory();
    void make_progress();
    void shutdown();

    template<typename F>
    void with_task_graph(F fun) {
        std::lock_guard guard {m_graph_mutex};

        try {
            fun(m_graph);
            m_graph.commit();
        } catch (...) {
            m_graph.rollback();
            throw;
        }
    }

    const SystemInfo& system_info() const {
        return m_info;
    }

  private:
    void flush_events_impl();
    void make_progress_impl();
    bool is_idle_impl();

    mutable std::mutex m_mutex;
    mutable bool m_has_shutdown = false;
    std::shared_ptr<MemorySystem> m_memory_system;
    std::shared_ptr<MemoryManager> m_memory_manager;
    std::shared_ptr<BufferRegistry> m_buffer_registry;
    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    std::shared_ptr<DeviceResourceManager> m_devices;
    std::shared_ptr<Scheduler> m_scheduler;
    SystemInfo m_info;
    Executor m_executor;

    mutable std::mutex m_graph_mutex;
    TaskGraph m_graph;
};

std::shared_ptr<Runtime> make_worker(const RuntimeConfig& config);

}  // namespace kmm