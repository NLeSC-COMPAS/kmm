#pragma once

#include <mutex>

#include "kmm/core/config.hpp"
#include "kmm/core/system_info.hpp"
#include "kmm/runtime/memory_system.hpp"
#include "kmm/runtime/scheduler.hpp"
#include "kmm/runtime/task_graph.hpp"

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

    BufferId create_buffer(BufferLayout layout);
    void delete_buffer(BufferId buffer_id, EventList deps = {});
    void check_buffer(BufferId id);

    bool query_event(EventId event_id, std::chrono::system_clock::time_point deadline);
    bool is_idle();
    void trim_memory();
    void make_progress();
    void shutdown();

    template<typename F, typename R = std::invoke_result_t<F, TaskGraph&>>
    R schedule(F fun) {
        std::lock_guard guard {m_mutex};

        if constexpr (std::is_void_v<R>) {
            auto stage = TaskGraph(&m_graph_state);
            fun(stage);
            this->commit_impl(stage);
        } else {
            auto stage = TaskGraph(&m_graph_state);
            auto result = fun(stage);
            this->commit_impl(stage);
            return result;
        }
    }

    const SystemInfo& system_info() const {
        return m_info;
    }

  private:
    EventId commit_impl(TaskGraph& g);
    void make_progress_impl();
    bool is_idle_impl();

    mutable std::mutex m_mutex;
    std::chrono::system_clock::time_point m_next_updated_planned = std::chrono::system_clock::now();
    mutable bool m_has_shutdown = false;
    std::shared_ptr<MemorySystem> m_memory_system;
    std::shared_ptr<MemoryManager> m_memory_manager;
    std::shared_ptr<BufferRegistry> m_buffer_registry;
    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    std::shared_ptr<DeviceResources> m_devices;
    SystemInfo m_info;
    Scheduler m_scheduler;
    TaskGraphState m_graph_state;
};

std::shared_ptr<Runtime> make_worker(const RuntimeConfig& config);

}  // namespace kmm