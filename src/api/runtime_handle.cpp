#include "kmm/api/runtime_handle.hpp"
#include "kmm/core/resource.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

RuntimeHandle::RuntimeHandle(std::shared_ptr<Runtime> rt) : m_worker(std::move(rt)) {
    KMM_ASSERT(m_worker != nullptr);
}

RuntimeHandle::RuntimeHandle(Runtime& rt) : RuntimeHandle(rt.shared_from_this()) {}

MemoryId RuntimeHandle::memory_affinity_for_address(const void* address) const {
    if (auto device_opt = get_gpu_device_by_address(address)) {
        const auto& device = m_worker->system_info().device_by_ordinal(*device_opt);
        return device.memory_id();
    } else {
        return MemoryId::host();
    }
}

bool RuntimeHandle::is_done(EventId id) const {
    return m_worker->query_event(id, std::chrono::system_clock::time_point::min());
}

void RuntimeHandle::wait(EventId id) const {
    m_worker->query_event(id, std::chrono::system_clock::time_point::max());
}

bool RuntimeHandle::wait_until(EventId id, typename std::chrono::system_clock::time_point deadline)
    const {
    return m_worker->query_event(id, deadline);
}

bool RuntimeHandle::wait_for(EventId id, typename std::chrono::system_clock::duration duration)
    const {
    return m_worker->query_event(id, std::chrono::system_clock::now() + duration);
}

EventId RuntimeHandle::barrier() const {
    return m_worker->schedule([&](TaskGraphStage& g) { g.insert_barrier(); });
}

void RuntimeHandle::synchronize() const {
    wait(barrier());
}

const SystemInfo& RuntimeHandle::info() const {
    return m_worker->system_info();
}

const Runtime& RuntimeHandle::worker() const {
    return *m_worker;
}

RuntimeHandle make_runtime(const RuntimeConfig& config) {
    return make_worker(config);
}

}  // namespace kmm
