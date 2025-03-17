#include "kmm/api/runtime_handle.hpp"
#include "kmm/core/resource.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

class CopyInTask: public ComputeTask {
  public:
    CopyInTask(const void* data, size_t nbytes) : m_src_addr(data), m_nbytes(nbytes) {}

    void execute(Resource& proc, TaskContext context) override {
        KMM_ASSERT(context.accessors.size() == 1);
        KMM_ASSERT(context.accessors[0].layout.size_in_bytes == m_nbytes);

        void* dst_addr = context.accessors[0].address;

        if (auto* device = proc.cast_if<DeviceResource>()) {
            device->copy_bytes(m_src_addr, dst_addr, m_nbytes);
        } else if (proc.is<HostResource>()) {
            ::memcpy(dst_addr, m_src_addr, m_nbytes);
        } else {
            throw std::runtime_error("invalid execution context");
        }
    }

  private:
    const void* m_src_addr;
    size_t m_nbytes;
};

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

BufferId RuntimeHandle::allocate_bytes(const void* data, BufferLayout layout, MemoryId memory_id)
    const {
    ProcessorId proc = info().affinity_processor(memory_id);
    BufferId buffer_id = m_worker->create_buffer(layout);
    EventId event_id;

    m_worker->schedule([&](TaskGraphStage& graph) {
        auto task = std::make_unique<CopyInTask>(data, layout.size_in_bytes);
        auto req = BufferRequirement {
            .buffer_id = buffer_id,  //
            .memory_id = memory_id,
            .access_mode = AccessMode::Exclusive};

        event_id = graph.insert_compute_task(proc, std::move(task), {req});
    });

    wait(event_id);
    return buffer_id;
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
