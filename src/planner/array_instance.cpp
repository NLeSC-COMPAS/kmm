#include "kmm/planner/array_instance.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

template<size_t N>
ArrayDescriptor<N>::ArrayDescriptor(
    Distribution<N> distribution,
    DataType dtype,
    std::vector<BufferDescriptor> buffers
) :
    m_distribution(std::move(distribution)),
    m_dtype(dtype),
    m_buffers(std::move(buffers)) {}

template<size_t N>
EventId ArrayDescriptor<N>::join_events(TaskGraphStage& stage) const {
    EventList deps;

    for (const auto& buffer : m_buffers) {
        deps.insert_all(buffer.last_access_events);
    }

    return stage.join_events(deps);
}

template<size_t N>
void ArrayDescriptor<N>::destroy(TaskGraphStage& stage) {
    KMM_ASSERT(m_num_readers == 0 && m_num_writers == 0);

    for (BufferDescriptor& meta : m_buffers) {
        stage.delete_buffer(meta.id, std::move(meta.last_access_events));
    }

    m_distribution = Distribution<N>();
    m_buffers.clear();
}

template<size_t N>
static std::vector<BufferDescriptor> allocate_buffers(
    Runtime& rt,
    const Distribution<N>& dist,
    DataType dtype
) {
    size_t num_chunks = dist.num_chunks();
    std::vector<BufferDescriptor> buffers(num_chunks);

    for (size_t i = 0; i < num_chunks; i++) {
        auto chunk = dist.chunk(i);
        auto num_elements = chunk.size.volume();
        auto layout = BufferLayout::for_type(dtype, num_elements);
        auto buffer_id = rt.create_buffer(layout);

        buffers[i] = BufferDescriptor {.id = buffer_id, .layout = layout};
    }

    return buffers;
}

template<size_t N>
ArrayInstance<N>::ArrayInstance(Runtime& rt, Distribution<N> dist, DataType dtype) :
    ArrayDescriptor<N>(dist, dtype, allocate_buffers(rt, dist, dtype)) {}

template<size_t N>
ArrayInstance<N>::~ArrayInstance() {
    m_rt->schedule([&](TaskGraphStage& stage) { this->destroy(stage); });
}

template<size_t N>
void ArrayInstance<N>::copy_bytes(void* data) const {
    KMM_TODO();
}

template<size_t N>
void ArrayInstance<N>::synchronize() const {
    EventId event_id;
    m_rt->schedule([&](TaskGraphStage& stage) { event_id = this->join_events(stage); });

    m_rt->query_event(event_id, std::chrono::system_clock::time_point::max());
}

}  // namespace kmm
