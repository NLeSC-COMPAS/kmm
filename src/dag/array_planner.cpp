#include "kmm/dag/array_planner.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {

template<size_t N>
ArrayInstance<N> ArrayInstance<N>::instantiate(
    TaskGraph& graph,
    Distribution<N> dist,
    DataLayout element_layout
) {
    size_t n = dist.num_chunks();
    auto buffers = std::vector<BufferId>(n);

    for (size_t i = 0; i < n; i++) {
        auto chunk = dist.chunk(i);
        buffers[i] = graph.create_buffer(element_layout.repeat(chunk.size.volume()));
    }

    return {std::move(dist), std::move(buffers), element_layout};
}

template<size_t N>
void ArrayInstance<N>::destroy(TaskGraph& graph) {
    for (const auto& buffer_id : m_buffers) {
        graph.delete_buffer(buffer_id);
    }

    this->m_distribution = Distribution<N>();
    this->m_buffers.clear();
}

template<size_t N>
BufferRequirement ArrayReadPlanner<N>::plan_access(
    TaskGraph& graph,
    MemoryId memory_id,
    Bounds<N>& region
) {
    KMM_ASSERT(m_instance);

    const auto& dist = m_instance->distribution();
    auto chunk_index = dist.region_to_chunk_index(region);
    auto chunk = dist.chunk(chunk_index);
    auto buffer_id = m_instance->buffers()[chunk_index];

    region = Bounds<N>::from_offset_size(chunk.offset, chunk.size);

    return BufferRequirement {
        .buffer_id = buffer_id,
        .memory_id = memory_id,
        .access_mode = AccessMode::Read};
}

template<size_t N>
void ArrayReadPlanner<N>::finalize(TaskGraph& graph, const EventList& access_events) {
    // Nothing for now
}

template<size_t N>
BufferRequirement ArrayWritePlanner<N>::plan_access(
    TaskGraph& graph,
    MemoryId memory_id,
    Bounds<N>& region
) {
    KMM_ASSERT(m_instance);

    const auto& dist = m_instance->distribution();
    auto chunk_index = dist.region_to_chunk_index(region);
    auto chunk = dist.chunk(chunk_index);
    auto buffer_id = m_instance->buffers()[chunk_index];

    // Only allow writing from the same memory as the owner
    if (chunk.owner_id != memory_id) {
        throw std::runtime_error(fmt::format(
            "array chunk for region {} is owned by memory {}, it cannot be written to from a different memory {}",
            region,
            chunk.owner_id,
            memory_id
        ));
    }

    region = Bounds<N>::from_offset_size(chunk.offset, chunk.size);

    return BufferRequirement {
        .buffer_id = buffer_id,
        .memory_id = memory_id,
        .access_mode = AccessMode::ReadWrite};
}

template<size_t N>
void ArrayWritePlanner<N>::finalize(TaskGraph& graph, const EventList& access_events) {
    // Nothing for now
}

KMM_INSTANTIATE_ARRAY_IMPL(ArrayInstance)
KMM_INSTANTIATE_ARRAY_IMPL(ArrayReadPlanner)
KMM_INSTANTIATE_ARRAY_IMPL(ArrayWritePlanner)

}  // namespace kmm