#include "kmm/dag/array_builder.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {

template<size_t N>
BufferRequirement ArrayBuilder<N>::add_chunk(
    TaskGraph& graph,
    MemoryId memory_id,
    Bounds<N> access_region
) {
    auto num_elements = checked_cast<size_t>(access_region.size());
    auto buffer_id = graph.create_buffer(m_element_layout.repeat(num_elements));

    m_buffers.push_back(buffer_id);
    m_chunks.push_back(DataChunk<N> {
        .owner_id = memory_id,
        .offset = access_region.begin(),
        .size = access_region.sizes()});

    return {//
            .buffer_id = buffer_id,
            .memory_id = memory_id,
            .access_mode = AccessMode::Exclusive};
}

template<size_t N>
std::pair<DataDistribution<N>, std::vector<BufferId>> ArrayBuilder<N>::build(TaskGraph& graph) {
    auto dist = DataDistribution<N>::from_chunks(m_sizes, std::move(m_chunks), m_buffers);
    return {std::move(dist), std::move(m_buffers)};
}

KMM_INSTANTIATE_ARRAY_IMPL(ArrayBuilder)

}  // namespace kmm