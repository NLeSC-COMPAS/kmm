#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/dag/data_distribution.hpp"
#include "kmm/utils/geometry.hpp"
#include "kmm/utils/view.hpp"

namespace kmm {

class TaskGraph;

template<size_t N>
class ArrayBuilder {
  public:
    ArrayBuilder() = default;

    ArrayBuilder(Dim<N> sizes, DataLayout element_layout) :
        m_sizes(sizes),
        m_element_layout(element_layout) {}

    BufferRequirement add_chunk(TaskGraph& graph, MemoryId memory_id, Bounds<N> access_region);

    std::pair<DataDistribution<N>, std::vector<BufferId>> build(TaskGraph& graph);

    Dim<N> sizes() const {
        return m_sizes;
    }

  private:
    Dim<N> m_sizes = 0;
    DataLayout m_element_layout;
    std::vector<DataChunk<N>> m_chunks;
    std::vector<BufferId> m_buffers;
};
}  // namespace kmm