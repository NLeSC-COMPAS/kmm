#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/dag/data_distribution.hpp"
#include "kmm/utils/geometry.hpp"
#include "kmm/utils/view.hpp"

namespace kmm {

class TaskGraph;

template<size_t N>
class DistDataPlanner {
    KMM_NOT_COPYABLE(DistDataPlanner)

  public:
    DistDataPlanner() = default;

    DistDataPlanner(Dim<N> shape, DataLayout element_layout) :
        m_shape(shape),
        m_element_layout(element_layout) {}

    BufferRequirement add_chunk(TaskGraph& graph, MemoryId memory_id, Bounds<N> access_region);

    std::pair<DataDistribution<N>, std::vector<BufferId>> finalize(TaskGraph& graph);

    Dim<N> shape() const {
        return m_shape;
    }

  private:
    Dim<N> m_shape = 0;
    DataLayout m_element_layout;
    std::vector<DataChunk<N>> m_chunks;
    std::vector<BufferId> m_buffers;
};
}  // namespace kmm