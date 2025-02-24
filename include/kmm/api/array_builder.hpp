#pragma once

#include "kmm/api/data_distribution.hpp"
#include "kmm/core/buffer.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/utils/geometry.hpp"
#include "kmm/utils/view.hpp"

namespace kmm {

class TaskGraph;

template<size_t N>
class ArrayBuilder {
  public:
    ArrayBuilder(Dim<N> sizes, DataLayout element_layout) :
        m_sizes(sizes),
        m_element_layout(element_layout) {}

    BufferRequirement add_chunk(TaskGraph& graph, MemoryId memory_id, Bounds<N> access_region);
    std::pair<DataDistribution<N>, std::vector<BufferId>> build(TaskGraph& graph);

    Dim<N> sizes() const {
        return m_sizes;
    }

  private:
    Dim<N> m_sizes;
    DataLayout m_element_layout;
    std::vector<DataChunk<N>> m_chunks;
    std::vector<BufferId> m_buffers;
};

template<size_t N>
class ArrayReductionBuilder {
  public:
    ArrayReductionBuilder(Dim<N> sizes, DataType data_type, Reduction operation) :
        m_sizes(sizes),
        m_dtype(data_type),
        m_reduction(operation) {}

    BufferRequirement add_chunk(
        TaskGraph& graph,
        MemoryId memory_id,
        Bounds<N> access_region,
        size_t replication_factor = 1
    );

    void add_chunks(ArrayReductionBuilder<N>&& other);

    std::pair<DataDistribution<N>, std::vector<BufferId>> build(TaskGraph& graph);

    Dim<N> sizes() const {
        return m_sizes;
    }

  private:
    Dim<N> m_sizes;
    DataType m_dtype;
    Reduction m_reduction;
    std::unordered_map<Bounds<N>, std::vector<ReductionInput>> m_partial_inputs;
};

}  // namespace kmm