#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/dag/data_distribution.hpp"
#include "kmm/utils/geometry.hpp"
#include "kmm/utils/view.hpp"

namespace kmm {

class TaskGraph;

template<size_t N>
class ReductionBuilder {
  public:
    ReductionBuilder() = default;

    ReductionBuilder(Dim<N> sizes, DataType data_type, Reduction operation) :
        m_sizes(sizes),
        m_dtype(data_type),
        m_reduction(operation) {}

    BufferRequirement add_chunk(
        TaskGraph& graph,
        MemoryId memory_id,
        Bounds<N> access_region,
        size_t replication_factor = 1
    );

    void add_chunks(ReductionBuilder<N>&& other);

    std::pair<DataDistribution<N>, std::vector<BufferId>> build(TaskGraph& graph);

    Dim<N> sizes() const {
        return m_sizes;
    }

  private:
    Dim<N> m_sizes;
    DataType m_dtype;
    Reduction m_reduction = Reduction::Invalid;
    std::unordered_map<Bounds<N>, std::vector<ReductionInput>> m_partial_inputs;
};

}  // namespace kmm