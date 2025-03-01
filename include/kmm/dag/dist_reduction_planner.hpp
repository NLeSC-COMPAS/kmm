#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/dag/data_distribution.hpp"
#include "kmm/dag/reduction_planner.hpp"
#include "kmm/utils/geometry.hpp"
#include "kmm/utils/view.hpp"

namespace kmm {

class TaskGraph;

template<size_t N>
class DistReductionPlanner {
    KMM_NOT_COPYABLE(DistReductionPlanner)

  public:
    DistReductionPlanner() = default;

    DistReductionPlanner(Dim<N> shape, DataType data_type, Reduction operation) :
        m_shape(shape),
        m_dtype(data_type),
        m_reduction(operation) {}

    BufferRequirement add_chunk(
        TaskGraph& graph,
        MemoryId memory_id,
        Bounds<N> access_region,
        size_t replication_factor = 1
    );

    std::pair<DataDistribution<N>, std::vector<BufferId>> finalize(TaskGraph& graph);

    Dim<N> shape() const {
        return m_shape;
    }

  private:
    Dim<N> m_shape;
    DataType m_dtype;
    Reduction m_reduction = Reduction::Invalid;
    std::unordered_map<Bounds<N>, ReductionPlanner> m_partial_inputs;
};

}  // namespace kmm