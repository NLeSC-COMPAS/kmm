#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/dag/data_distribution.hpp"
#include "kmm/dag/reduction_planner.hpp"
#include "kmm/utils/geometry.hpp"
#include "kmm/utils/view.hpp"

namespace kmm {

class TaskGraph;

class ReductionPlanner {
    KMM_NOT_COPYABLE(ReductionPlanner)

  public:
    ReductionPlanner() = default;

    ReductionPlanner(size_t num_elements, DataType data_type, Reduction operation) :
        m_num_elements(num_elements),
        m_dtype(data_type),
        m_reduction(operation) {}

    MemoryId affinity_memory() const;
    BufferRequirement add_chunk(
        TaskGraph& graph,
        MemoryId memory_id,
        size_t replication_factor = 1
    );
    EventId finalize(TaskGraph& graph, BufferId buffer_id, MemoryId memory_id);

    size_t m_num_elements = 0;
    DataType m_dtype = ScalarKind::Invalid;
    Reduction m_reduction = Reduction::Invalid;
    std::vector<ReductionInput> m_inputs;
};

}  // namespace kmm