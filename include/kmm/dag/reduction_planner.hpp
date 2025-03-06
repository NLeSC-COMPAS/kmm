#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/dag/distribution.hpp"
#include "kmm/dag/reduction_planner.hpp"
#include "kmm/utils/geometry.hpp"
#include "kmm/utils/view.hpp"

namespace kmm {

class TaskGraph;

class LocalReductionPlanner {
    KMM_NOT_COPYABLE(LocalReductionPlanner)

  public:
    LocalReductionPlanner() = default;

    LocalReductionPlanner(size_t num_elements, DataType data_type, Reduction operation) :
        m_num_elements(num_elements),
        m_dtype(data_type),
        m_reduction(operation) {}

    BufferRequirement plan_access(
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

template<size_t N>
class ReductionPlanner {
    KMM_NOT_COPYABLE(ReductionPlanner)

  public:
    ReductionPlanner() = default;
    ReductionPlanner(const ArrayInstance<N>* instance, DataType data_type, Reduction operation);

    BufferRequirement plan_access(
        TaskGraph& graph,
        MemoryId memory_id,
        Bounds<N>& access_region,
        size_t replication_factor = 1
    );

    EventId finalize(TaskGraph& graph, const EventList& events);

  private:
    const ArrayInstance<N>* m_instance = nullptr;
    DataType m_dtype = ScalarKind::Invalid;
    Reduction m_reduction = Reduction::Invalid;
    std::vector<std::unique_ptr<LocalReductionPlanner>> m_partial_inputs;
    std::vector<std::pair<size_t, size_t>> m_access_indices;
};

}  // namespace kmm