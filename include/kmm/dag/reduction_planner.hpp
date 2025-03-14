#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/core/view.hpp"
#include "kmm/dag/distribution.hpp"
#include "kmm/dag/reduction_planner.hpp"
#include "kmm/utils/geometry.hpp"

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

    BufferRequirement prepare_access(
        TaskGraph& graph,
        MemoryId memory_id,
        size_t replication_factor = 1
    );

    EventId finalize(TaskGraph& graph, BufferId buffer_id, MemoryId memory_id);

    size_t m_num_elements = 0;
    DataType m_dtype;
    Reduction m_reduction = Reduction::Invalid;
    std::vector<ReductionInput> m_inputs;
};

template<size_t N>
class ReductionPlanner {
    KMM_NOT_COPYABLE(ReductionPlanner)

  public:
    ReductionPlanner() = default;
    ReductionPlanner(const ArrayInstance<N>* instance, Reduction operation);

    BufferRequirement prepare_access(
        TaskGraph& graph,
        MemoryId memory_id,
        Bounds<N>& access_region,
        size_t replication_factor = 1
    );

    void finalize_access(TaskGraph& graph, EventId event_id);

    EventId finalize(TaskGraph& graph);

  private:
    const ArrayInstance<N>* m_instance = nullptr;
    Reduction m_reduction = Reduction::Invalid;
    std::vector<std::unique_ptr<LocalReductionPlanner>> m_partial_inputs;
    LocalReductionPlanner* m_last_access;
};

}  // namespace kmm