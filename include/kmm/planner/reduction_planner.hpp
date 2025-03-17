#pragma once

#include "kmm/core/reduction.hpp"
#include "kmm/planner/array_instance.hpp"

namespace kmm {

template<size_t N>
class ArrayReductionPlanner {
  public:
    ArrayReductionPlanner();
    ArrayReductionPlanner(std::shared_ptr<ArrayDescriptor<N>> instance, Reduction op);
    ~ArrayReductionPlanner();

    BufferRequirement prepare_access(
        TaskGraphStage& stage,
        MemoryId memory_id,
        Bounds<N>& region,
        size_t replication_factor,
        EventList& deps_out
    );

    void finalize_access(TaskGraphStage& stage, EventId event_id);

    void commit(TaskGraphStage& stage);

  private:
    std::shared_ptr<ArrayDescriptor<N>> m_instance;
    Reduction m_reduction;
};

KMM_INSTANTIATE_ARRAY_IMPL(ArrayReductionPlanner)

}  // namespace kmm