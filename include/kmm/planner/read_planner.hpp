#pragma once

#include "kmm/planner/array_instance.hpp"

namespace kmm {

template<size_t N>
class ArrayReadPlanner {
    KMM_NOT_COPYABLE_OR_MOVABLE(ArrayReadPlanner)

  public:
    ArrayReadPlanner(std::shared_ptr<ArrayDescriptor<N>> instance);
    ~ArrayReadPlanner();

    BufferRequirement prepare_access(
        TaskGraphStage& stage,
        MemoryId memory_id,
        Bounds<N>& region,
        EventList& deps_out
    );

    void finalize_access(TaskGraphStage& stage, EventId event_id);

    void commit(TaskGraphStage& stage);

  private:
    std::shared_ptr<ArrayDescriptor<N>> m_instance;
    std::vector<std::pair<size_t, EventId>> m_read_events;
};

KMM_INSTANTIATE_ARRAY_IMPL(ArrayReadPlanner)

}  // namespace kmm