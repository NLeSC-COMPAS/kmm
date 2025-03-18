#pragma once

#include "kmm/planner/array_instance.hpp"

namespace kmm {

template<size_t N>
class ArrayWritePlanner {
    KMM_NOT_COPYABLE_OR_MOVABLE(ArrayWritePlanner)

  public:
    ArrayWritePlanner(std::shared_ptr<ArrayDescriptor<N>> instance);
    ~ArrayWritePlanner();

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
    std::vector<std::pair<size_t, EventId>> m_write_events;
};

KMM_INSTANTIATE_ARRAY_IMPL(ArrayWritePlanner)
}  // namespace kmm