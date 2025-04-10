#pragma once

#include "kmm/planner/array_descriptor.hpp"

namespace kmm {

template<size_t N>
class ArrayWritePlanner {
    KMM_NOT_COPYABLE_OR_MOVABLE(ArrayWritePlanner)

  public:
    ArrayWritePlanner(std::shared_ptr<ArrayDescriptor<N>> instance);
    ~ArrayWritePlanner();

    BufferRequirement prepare_access(
        TaskGraph& stage,
        MemoryId memory_id,
        Bounds<N>& region,
        EventList& deps_out
    );

    void finalize_access(TaskGraph& stage, EventId event_id);

    void commit(TaskGraph& stage);

  private:
    std::unique_lock<std::shared_mutex> m_lock;
    std::shared_ptr<ArrayDescriptor<N>> m_instance;
    std::vector<std::pair<size_t, EventId>> m_write_events;
};
}  // namespace kmm