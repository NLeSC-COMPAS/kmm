#pragma once

#include "kmm/core/reduction.hpp"
#include "kmm/planner/array_descriptor.hpp"

namespace kmm {

struct PartialReductionBuffer {
    size_t chunk_index;
    BufferId buffer_id;
    MemoryId memory_id;
    size_t replication_factor;
    EventId creation_event;
    EventList write_events;
};

template<size_t N>
class ArrayReductionPlanner {
    KMM_NOT_COPYABLE_OR_MOVABLE(ArrayReductionPlanner)

  public:
    ArrayReductionPlanner(std::shared_ptr<ArrayDescriptor<N>> instance, Reduction op);
    ~ArrayReductionPlanner();

    BufferRequirement prepare_access(
        TaskGraph& stage,
        MemoryId memory_id,
        Bounds<N>& region,
        size_t replication_factor,
        EventList& deps_out
    );

    void finalize_access(TaskGraph& stage, EventId event_id);

    void commit(TaskGraph& stage);

  private:
    EventId reduce_per_chunk(
        TaskGraph& stage,
        size_t chunk_index,
        PartialReductionBuffer** buffers,
        size_t num_buffers
    );

    std::pair<BufferId, EventId> reduce_per_chunk_and_memory(
        TaskGraph& stage,
        size_t chunk_index,
        MemoryId memory_id,
        PartialReductionBuffer** buffers,
        size_t num_buffers
    );

    std::unique_lock<std::shared_mutex> m_lock;
    std::shared_ptr<ArrayDescriptor<N>> m_instance;
    std::vector<PartialReductionBuffer> m_partial_buffers;
    Reduction m_reduction;
};

}  // namespace kmm