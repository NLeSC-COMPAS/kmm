#include "kmm/planner/read_planner.hpp"

namespace kmm {

template<size_t N>
ArrayReadPlanner<N>::ArrayReadPlanner(std::shared_ptr<ArrayDescriptor<N>> instance) :
    m_instance(std::move(instance)) {
    KMM_ASSERT(m_instance);
    KMM_ASSERT(m_instance->m_num_writers == 0);
    m_instance->m_num_readers++;
    fprintf(stderr, "increment readers for %p\n", m_instance.get());
}

template<size_t N>
ArrayReadPlanner<N>::~ArrayReadPlanner() {
    if (m_instance) {
        fprintf(stderr, "decrement readers for %p\n", m_instance.get());
        m_instance->m_num_readers--;
    }
}

template<size_t N>
BufferRequirement ArrayReadPlanner<N>::prepare_access(
    TaskGraphStage& stage,
    MemoryId memory_id,
    Bounds<N>& region,
    EventList& deps_out
) {
    KMM_ASSERT(m_instance);
    size_t chunk_index = m_instance->m_distribution.region_to_chunk_index(region);
    auto chunk = m_instance->m_distribution.chunk(chunk_index);
    const auto& buffer = m_instance->m_buffers[chunk_index];

    region = Bounds<N>::from_offset_size(chunk.offset, chunk.size);
    deps_out.push_back(buffer.last_write_event);
    m_read_events.push_back({chunk_index, EventId()});

    return BufferRequirement {
        .buffer_id = buffer.id,
        .memory_id = memory_id,
        .access_mode = AccessMode::Read};
}

template<size_t N>
void ArrayReadPlanner<N>::finalize_access(TaskGraphStage& stage, EventId event_id) {
    KMM_ASSERT(m_read_events.size() > 0);
    m_read_events.back().second = event_id;
}

template<size_t N>
void ArrayReadPlanner<N>::commit(TaskGraphStage& stage) {
    KMM_ASSERT(m_instance);

    for (const auto& [chunk_index, event_id] : m_read_events) {
        m_instance->m_buffers[chunk_index].last_access_events.push_back(event_id);
    }

    m_read_events.clear();
}

}  // namespace kmm