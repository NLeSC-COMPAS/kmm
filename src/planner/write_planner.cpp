#include "kmm/planner/write_planner.hpp"
#include "kmm/runtime/task_graph.hpp"

namespace kmm {

template<size_t N>
ArrayWritePlanner<N>::ArrayWritePlanner() {}

template<size_t N>
ArrayWritePlanner<N>::ArrayWritePlanner(std::shared_ptr<ArrayDescriptor<N>> instance) :
    m_instance(std::move(instance)) {
    KMM_ASSERT(m_instance);
    KMM_ASSERT(m_instance->m_num_writers == 0);
    KMM_ASSERT(m_instance->m_num_readers == 0);
    m_instance->m_num_writers++;
}

template<size_t N>
ArrayWritePlanner<N>::~ArrayWritePlanner() {
    if (m_instance) {
        m_instance->m_num_writers--;
    }
}

template<size_t N>
BufferRequirement ArrayWritePlanner<N>::prepare_access(
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
    deps_out.insert_all(buffer.last_access_events);
    m_write_events.push_back({chunk_index, EventId()});

    return BufferRequirement {
        .buffer_id = buffer.id,
        .memory_id = memory_id,
        .access_mode = AccessMode::ReadWrite};
}

template<size_t N>
void ArrayWritePlanner<N>::finalize_access(TaskGraphStage& stage, EventId event_id) {
    KMM_ASSERT(m_write_events.size() > 0);
    m_write_events.back().second = event_id;
}

template<size_t N>
void ArrayWritePlanner<N>::commit(TaskGraphStage& stage) {
    KMM_ASSERT(m_instance);
    std::sort(m_write_events.begin(), m_write_events.end(), [&](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    size_t begin = 0;
    while (begin < m_write_events.size()) {
        size_t chunk_index = m_write_events[begin].first;
        auto& buffer = m_instance->m_buffers[chunk_index];

        EventId write_event;
        size_t end = begin;

        while (end < m_write_events.size() && chunk_index == m_write_events[end].first) {
            end++;
        }

        if (begin + 1 == end) {
            write_event = m_write_events[begin].second;
        } else {
            EventList deps;

            for (size_t i = begin; i < end; i++) {
                deps.push_back(m_write_events[i].second);
            }

            write_event = stage.join_events(std::move(deps));
        }

        buffer.last_write_event = write_event;
        buffer.last_access_events = {write_event};

        begin = end;
    }

    m_write_events.clear();
}

}  // namespace kmm