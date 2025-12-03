#include <algorithm>

#include "kmm/planner/write_planner.hpp"
#include "kmm/runtime/task_graph.hpp"

namespace kmm {

template<size_t N>
ArrayWritePlanner<N>::ArrayWritePlanner(std::shared_ptr<ArrayDescriptor<N>> instance) :
    m_lock(instance->m_mutex, std::try_to_lock),
    m_instance(std::move(instance)) {
    KMM_ASSERT(m_instance);

    if (!m_lock) {
        throw std::runtime_error(
            "array could not be locked for writing, which may happen if the "
            "same array is provided multiple times as an argument to a kernel"
        );
    }
}

template<size_t N>
ArrayWritePlanner<N>::~ArrayWritePlanner() {}

template<size_t N>
BufferRequirement ArrayWritePlanner<N>::prepare_access(
    TaskGraph& stage,
    MemoryId memory_id,
    Bounds<N>& region,
    EventList& deps_out
) {
    size_t chunk_index = m_instance->m_distribution.region_to_chunk_index(region);
    auto chunk = m_instance->m_distribution.chunk(chunk_index);
    const auto& buffer = m_instance->m_buffers[chunk_index];

    region = Bounds<N>::from_offset_size(chunk.offset, chunk.size);
    deps_out.insert_all(buffer.last_access_events);
    m_write_events.push_back({chunk_index, EventId()});

    return BufferRequirement {
        .buffer_id = buffer.id,
        .memory_id = memory_id,
        .access_mode = AccessMode::ReadWrite
    };
}

template<size_t N>
void ArrayWritePlanner<N>::finalize_access(TaskGraph& stage, EventId event_id) {
    KMM_ASSERT(!m_write_events.empty());
    m_write_events.back().second = event_id;
}

template<size_t N>
void ArrayWritePlanner<N>::commit(TaskGraph& stage) {
    std::sort(m_write_events.begin(), m_write_events.end(), [&](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    for (size_t begin = 0, end = 0; begin < m_write_events.size(); begin = end) {
        size_t chunk_index = m_write_events[begin].first;
        auto& buffer = m_instance->m_buffers[chunk_index];

        EventId write_event;
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
    }

    m_write_events.clear();
}

KMM_INSTANTIATE_ARRAY_IMPL(ArrayWritePlanner)

}  // namespace kmm