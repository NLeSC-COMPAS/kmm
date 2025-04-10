#pragma once

#include <algorithm>
#include <mutex>
#include <shared_mutex>

#include "kmm/core/distribution.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/utils/geometry.hpp"

namespace kmm {

class TaskGraph;

struct BufferDescriptor {
    BufferId id;
    BufferLayout layout;
    EventId last_write_event {};
    EventList last_access_events {};
};

template<size_t N>
class ArrayDescriptor {
    KMM_NOT_COPYABLE_OR_MOVABLE(ArrayDescriptor)

  public:
    ArrayDescriptor(TaskGraph& stage, Distribution<N> distribution, DataType dtype);

    const Distribution<N>& distribution() const {
        return m_distribution;
    }

    DataType data_type() const {
        return m_dtype;
    }

    EventId copy_bytes_into_buffer(TaskGraph& stage, void* dst_data);
    EventId copy_bytes_from_buffer(TaskGraph& stage, const void* src_data);

    EventId join_events(TaskGraph& stage) const;
    void destroy(TaskGraph& stage);

  public:
    mutable std::shared_mutex m_mutex;
    Distribution<N> m_distribution;
    DataType m_dtype;
    std::vector<BufferDescriptor> m_buffers;
};

}  // namespace kmm