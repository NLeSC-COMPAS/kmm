#pragma once

#include "kmm/core/distribution.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/utils/geometry.hpp"

namespace kmm {

class Runtime;
class TaskGraphStage;

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
    ArrayDescriptor(
        Distribution<N> distribution,
        DataType dtype,
        std::vector<BufferDescriptor> buffers
    );

    const Distribution<N>& distribution() const {
        return m_distribution;
    }

    DataType data_type() const {
        return m_dtype;
    }

    EventId join_events(TaskGraphStage& stage) const;
    void destroy(TaskGraphStage& stage);

  public:
    Distribution<N> m_distribution;
    DataType m_dtype;
    std::vector<BufferDescriptor> m_buffers;
    size_t m_num_readers = 0;
    size_t m_num_writers = 0;
};

template<size_t N>
class ArrayInstance:
    public ArrayDescriptor<N>,
    public std::enable_shared_from_this<ArrayInstance<N>> {
    KMM_NOT_COPYABLE_OR_MOVABLE(ArrayInstance)

  public:
    ArrayInstance(Runtime& rt, Distribution<N> dist, DataType dtype);
    ~ArrayInstance();

    void copy_bytes(void* data) const;
    void synchronize() const;

    const Runtime& runtime() const {
        return *m_rt;
    }

  private:
    std::shared_ptr<Runtime> m_rt;
};

KMM_INSTANTIATE_ARRAY_IMPL(ArrayDescriptor)
KMM_INSTANTIATE_ARRAY_IMPL(ArrayInstance)

}  // namespace kmm