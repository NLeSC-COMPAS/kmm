#pragma once

#include "distribution.hpp"
#include "task_graph.hpp"

#include "kmm/utils/macros.hpp"

namespace kmm {

template<size_t N>
class ArrayInstance {
    KMM_NOT_COPYABLE(ArrayInstance)

  private:
    ArrayInstance(Distribution<N> distribution, std::vector<BufferId> buffers, DataType data_type) :
        m_distribution(std::move(distribution)),
        m_buffers(std::move(buffers)),
        m_dtype(data_type) {}

  public:
    ArrayInstance() = default;

    static std::unique_ptr<ArrayInstance> instantiate(
        TaskGraph& graph,
        Distribution<N> dist,
        DataType data_type
    );

    void destroy(TaskGraph& graph);

    const Distribution<N>& distribution() const {
        return m_distribution;
    }

    const std::vector<BufferId>& buffers() const {
        return m_buffers;
    }

    DataType data_type() const {
        return m_dtype;
    }

  private:
    Distribution<N> m_distribution;
    std::vector<BufferId> m_buffers;
    DataType m_dtype;
};

template<size_t N>
class ArrayReadPlanner {
  public:
    ArrayReadPlanner(const ArrayInstance<N>* instance = nullptr) : m_instance(instance) {}

    BufferRequirement prepare_access(TaskGraph& graph, MemoryId memory_id, Bounds<N>& region);

    void finalize_access(TaskGraph& graph, EventId event_id);

  private:
    const ArrayInstance<N>* m_instance;
};

template<size_t N>
class ArrayWritePlanner {
  public:
    ArrayWritePlanner(const ArrayInstance<N>* instance = nullptr) : m_instance(instance) {}

    BufferRequirement prepare_access(TaskGraph& graph, MemoryId memory_id, Bounds<N>& region);

    void finalize_access(TaskGraph& graph, EventId event_id);

  private:
    const ArrayInstance<N>* m_instance;
};

}  // namespace kmm
