#pragma once

#include "distribution.hpp"
#include "task_graph.hpp"

#include "kmm/utils/macros.hpp"

namespace kmm {

template<size_t N>
class ArrayInstance {
    KMM_NOT_COPYABLE(ArrayInstance)

  private:
    ArrayInstance(
        Distribution<N> distribution,
        std::vector<BufferId> buffers,
        DataLayout element_layout
    ) :
        m_distribution(std::move(distribution)),
        m_buffers(std::move(buffers)),
        m_element_layout(element_layout) {}

  public:
    ArrayInstance() = default;

    static ArrayInstance instantiate(
        TaskGraph& graph,
        Distribution<N> dist,
        DataLayout element_layout
    );

    void destroy(TaskGraph& graph);

    const Distribution<N>& distribution() const {
        return m_distribution;
    }

    const std::vector<BufferId>& buffers() const {
        return m_buffers;
    }

    DataLayout element_layout() const {
        return m_element_layout;
    }

  private:
    Distribution<N> m_distribution;
    std::vector<BufferId> m_buffers;
    DataLayout m_element_layout;
};

template<size_t N>
class ArrayReadPlanner {
  public:
    ArrayReadPlanner(const ArrayInstance<N>* instance = nullptr) : m_instance(instance) {}

    BufferRequirement plan_access(TaskGraph& graph, MemoryId memory_id, Bounds<N>& region);

    void finalize(TaskGraph& graph, const EventList& access_events);

  private:
    const ArrayInstance<N>* m_instance;
};

template<size_t N>
class ArrayWritePlanner {
  public:
    ArrayWritePlanner(const ArrayInstance<N>* instance = nullptr) : m_instance(instance) {}

    BufferRequirement plan_access(TaskGraph& graph, MemoryId memory_id, Bounds<N>& region);

    void finalize(TaskGraph& graph, const EventList& access_events);

  private:
    const ArrayInstance<N>* m_instance;
};

}  // namespace kmm