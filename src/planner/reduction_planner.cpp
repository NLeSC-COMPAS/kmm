#include "kmm/planner/reduction_planner.hpp"
#include "kmm/runtime/task_graph.hpp"

namespace kmm {

template<size_t N>
ArrayReductionPlanner<N>::ArrayReductionPlanner(
    std::shared_ptr<ArrayDescriptor<N>> instance,
    Reduction op
) :
    m_instance(std::move(instance)),
    m_reduction(op) {
    KMM_ASSERT(m_instance);
    KMM_ASSERT(m_instance->m_num_writers == 0);
    KMM_ASSERT(m_instance->m_num_readers == 0);
    m_instance->m_num_writers++;
}

template<size_t N>
ArrayReductionPlanner<N>::~ArrayReductionPlanner() {
    m_instance->m_num_writers--;
}

template<size_t N>
BufferRequirement ArrayReductionPlanner<N>::prepare_access(
    TaskGraphStage& stage,
    MemoryId memory_id,
    Bounds<N>& region,
    size_t replication_factor,
    EventList& deps_out
) {
    KMM_TODO();
}

template<size_t N>
void ArrayReductionPlanner<N>::finalize_access(TaskGraphStage& stage, EventId event_id) {
    KMM_TODO();
}

template<size_t N>
void ArrayReductionPlanner<N>::commit(TaskGraphStage& stage) {
    KMM_TODO();
}

}  // namespace kmm