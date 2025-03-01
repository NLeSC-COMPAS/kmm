#include "kmm/core/buffer.hpp"
#include "kmm/dag/dist_data_planner.hpp"
#include "kmm/dag/dist_reduction_planner.hpp"

namespace kmm {

template<size_t N>
BufferRequirement DistReductionPlanner<N>::add_chunk(
    TaskGraph& graph,
    MemoryId memory_id,
    Bounds<N> access_region,
    size_t replication_factor
) {
    auto it = m_partial_inputs.find(access_region);

    if (it == m_partial_inputs.end()) {
        it = m_partial_inputs.emplace_hint(
            it,
            access_region,
            ReductionPlanner(checked_cast<size_t>(access_region.size()), m_dtype, m_reduction)
        );
    }

    return it->second.add_chunk(graph, memory_id, replication_factor);
}

template<size_t N>
std::pair<DataDistribution<N>, std::vector<BufferId>> DistReductionPlanner<N>::finalize(
    TaskGraph& graph
) {
    DistDataPlanner<N> planner(m_shape, DataLayout::for_type(m_dtype));

    for (auto& p : m_partial_inputs) {
        auto access_region = p.first;
        auto& reduction = p.second;

        auto memory_id = reduction.affinity_memory();
        auto buffer_id = planner.add_chunk(graph, memory_id, access_region).buffer_id;
        reduction.finalize(graph, buffer_id, memory_id);
    }

    return planner.finalize(graph);
}

KMM_INSTANTIATE_ARRAY_IMPL(DistReductionPlanner)

}  // namespace kmm