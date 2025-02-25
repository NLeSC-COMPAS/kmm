#include "kmm/dag/array_builder.hpp"
#include "kmm/dag/reduction_builder.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {

template<size_t N>
BufferRequirement ReductionBuilder<N>::add_chunk(
    TaskGraph& graph,
    MemoryId memory_id,
    Bounds<N> access_region,
    size_t replication_factor
) {
    auto num_elements = checked_mul(checked_cast<size_t>(access_region.size()), replication_factor);
    auto layout = DataLayout {
        .size_in_bytes = checked_mul(m_dtype.size_in_bytes(), num_elements),
        .alignment = m_dtype.alignment()};

    // Create a new buffer
    auto buffer_id = graph.create_buffer(layout);

    // Fill the buffer with the identity value
    auto event_id = graph.insert_fill(
        memory_id,
        buffer_id,
        FillDef(
            m_dtype.size_in_bytes(),
            num_elements,
            reduction_identity_value(m_dtype, m_reduction).data()
        )
    );

    m_partial_inputs[access_region].push_back(ReductionInput {
        .buffer_id = buffer_id,
        .memory_id = memory_id,
        .dependencies = {event_id},
        .num_inputs_per_output = replication_factor});

    return {
        .buffer_id = buffer_id,  //
        .memory_id = memory_id,
        .access_mode = AccessMode::Exclusive};
}

template<size_t N>
void ReductionBuilder<N>::add_chunks(ReductionBuilder<N>&& other) {
    KMM_ASSERT(m_sizes == other.m_sizes);
    KMM_ASSERT(m_dtype == other.m_dtype);
    KMM_ASSERT(m_reduction == other.m_reduction);

    // Cannot be the same builder
    if (this == &other) {
        return;
    }

    if (m_partial_inputs.empty()) {
        m_partial_inputs = std::move(other.m_partial_inputs);
        return;
    }

    for (auto& [region, inputs] : other.m_partial_inputs) {
        auto& dest = m_partial_inputs[region];

        if (dest.empty()) {
            dest = std::move(inputs);
        } else {
            dest.insert(
                dest.end(),
                std::make_move_iterator(inputs.begin()),
                std::make_move_iterator(inputs.end())
            );
        }
    }

    other.m_partial_inputs.clear();
}

template<size_t N>
std::pair<DataDistribution<N>, std::vector<BufferId>> ReductionBuilder<N>::build(TaskGraph& graph) {
    std::vector<BufferId> buffers;
    ArrayBuilder<N> builder(m_sizes, DataLayout::for_type(m_dtype));

    for (auto& p : m_partial_inputs) {
        auto access_region = p.first;
        auto& inputs = p.second;

        MemoryId memory_id = inputs[0].memory_id;
        auto buffer_id = builder.add_chunk(graph, memory_id, access_region).buffer_id;

        auto num_elements = checked_cast<size_t>(access_region.size());
        auto reduction = ReductionOutput {
            .operation = m_reduction,  //
            .data_type = m_dtype,
            .num_outputs = num_elements};

        auto event_id = graph.insert_multilevel_reduction(buffer_id, memory_id, reduction, inputs);

        for (const auto& input : inputs) {
            graph.delete_buffer(input.buffer_id, {event_id});
        }
    }

    return builder.build(graph);
}

KMM_INSTANTIATE_ARRAY_IMPL(ReductionBuilder)

}  // namespace kmm