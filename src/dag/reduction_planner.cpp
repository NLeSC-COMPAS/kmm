#include <algorithm>

#include "kmm/dag/array_planner.hpp"
#include "kmm/dag/reduction_planner.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {

BufferRequirement LocalReductionPlanner::prepare_access(
    TaskGraph& graph,
    MemoryId memory_id,
    size_t replication_factor
) {
    auto num_elements = checked_mul(m_num_elements, replication_factor);
    auto layout = DataLayout::for_type(m_dtype).repeat(num_elements);

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

    m_inputs.push_back(ReductionInput {
        .buffer_id = buffer_id,
        .memory_id = memory_id,
        .dependencies = {event_id},
        .num_inputs_per_output = replication_factor});

    return {
        .buffer_id = buffer_id,  //
        .memory_id = memory_id,
        .access_mode = AccessMode::Exclusive};
}

EventId insert_multi_reduction(
    TaskGraph& graph,
    MemoryId memory_id,
    BufferId buffer_id,
    ReductionOutput reduction,
    std::vector<ReductionInput> inputs
) {
    auto dtype = reduction.data_type;
    auto op = reduction.operation;
    auto num_elements = reduction.num_outputs;

    if (inputs.size() == 1) {
        auto& input = inputs[0];

        return graph.insert_reduction(
            input.buffer_id,
            input.memory_id,
            buffer_id,
            memory_id,
            ReductionDef {
                .operation = op,
                .data_type = dtype,
                .num_outputs = num_elements,
                .num_inputs_per_output = input.num_inputs_per_output},
            std::move(input.dependencies)
        );
    }

    auto scratch_layout = DataLayout::for_type(dtype).repeat(num_elements).repeat(inputs.size());
    auto scratch_id = graph.create_buffer(scratch_layout);
    auto scratch_deps = EventList {};

    for (size_t i = 0; i < inputs.size(); i++) {
        auto& input = inputs[i];

        EventId event_id = graph.insert_reduction(
            input.buffer_id,
            input.memory_id,
            scratch_id,
            memory_id,
            ReductionDef {
                .operation = op,
                .data_type = dtype,
                .num_outputs = num_elements,
                .num_inputs_per_output = input.num_inputs_per_output,
                .input_offset_elements = 0,
                .output_offset_elements = i * num_elements},
            std::move(input.dependencies)
        );

        scratch_deps.push_back(event_id);
    }

    auto event_id = graph.insert_reduction(
        scratch_id,
        memory_id,
        buffer_id,
        memory_id,
        ReductionDef {
            .operation = op,
            .data_type = dtype,
            .num_outputs = num_elements,
            .num_inputs_per_output = inputs.size(),
        },
        std::move(scratch_deps)
    );

    graph.delete_buffer(scratch_id, {event_id});

    return event_id;
}

EventId insert_hierarchical_reduction(
    TaskGraph& graph,
    BufferId final_buffer_id,
    MemoryId final_memory_id,
    ReductionOutput reduction,
    std::vector<ReductionInput> inputs
) {
    auto dtype = reduction.data_type;
    auto num_elements = reduction.num_outputs;

    if (std::all_of(inputs.begin(), inputs.end(), [&](const auto& a) {
            return a.memory_id == final_memory_id;
        })) {
        return insert_multi_reduction(graph, final_memory_id, final_buffer_id, reduction, inputs);
    }

    std::stable_sort(inputs.begin(), inputs.end(), [&](const auto& a, const auto& b) {
        return a.memory_id < b.memory_id;
    });

    auto temporary_layout = DataLayout::for_type(dtype).repeat(num_elements);
    auto temporary_buffers = std::vector<BufferId> {};
    std::vector<ReductionInput> result_per_device;
    size_t cursor = 0;

    while (cursor < inputs.size()) {
        auto memory_id = inputs[cursor].memory_id;
        size_t begin = cursor;

        while (cursor < inputs.size() && memory_id == inputs[cursor].memory_id) {
            cursor++;
        }

        size_t length = cursor - begin;

        // Special case: if there is only one buffer having one input, then we do not need to
        // create a local scratch buffer.
        if (length == 1 && inputs[begin].num_inputs_per_output == 1) {
            result_per_device.push_back(inputs[begin]);
            continue;
        }

        auto local_inputs = std::vector<ReductionInput> {&inputs[begin], &inputs[cursor]};

        auto local_buffer_id = graph.create_buffer(temporary_layout);
        auto event_id = insert_multi_reduction(  //
            graph,
            memory_id,
            local_buffer_id,
            reduction,
            local_inputs
        );

        temporary_buffers.push_back(local_buffer_id);
        result_per_device.push_back(ReductionInput {
            .buffer_id = local_buffer_id,
            .memory_id = memory_id,
            .dependencies = {event_id}});
    }

    auto event_id = insert_multi_reduction(
        graph,
        final_memory_id,
        final_buffer_id,
        reduction,
        result_per_device
    );

    for (auto& buffer_id : temporary_buffers) {
        graph.delete_buffer(buffer_id);
    }

    return event_id;
}

EventId LocalReductionPlanner::finalize(TaskGraph& graph, BufferId buffer_id, MemoryId memory_id) {
    auto reduction = ReductionOutput {
        .operation = m_reduction,  //
        .data_type = m_dtype,
        .num_outputs = m_num_elements};

    auto event_id = insert_hierarchical_reduction(graph, buffer_id, memory_id, reduction, m_inputs);

    for (const auto& input : m_inputs) {
        graph.delete_buffer(input.buffer_id, {event_id});
    }

    m_inputs.clear();
    return event_id;
}

template<size_t N>
ReductionPlanner<N>::ReductionPlanner(
    const ArrayInstance<N>* instance,
    DataType data_type,
    Reduction operation
) :
    m_instance(instance),
    m_dtype(data_type),
    m_reduction(operation) {
    KMM_ASSERT(m_instance);
    size_t num_chunks = m_instance->distribution().num_chunks();
    m_partial_inputs.resize(num_chunks);
}

template<size_t N>
BufferRequirement ReductionPlanner<N>::prepare_access(
    TaskGraph& graph,
    MemoryId memory_id,
    Bounds<N>& access_region,
    size_t replication_factor
) {
    KMM_ASSERT(m_instance != nullptr);
    const auto& dist = m_instance->distribution();
    auto chunk_index = dist.region_to_chunk_index(access_region);
    auto chunk = dist.chunk(chunk_index);

    access_region = Bounds<N>::from_offset_size(chunk.offset, chunk.size);

    if (m_partial_inputs[chunk_index] == nullptr) {
        m_partial_inputs[chunk_index] =
            std::make_unique<LocalReductionPlanner>(chunk.size.volume(), m_dtype, m_reduction);
    }

    m_last_access = m_partial_inputs[chunk_index].get();

    return m_partial_inputs[chunk_index]->prepare_access(graph, memory_id, replication_factor);
}

template<size_t N>
void ReductionPlanner<N>::finalize_access(TaskGraph& graph, EventId event_id) {
    KMM_ASSERT(m_last_access != nullptr);
    m_last_access->m_inputs.back().dependencies.push_back(event_id);
    m_last_access = nullptr;
}

template<size_t N>
EventId ReductionPlanner<N>::finalize(TaskGraph& graph) {
    KMM_ASSERT(m_instance != nullptr);
    auto deps = EventList();
    const auto& dist = m_instance->distribution();

    for (size_t i = 0; i < dist.num_chunks(); i++) {
        if (m_partial_inputs[i] == nullptr) {
            continue;
        }

        auto event_id =
            m_partial_inputs[i]->finalize(graph, m_instance->buffers()[i], dist.chunk(i).owner_id);

        deps.push_back(event_id);
    }

    return graph.join_events(deps);
}

KMM_INSTANTIATE_ARRAY_IMPL(ReductionPlanner)

}  // namespace kmm