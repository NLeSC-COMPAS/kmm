#include "spdlog/spdlog.h"

#include "kmm/api/array.hpp"
#include "kmm/internals/worker.hpp"
#include "kmm/memops/host_copy.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

template<size_t N>
Rect<N> index2region(
    size_t index,
    std::array<size_t, N> num_chunks,
    Dim<N> chunk_size,
    Dim<N> array_size) {
    Point<N> offset;
    Dim<N> sizes;

    for (size_t j = 0; j < N; j++) {
        size_t i = N - 1 - j;
        auto k = index % num_chunks[i];
        index /= num_chunks[i];

        offset[i] = int64_t(k) * chunk_size[i];
        sizes[i] = std::min(chunk_size[i], array_size[i] - offset[i]);
    }

    return {offset, sizes};
}

template<size_t N>
ArrayBackend<N>::ArrayBackend(
    std::shared_ptr<Worker> worker,
    Dim<N> array_size,
    std::vector<ArrayChunk<N>> chunks) :
    m_worker(worker),
    m_array_size(array_size) {
    for (const auto& chunk : chunks) {
        if (chunk.offset == Point<N>::zero()) {
            m_chunk_size = chunk.size;
        }
    }

    if (m_chunk_size.is_empty()) {
        throw std::runtime_error("chunk size cannot be empty");
    }

    size_t num_total_chunks = 1;

    for (size_t i = 0; i < N; i++) {
        m_num_chunks[i] = checked_cast<size_t>(div_ceil(array_size[i], m_chunk_size[i]));
        num_total_chunks *= m_num_chunks[i];
    }

    static constexpr size_t INVALID_INDEX = static_cast<size_t>(-1);
    std::vector<size_t> buffer_locs(num_total_chunks, INVALID_INDEX);

    for (const auto& chunk : chunks) {
        size_t buffer_index = 0;
        bool is_valid = true;
        Point<N> expected_offset;
        Dim<N> expected_size;

        for (size_t i = 0; i < N; i++) {
            auto k = div_floor(chunk.offset[i], m_chunk_size[i]);

            expected_offset[i] = k * m_chunk_size[i];
            expected_size[i] = std::min(m_chunk_size[i], array_size[i] - expected_offset[i]);

            buffer_index = buffer_index * m_num_chunks[i] + static_cast<size_t>(k);
        }

        if (chunk.offset != expected_offset || chunk.size != expected_size) {
            throw std::runtime_error(fmt::format(
                "invalid write access pattern, the region {} is not aligned to the chunk size of {}",
                Rect<N>(chunk.offset, chunk.size),
                m_chunk_size));
        }

        if (buffer_locs[buffer_index] != INVALID_INDEX) {
            throw std::runtime_error(fmt::format(
                "invalid write access pattern, the region {} is written to by more one task",
                Rect<N>(expected_offset, expected_size)));
        }

        buffer_locs[buffer_index] = buffer_index;
    }

    for (size_t index : buffer_locs) {
        if (index == INVALID_INDEX) {
            auto region = index2region(index, m_num_chunks, m_chunk_size, m_array_size);
            throw std::runtime_error(
                fmt::format("invalid write access pattern, no task writes to region {}", region));
        }
    }

    for (size_t index : buffer_locs) {
        m_buffers.push_back(chunks[index].buffer_id);
    }
}

template<size_t N>
ArrayBackend<N>::~ArrayBackend() {
    m_worker->with_task_graph([&](auto& builder) {
        for (auto id : m_buffers) {
            builder.delete_buffer(id);
        }
    });
}

template<size_t N>
ArrayChunk<N> ArrayBackend<N>::find_chunk(Rect<N> region) const {
    size_t buffer_index = 0;
    Point<N> offset;
    Dim<N> sizes;

    for (size_t i = 0; i < N; i++) {
        auto k = div_floor(region.offset[i], m_chunk_size[i]);
        auto w = region.offset[i] % m_chunk_size[i] + region.sizes[i];

        if (!in_range(k, m_num_chunks[i])) {
            throw std::out_of_range(fmt::format(
                "invalid read pattern, the region {} exceeds the array dimensions {}",
                region,
                m_array_size));
        }

        if (w > m_chunk_size[i]) {
            throw std::out_of_range(fmt::format(
                "invalid read pattern, the region {} does not align to the chunk size of {}",
                region,
                m_chunk_size));
        }

        buffer_index = buffer_index * m_num_chunks[i] + static_cast<size_t>(k);
        offset[i] = k * m_chunk_size[i];
        sizes[i] = m_chunk_size[i];
    }

    // TODO?
    MemoryId memory_id = MemoryId::host();

    return {m_buffers[buffer_index], memory_id, offset, sizes};
}

template<size_t N>
ArrayChunk<N> ArrayBackend<N>::chunk(size_t index) const {
    if (index >= m_buffers.size()) {
        throw std::runtime_error(fmt::format(
            "chunk {} is out of range, there are only {} chunks",
            index,
            m_buffers.size()));
    }

    // TODO?
    MemoryId memory_id = MemoryId::host();
    auto region = index2region(index, m_num_chunks, m_chunk_size, m_array_size);

    return {m_buffers[index], memory_id, region.offset, region.sizes};
}

template<size_t N>
void ArrayBackend<N>::synchronize() const {
    auto event_id = m_worker->with_task_graph([&](TaskGraph& graph) {
        auto deps = EventList {};

        for (const auto& buffer_id : m_buffers) {
            graph.access_buffer(buffer_id, AccessMode::ReadWrite, deps);
        }

        return graph.join_events(deps);
    });

    m_worker->query_event(event_id, std::chrono::system_clock::time_point::max());

    // Access each buffer once to check for errors.
    for (size_t i = 0; i < m_buffers.size(); i++) {
        auto memory_id = this->chunk(i).owner_id;
        m_worker->access_buffer(m_buffers[i], memory_id, AccessMode::Read);
    }
}

template<size_t N>
class CopyOutTask: public Task {
  public:
    CopyOutTask(void* data, size_t element_size, Dim<N> array_size, Rect<N> region) :
        m_dst_addr(data),
        m_element_size(element_size),
        m_array_size(array_size),
        m_region(region) {}

    void execute(ExecutionContext& proc, TaskContext context) override {
        KMM_ASSERT(context.accessors.size() == 1);
        const void* src_addr = context.accessors[0].address;
        size_t src_stride = 1;
        size_t dst_stride = 1;

        CopyDescription copy(m_element_size);

        for (size_t j = 0; j < N; j++) {
            size_t i = N - j - 1;

            copy.add_dimension(
                checked_cast<size_t>(m_region.size(i)),
                checked_cast<size_t>(0),
                checked_cast<size_t>(m_region.begin(i)),
                src_stride,
                dst_stride);

            src_stride *= checked_cast<size_t>(m_region.size(i));
            dst_stride *= checked_cast<size_t>(m_array_size.get(i));
        }

        execute_copy(src_addr, m_dst_addr, copy);
    }

  private:
    void* m_dst_addr;
    size_t m_element_size;
    Dim<N> m_array_size;
    Rect<N> m_region;
};

template<size_t N>
void ArrayBackend<N>::copy_bytes(void* dest_addr, size_t element_size) const {
    auto event_id = m_worker->with_task_graph([&](TaskGraph& graph) {
        EventList deps;

        for (size_t i = 0; i < m_buffers.size(); i++) {
            auto region = index2region(i, m_num_chunks, m_chunk_size, m_array_size);

            auto task =
                std::make_shared<CopyOutTask<N>>(dest_addr, element_size, m_array_size, region);
            auto buffer = BufferRequirement {
                .buffer_id = m_buffers[i],
                .memory_id = MemoryId::host(),
                .access_mode = AccessMode::Read,
            };

            auto event_id = graph.insert_task(ProcessorId::host(), std::move(task), {buffer});

            deps.push_back(event_id);
        }

        return graph.join_events(deps);
    });

    m_worker->query_event(event_id, std::chrono::system_clock::time_point::max());
}

template<size_t N>
size_t ArrayBuilder<N>::add_chunk(TaskBuilder& builder, Rect<N> access_region) {
    auto num_elements = access_region.size();
    auto buffer_id = builder.graph.create_buffer(m_element_layout.repeat(num_elements));

    m_chunks.push_back(ArrayChunk<N> {
        .buffer_id = buffer_id,
        .owner_id = builder.memory_id,
        .offset = access_region.offset,
        .size = access_region.sizes});

    size_t buffer_index = builder.buffers.size();
    builder.buffers.emplace_back(BufferRequirement {
        .buffer_id = buffer_id,
        .memory_id = builder.memory_id,
        .access_mode = AccessMode::Exclusive});

    return buffer_index;
}

template<size_t N>
std::shared_ptr<ArrayBackend<N>> ArrayBuilder<N>::build(std::shared_ptr<Worker> worker) {
    return std::make_shared<ArrayBackend<N>>(worker, m_sizes, std::move(m_chunks));
}

BufferLayout make_layout(size_t num_elements, DataType dtype, ReductionOp reduction) {
    return BufferLayout {
        .size_in_bytes = dtype.size_in_bytes() * num_elements,
        .alignment = dtype.alignment(),
        .fill_pattern = reduction_identity_value(dtype, reduction),
    };
}

template<size_t N>
size_t ArrayReductionBuilder<N>::add_chunk(
    TaskBuilder& builder,
    Rect<N> access_region,
    size_t replication_factor) {
    auto num_elements = checked_mul(checked_cast<size_t>(access_region.size()), replication_factor);
    auto memory_id = builder.memory_id;

    auto buffer_id = builder.graph.create_buffer(make_layout(num_elements, m_dtype, m_reduction));

    m_partial_inputs[access_region].push_back(ReductionInput {
        .buffer_id = buffer_id,
        .memory_id = memory_id,
        .dependencies = {},
        .num_inputs_per_output = replication_factor});

    size_t buffer_index = builder.buffers.size();
    builder.buffers.emplace_back(BufferRequirement {
        .buffer_id = buffer_id,
        .memory_id = memory_id,
        .access_mode = AccessMode::Exclusive});

    return buffer_index;
}

template<size_t N>
std::shared_ptr<ArrayBackend<N>> ArrayReductionBuilder<N>::build(
    std::shared_ptr<Worker> worker,
    TaskGraph& graph) {
    std::vector<ArrayChunk<N>> chunks;

    for (auto& p : m_partial_inputs) {
        auto access_region = p.first;
        auto& inputs = p.second;

        MemoryId memory_id = inputs[0].memory_id;
        auto num_elements = access_region.size();

        auto buffer_id = graph.create_buffer(make_layout(num_elements, m_dtype, m_reduction));

        auto event_id =
            graph
                .insert_reduction(m_reduction, buffer_id, memory_id, m_dtype, num_elements, inputs);

        chunks.push_back(ArrayChunk<N> {
            .buffer_id = buffer_id,
            .owner_id = memory_id,
            .offset = access_region.offset,
            .size = access_region.sizes});

        for (const auto& input : inputs) {
            graph.delete_buffer(input.buffer_id, {event_id});
        }
    }

    return std::make_shared<ArrayBackend<N>>(worker, m_sizes, std::move(chunks));
}

#define INSTANTITATE_ARRAY_IMPL(NAME) \
    template class NAME<0>;           \
    template class NAME<1>;           \
    template class NAME<2>;           \
    template class NAME<3>;           \
    template class NAME<4>;           \
    template class NAME<5>;           \
    template class NAME<6>;

INSTANTITATE_ARRAY_IMPL(ArrayBackend)
INSTANTITATE_ARRAY_IMPL(ArrayBuilder)
INSTANTITATE_ARRAY_IMPL(ArrayReductionBuilder)

}  // namespace kmm