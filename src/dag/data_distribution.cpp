#include "kmm/dag/data_distribution.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

template<size_t N>
Bounds<N> index2region(
    size_t index,
    std::array<size_t, N> num_chunks,
    Dim<N> chunk_size,
    Dim<N> array_size
) {
    Bounds<N> result;

    for (size_t j = 0; compare_less(j, N); j++) {
        size_t i = N - 1 - j;
        auto k = index % num_chunks[i];
        index /= num_chunks[i];

        result[i].begin = int64_t(k) * chunk_size[i];
        result[i].end = std::min(result[i].begin + chunk_size[i], array_size[i]);
    }

    return result;
}

template<size_t N>
size_t region2index(
    Bounds<N> region,
    Dim<N> chunk_size,
    Dim<N> array_size,
    std::array<size_t, N> chunks_count
) {
    size_t index = 0;

    for (size_t i = 0; compare_less(i, N); i++) {
        auto k = div_floor(region.begin(i), chunk_size[i]);

        if (!in_range(k, chunks_count[i])) {
            throw std::out_of_range(fmt::format(
                "invalid read pattern, the region {} exceeds the array dimensions {}",
                region,
                array_size
            ));
        }

        if (region.end(i) > (k + 1) * chunk_size[i]) {
            throw std::out_of_range(fmt::format(
                "invalid read pattern, the region {} does not align to the chunk size of {}",
                region,
                chunk_size
            ));
        }

        index = index * chunks_count[i] + static_cast<size_t>(k);
    }

    return index;
}

template<size_t N>
DataDistribution<N>::DataDistribution() : DataDistribution(Dim<N>::zero(), Dim<N>::one(), {}) {}

template<size_t N>
DataDistribution<N>::DataDistribution(
    Dim<N> array_size,
    Dim<N> chunk_size,
    std::vector<MemoryId> memories
) :
    m_array_size(array_size),
    m_chunk_size(chunk_size),
    m_memories(std::move(memories)) {
    size_t total_chunk_count = 1;

    for (size_t i = 0; compare_less(i, N); i++) {
        m_chunks_count[i] = checked_cast<size_t>(div_ceil(array_size[i], chunk_size[i]));
        total_chunk_count = checked_mul(total_chunk_count, m_chunks_count[i]);
    }

    if (total_chunk_count != m_memories.size()) {
        throw std::runtime_error(fmt::format(
            "data distribution contains {} chunks, only {} memory locations provided",
            total_chunk_count,
            m_memories.size()
        ));
    }
}

template<size_t N>
DataDistribution<N> DataDistribution<N>::from_chunks(
    Dim<N> array_size,
    std::vector<DataChunk<N>> chunks,
    std::vector<BufferId>& buffers
) {
    size_t total_chunk_count = 1;
    std::array<size_t, N> chunks_count;
    Dim<N> chunk_size = Dim<N>::zero();

    for (const auto& chunk : chunks) {
        if (chunk.offset == Index<N>::zero()) {
            chunk_size = chunk.size;
        }
    }

    if (chunk_size.is_empty()) {
        throw std::runtime_error("chunk size cannot be empty");
    }

    for (size_t i = 0; compare_less(i, N); i++) {
        chunks_count[i] = checked_cast<size_t>(div_ceil(array_size[i], chunk_size[i]));
        total_chunk_count = checked_mul(total_chunk_count, chunks_count[i]);
    }

    static constexpr BufferId INVALID_BUFFER_ID = BufferId(~uint64_t(0));
    auto new_buffers = std::vector<BufferId>(total_chunk_count, INVALID_BUFFER_ID);
    auto memories = std::vector<MemoryId>(total_chunk_count, MemoryId::host());

    for (size_t index = 0; index < chunks.size(); index++) {
        const auto chunk = chunks[index];
        size_t linear_index = 0;
        Index<N> expected_offset;
        Dim<N> expected_size;

        for (size_t i = 0; compare_less(i, N); i++) {
            auto k = div_floor(chunk.offset[i], chunk_size[i]);

            expected_offset[i] = k * chunk_size[i];
            expected_size[i] = std::min(chunk_size[i], array_size[i] - expected_offset[i]);

            linear_index = linear_index * chunks_count[i] + static_cast<size_t>(k);
        }

        if (chunk.offset != expected_offset || chunk.size != expected_size) {
            throw std::runtime_error(fmt::format(
                "invalid write access pattern, the region {} is not aligned to the chunk size of {}",
                Bounds<N>::from_offset_size(chunk.offset, chunk.size),
                chunk_size
            ));
        }

        if (new_buffers[linear_index] != INVALID_BUFFER_ID) {
            throw std::runtime_error(fmt::format(
                "invalid write access pattern, the region {} is written to by more one task",
                Bounds<N>::from_offset_size(expected_offset, expected_size)
            ));
        }

        memories[linear_index] = chunk.owner_id;
        new_buffers[linear_index] = buffers[index];
    }

    for (size_t i = 0; i < total_chunk_count; i++) {
        if (new_buffers[i] == INVALID_BUFFER_ID) {
            auto region = index2region(i, chunks_count, chunk_size, array_size);

            throw std::runtime_error(
                fmt::format("invalid write access pattern, no task writes to region {}", region)
            );
        }
    }

    buffers = std::move(new_buffers);

    return {array_size, chunk_size, std::move(memories)};
}

template<size_t N>
size_t DataDistribution<N>::region_to_chunk_index(Bounds<N> region) const {
    return region2index(region, m_chunk_size, m_array_size, m_chunks_count);
}

template<size_t N>
DataChunk<N> DataDistribution<N>::chunk(size_t index) const {
    if (index >= m_memories.size()) {
        throw std::runtime_error(fmt::format(
            "chunk {} is out of range, there are only {} chunk(s)",
            index,
            m_memories.size()
        ));
    }

    auto region = index2region(index, m_chunks_count, m_chunk_size, m_array_size);

    return DataChunk<N> {
        .owner_id = m_memories[index],
        .offset = region.begin(),
        .size = region.sizes()};
}

KMM_INSTANTIATE_ARRAY_IMPL(DataDistribution)

}  // namespace kmm