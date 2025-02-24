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
DataDistribution<N>::DataDistribution(Dim<N> array_size, std::vector<DataChunk<N>> chunks) :
    m_chunks(std::move(chunks)),
    m_array_size(array_size) {
    for (const auto& chunk : m_chunks) {
        if (chunk.offset == Index<N>::zero()) {
            m_chunk_size = chunk.size;
        }
    }

    if (m_chunk_size.is_empty()) {
        throw std::runtime_error("chunk size cannot be empty");
    }

    for (size_t i = 0; compare_less(i, N); i++) {
        m_chunks_count[i] = checked_cast<size_t>(div_ceil(array_size[i], m_chunk_size[i]));
    }

    static constexpr size_t INVALID_INDEX = static_cast<size_t>(-1);
    m_mapping = std::vector<size_t>(m_chunks.size(), INVALID_INDEX);

    for (size_t index = 0; index < m_chunks.size(); index++) {
        const auto chunk = m_chunks[index];
        size_t linear_index = 0;
        Index<N> expected_offset;
        Dim<N> expected_size;

        for (size_t i = 0; compare_less(i, N); i++) {
            auto k = div_floor(chunk.offset[i], m_chunk_size[i]);

            expected_offset[i] = k * m_chunk_size[i];
            expected_size[i] = std::min(m_chunk_size[i], array_size[i] - expected_offset[i]);

            linear_index = linear_index * m_chunks_count[i] + static_cast<size_t>(k);
        }

        if (chunk.offset != expected_offset || chunk.size != expected_size) {
            throw std::runtime_error(fmt::format(
                "invalid write access pattern, the region {} is not aligned to the chunk size of {}",
                Bounds<N>::from_offset_size(chunk.offset, chunk.size),
                m_chunk_size
            ));
        }

        if (m_mapping[linear_index] != INVALID_INDEX) {
            throw std::runtime_error(fmt::format(
                "invalid write access pattern, the region {} is written to by more one task",
                Bounds<N>::from_offset_size(expected_offset, expected_size)
            ));
        }

        m_mapping[linear_index] = index;
    }

    for (size_t index : m_mapping) {
        if (index == INVALID_INDEX) {
            auto region = index2region(index, m_chunks_count, m_chunk_size, m_array_size);
            throw std::runtime_error(
                fmt::format("invalid write access pattern, no task writes to region {}", region)
            );
        }
    }
}

template<size_t N>
size_t DataDistribution<N>::region_to_chunk_index(Bounds<N> region) const {
    size_t index = 0;

    for (size_t i = 0; compare_less(i, N); i++) {
        auto k = div_floor(region.begin(i), m_chunk_size[i]);

        if (!in_range(k, m_chunks_count[i])) {
            throw std::out_of_range(fmt::format(
                "invalid read pattern, the region {} exceeds the array dimensions {}",
                region,
                m_array_size
            ));
        }

        if (region.end(i) > (k + 1) * m_chunk_size[i]) {
            throw std::out_of_range(fmt::format(
                "invalid read pattern, the region {} does not align to the chunk size of {}",
                region,
                m_chunk_size
            ));
        }

        index = index * m_chunks_count[i] + static_cast<size_t>(k);
    }

    return index;
}

template<size_t N>
DataChunk<N> DataDistribution<N>::chunk(size_t index) const {
    if (index >= m_chunks.size()) {
        throw std::runtime_error(fmt::format(
            "chunk {} is out of range, there are only {} chunk(s)",
            index,
            m_chunks.size()
        ));
    }

    return m_chunks[index];
}

#define INSTANTIATE_ARRAY_IMPL(NAME)     \
    template class NAME<0>; /* NOLINT */ \
    template class NAME<1>; /* NOLINT */ \
    template class NAME<2>; /* NOLINT */ \
    template class NAME<3>; /* NOLINT */ \
    template class NAME<4>; /* NOLINT */ \
    template class NAME<5>; /* NOLINT */ \
    template class NAME<6>; /* NOLINT */

INSTANTIATE_ARRAY_IMPL(DataDistribution)

}  // namespace kmm