#pragma once

#include "kmm/core/identifiers.hpp"
#include "kmm/utils/geometry.hpp"

namespace kmm {

template<size_t N>
struct DataChunk {
    MemoryId owner_id;
    Index<N> offset;
    Dim<N> size;
};

template<size_t N>
class DataDistribution {
    DataDistribution();

    DataDistribution(Dim<N> array_size, Dim<N> chunk_size, std::vector<MemoryId> memories);

  public:
    static DataDistribution from_chunks(
        Dim<N> array_size,
        std::vector<DataChunk<N>> chunks,
        std::vector<BufferId>& buffers
    );

    size_t region_to_chunk_index(Bounds<N> region) const;

    DataChunk<N> chunk(size_t index) const;

    size_t num_chunks() const {
        return m_memories.size();
    }

    Dim<N> chunk_size() const {
        return m_chunk_size;
    }

    Dim<N> array_size() const {
        return m_array_size;
    }

  protected:
    Dim<N> m_array_size = Dim<N>::zero();
    Dim<N> m_chunk_size = Dim<N>::zero();
    std::array<size_t, N> m_chunks_count;
    std::vector<MemoryId> m_memories;
};

#define KMM_INSTANTIATE_ARRAY_IMPL(NAME) \
    template class NAME<0>; /* NOLINT */ \
    template class NAME<1>; /* NOLINT */ \
    template class NAME<2>; /* NOLINT */ \
    template class NAME<3>; /* NOLINT */ \
    template class NAME<4>; /* NOLINT */ \
    template class NAME<5>; /* NOLINT */ \
    template class NAME<6>; /* NOLINT */

}  // namespace kmm