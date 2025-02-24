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
  public:
    DataDistribution(Dim<N> array_size, std::vector<DataChunk<N>> chunks);

    size_t region_to_chunk_index(Bounds<N> region) const;

    DataChunk<N> chunk(size_t index) const;

    size_t num_chunks() const {
        return m_chunks.size();
    }

    Dim<N> chunk_size() const {
        return m_chunk_size;
    }

    Dim<N> array_size() const {
        return m_array_size;
    }

  protected:
    std::vector<DataChunk<N>> m_chunks;
    std::vector<size_t> m_mapping;
    std::array<size_t, N> m_chunks_count;
    Dim<N> m_array_size = Dim<N>::zero();
    Dim<N> m_chunk_size = Dim<N>::zero();
};

}  // namespace kmm