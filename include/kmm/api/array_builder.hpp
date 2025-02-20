#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/utils/geometry.hpp"
#include "kmm/utils/view.hpp"

namespace kmm {

class TaskGraph;

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

    const std::vector<DataChunk<N>>& chunks() const {
        return m_chunks;
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

template<size_t N>
class ArrayBuilder {
  public:
    ArrayBuilder(Dim<N> sizes, DataLayout element_layout) :
        m_sizes(sizes),
        m_element_layout(element_layout) {}

    BufferRequirement add_chunk(TaskGraph& graph, MemoryId memory_id, Bounds<N> access_region);
    std::pair<DataDistribution<N>, std::vector<BufferId>> build(TaskGraph& graph);

    Dim<N> sizes() const {
        return m_sizes;
    }

  private:
    Dim<N> m_sizes;
    DataLayout m_element_layout;
    std::vector<DataChunk<N>> m_chunks;
    std::vector<BufferId> m_buffers;
};

template<size_t N>
class ArrayReductionBuilder {
  public:
    ArrayReductionBuilder(Dim<N> sizes, DataType data_type, Reduction operation) :
        m_sizes(sizes),
        m_dtype(data_type),
        m_reduction(operation) {}

    BufferRequirement add_chunk(
        TaskGraph& graph,
        MemoryId memory_id,
        Bounds<N> access_region,
        size_t replication_factor = 1
    );

    void add_chunks(ArrayReductionBuilder<N>&& other);

    std::pair<DataDistribution<N>, std::vector<BufferId>> build(TaskGraph& graph);

    Dim<N> sizes() const {
        return m_sizes;
    }

  private:
    Dim<N> m_sizes;
    DataType m_dtype;
    Reduction m_reduction;
    std::unordered_map<Bounds<N>, std::vector<ReductionInput>> m_partial_inputs;
};

}  // namespace kmm