#pragma once

#include <cstddef>
#include <vector>

#include "kmm/core/identifiers.hpp"
#include "kmm/core/range.hpp"
#include "kmm/core/system_info.hpp"
#include "kmm/core/task.hpp"

namespace kmm {

struct WorkChunk {
    ProcessorId owner_id;
    WorkIndex offset;
    WorkDim size;
};

struct WorkPartition {
    std::vector<WorkChunk> chunks;
};

template<typename P>
struct IntoWorkPartition {
    static WorkPartition call(
        P partitioner,
        WorkBounds index_space,
        const SystemInfo& info,
        ExecutionSpace space
    ) {
        return (partitioner)(index_space, info, space);
    }
};

struct ChunkPartitioner {
    ChunkPartitioner(WorkDim chunk_size) : m_chunk_size(chunk_size) {}
    ChunkPartitioner(
        int64_t x,
        int64_t y = std::numeric_limits<int64_t>::max(),
        int64_t z = std::numeric_limits<int64_t>::max()
    ) :
        m_chunk_size(x, y, z) {}

    WorkPartition operator()(WorkBounds index_space, const SystemInfo& info, ExecutionSpace space)
        const;

  private:
    WorkDim m_chunk_size;
};

}  // namespace kmm
