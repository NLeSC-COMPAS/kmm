#pragma once

#include <cstddef>
#include <vector>

#include "kmm/core/identifiers.hpp"
#include "kmm/core/system_info.hpp"
#include "kmm/core/task.hpp"
#include "kmm/utils/geometry.hpp"

namespace kmm {

/**
 * Constant for the number of dimensions in the work space.
 */
static constexpr size_t WORK_DIMS = 3;

/**
 * Type alias for the index type used in the work space.
 */
using WorkIndex = Index<WORK_DIMS>;

/**
 * Type alias for the size of the work space.
 */
using WorkDim = Dim<WORK_DIMS>;

/**
 * Type alias for the size of the work space.
 */
using WorkBounds = Bounds<WORK_DIMS>;

struct WorkChunk {
    ProcessorId owner_id;
    WorkIndex offset;
    WorkDim size;
};

struct WorkDistribution {
    std::vector<WorkChunk> chunks;
};

template<typename P>
struct IntoWorkDistribution {
    static WorkDistribution call(P partition, const SystemInfo& info, ExecutionSpace space) {
        return partition;
    }
};

struct ChunkDistribution {
    ChunkDistribution(WorkDim total_size, WorkDim chunk_size) :
        m_total_size(total_size),
        m_chunk_size(chunk_size) {}

    WorkDistribution operator()(const SystemInfo& info, ExecutionSpace space) const;

  private:
    WorkBounds m_total_size;
    WorkDim m_chunk_size;
};

template<>
struct IntoWorkDistribution<ChunkDistribution> {
    static WorkDistribution call(
        ChunkDistribution partition,
        const SystemInfo& info,
        ExecutionSpace space
    ) {
        return partition(info, space);
    }
};

// Old name for `ChunkDistribution`
using ChunkDist = ChunkDistribution;

}  // namespace kmm
