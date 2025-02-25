#include "spdlog/spdlog.h"

#include "kmm/dag/work_distribution.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

WorkDistribution ChunkDist::operator()(const SystemInfo& info, ExecutionSpace space) const {
    std::vector<ProcessorId> devices;

    if (space == ExecutionSpace::Host) {
        devices.push_back(ProcessorId::host());
    } else if (space == ExecutionSpace::Device) {
        for (size_t i = 0; i < info.num_devices(); i++) {
            devices.push_back(DeviceId(i));
        }

        if (devices.empty()) {
            throw std::runtime_error("no devices found, cannot partition work");
        }
    }

    std::vector<WorkChunk> chunks;

    if (m_total_size.is_empty()) {
        return {chunks};
    }

    if (m_chunk_size.is_empty()) {
        throw std::runtime_error(fmt::format("invalid chunk size: {}", m_chunk_size));
    }

    std::array<int64_t, WORK_DIMS> num_chunks;

    for (size_t i = 0; i < WORK_DIMS; i++) {
        num_chunks[i] = div_ceil(m_total_size[i].size(), m_chunk_size[i]);
    }

    size_t owner_id = 0;
    auto offset = WorkIndex {};
    auto size = WorkDim {};

    for (int64_t z = 0; z < num_chunks[2]; z++) {
        for (int64_t y = 0; y < num_chunks[1]; y++) {
            for (int64_t x = 0; x < num_chunks[0]; x++) {
                auto current = Index<3> {x, y, z};

                for (size_t i = 0; i < WORK_DIMS; i++) {
                    offset[i] = m_total_size[i].begin + current[i] * m_chunk_size[i];
                    size[i] = std::min(m_chunk_size[i], m_total_size[i].end - offset[i]);
                }

                chunks.push_back({devices[owner_id], offset, size});
                owner_id = (owner_id + 1) % devices.size();
            }
        }
    }

    return {std::move(chunks)};
}

}  // namespace kmm
