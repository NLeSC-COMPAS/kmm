#include "spdlog/spdlog.h"

#include "kmm/core/domain.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

Domain TileDomain::operator()(const SystemInfo& info, ExecutionSpace space) const {
    std::vector<ResourceId> devices;

    if (space == ExecutionSpace::Host) {
        for (const auto& resource : info.resources()) {
            devices.push_back(ResourceId::host(resource.device_affinity()));
        }

        if (devices.empty()) {
            devices.push_back(ResourceId::host());
        }
    } else if (space == ExecutionSpace::Device) {
        devices = info.resources();
    }

    if (devices.empty()) {
        throw std::runtime_error("cannot partition work, no devices found");
    }

    std::vector<DomainChunk> chunks;

    if (m_domain_size.is_empty()) {
        return {chunks};
    }

    if (m_tile_size.is_empty()) {
        throw std::runtime_error(fmt::format("invalid chunk size: {}", m_tile_size));
    }

    std::array<int64_t, DOMAIN_DIMS> num_chunks;

    for (size_t i = 0; i < DOMAIN_DIMS; i++) {
        num_chunks[i] = div_ceil(m_domain_size[i].size(), m_tile_size[i]);
    }

    size_t owner_id = 0;
    auto offset = DomainPoint {};
    auto size = DomainDim {};

    for (int64_t z = 0; z < num_chunks[2]; z++) {
        for (int64_t y = 0; y < num_chunks[1]; y++) {
            for (int64_t x = 0; x < num_chunks[0]; x++) {
                auto current = Point<3> {x, y, z};

                for (size_t i = 0; i < DOMAIN_DIMS; i++) {
                    offset[i] = m_domain_size[i].begin + current[i] * m_tile_size[i];
                    size[i] = std::min(m_tile_size[i], m_domain_size[i].end - offset[i]);
                }

                chunks.push_back({devices[owner_id], offset, size});
                owner_id = (owner_id + 1) % devices.size();
            }
        }
    }

    return {std::move(chunks)};
}

}  // namespace kmm
