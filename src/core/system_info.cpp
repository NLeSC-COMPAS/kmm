#include <algorithm>

#include "fmt/format.h"

#include "kmm/core/system_info.hpp"

namespace kmm {

DeviceInfo::DeviceInfo(DeviceId id, GPUContextHandle context, size_t num_concurrent_streams) :
    m_id(id),
    m_concurrent_stream(num_concurrent_streams) {
    GPUContextGuard guard {context};

    KMM_GPU_CHECK(gpuCtxGetDevice(&m_device_id));

    char name[1024];
    KMM_GPU_CHECK(gpuDeviceGetName(name, 1024, m_device_id));
    m_name = std::string(name);

    for (size_t i = 1; i < NUM_ATTRIBUTES; i++) {
        auto attr = GPUdevice_attribute(i);
        KMM_GPU_CHECK(gpuDeviceGetAttribute(&m_attributes[i], attr, m_device_id));
    }

    size_t ignore_free_memory;
    KMM_GPU_CHECK(gpuMemGetInfo(&ignore_free_memory, &m_memory_capacity));
}

dim3 DeviceInfo::max_block_dim() const {
    return dim3(
        checked_cast<unsigned int>(attribute(GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)),
        checked_cast<unsigned int>(attribute(GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)),
        checked_cast<unsigned int>(attribute(GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))
    );
}

dim3 DeviceInfo::max_grid_dim() const {
    return dim3(
        checked_cast<unsigned int>(attribute(GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)),
        checked_cast<unsigned int>(attribute(GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)),
        checked_cast<unsigned int>(attribute(GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z))
    );
}

int DeviceInfo::compute_capability() const {
    return (attribute(GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) * 10)
        + attribute(GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
}

int DeviceInfo::max_threads_per_block() const {
    return attribute(GPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
}

int DeviceInfo::attribute(GPUdevice_attribute attrib) const {
    if (attrib < NUM_ATTRIBUTES) {
        return m_attributes[attrib];
    }

    throw std::runtime_error("unsupported attribute requested");
}

SystemInfo::SystemInfo(std::vector<DeviceInfo> devices) : m_devices(devices) {
    bool keep_going = true;

    for (size_t stream_index = 0; keep_going; stream_index++) {
        keep_going = false;

        for (const auto& dev : devices) {
            if (stream_index < dev.num_compute_streams()) {
                m_resources.push_back(ResourceId(dev.device_id(), stream_index));
                keep_going = true;
            }
        }
    }
}

SystemInfo::SystemInfo(SystemInfo info, std::vector<ResourceId> subresources) :
    m_devices(info.m_devices),
    m_resources(std::move(subresources)) {
    for (const auto& resource : m_resources) {
        auto occurrences = std::count(info.m_resources.begin(), info.m_resources.end(), resource);

        if (occurrences == 0) {
            throw std::runtime_error(fmt::format("invalid resource: {}", resource));
        }
    }
}

size_t SystemInfo::num_devices() const {
    return m_devices.size();
}

const DeviceInfo& SystemInfo::device(DeviceId id) const {
    return m_devices.at(id.get());
}

const DeviceInfo& SystemInfo::device_by_ordinal(GPUdevice ordinal) const {
    for (const auto& device : m_devices) {
        if (device.device_ordinal() == ordinal) {
            return device;
        }
    }

    throw std::runtime_error(fmt::format("cannot find device with ordinal {}", ordinal));
}

std::vector<ResourceId> SystemInfo::resources() const {
    return m_resources;
}

std::vector<MemoryId> SystemInfo::memories() const {
    std::vector<MemoryId> result {MemoryId::host()};
    for (const auto& device : m_devices) {
        result.push_back(device.memory_id());
    }

    return result;
}

MemoryId SystemInfo::affinity_memory(DeviceId device_id) const {
    return device(device_id).memory_id();
}

MemoryId SystemInfo::affinity_memory(ResourceId proc_id) const {
    if (proc_id.is_device()) {
        return affinity_memory(proc_id.as_device());
    } else {
        return MemoryId::host();
    }
}

ResourceId SystemInfo::affinity_processor(MemoryId memory_id) {
    if (memory_id.is_device()) {
        return memory_id.as_device();
    } else {
        return ResourceId::host();
    }
}

bool SystemInfo::is_memory_accessible(MemoryId memory_id, ResourceId proc_id) const {
    if (!memory_id.is_host() && proc_id.is_device()) {
        return affinity_memory(proc_id.as_device()) == memory_id;
    }

    return memory_id.is_host();
}

bool SystemInfo::is_memory_accessible(MemoryId memory_id, DeviceId device_id) const {
    return is_memory_accessible(memory_id, ResourceId(device_id));
}

}  // namespace kmm
