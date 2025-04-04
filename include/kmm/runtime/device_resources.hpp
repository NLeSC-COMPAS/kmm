#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/resource.hpp"
#include "kmm/core/system_info.hpp"
#include "kmm/runtime/stream_manager.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

class DeviceResourceOperation {
  public:
    virtual ~DeviceResourceOperation() = default;
    virtual void execute(DeviceResource& resource, std::vector<BufferAccessor> accessors) = 0;
};

class DeviceResources {
    KMM_NOT_COPYABLE_OR_MOVABLE(DeviceResources)

  public:
    DeviceResources(
        std::vector<GPUContextHandle> contexts,
        size_t streams_per_context,
        std::shared_ptr<DeviceStreamManager> stream_manager
    );

    ~DeviceResources();

    GPUContextHandle context(DeviceId device_id);

    DeviceEvent submit(
        DeviceId device_id,
        std::optional<uint64_t> stream_index,
        DeviceEventSet deps,
        DeviceResourceOperation& op,
        std::vector<BufferAccessor> accessors
    );

  private:
    size_t select_stream_for_operation(DeviceId device_id, const DeviceEventSet& deps);

    struct Device;
    struct Stream;

    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    size_t m_streams_per_device;
    std::vector<std::unique_ptr<Device>> m_devices;
    std::vector<std::unique_ptr<Stream>> m_streams;
};

}  // namespace kmm
