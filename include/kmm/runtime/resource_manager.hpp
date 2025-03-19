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

class DeviceResourceManager {
    KMM_NOT_COPYABLE_OR_MOVABLE(DeviceResourceManager)

  public:
    DeviceResourceManager(
        std::vector<GPUContextHandle> contexts,
        std::shared_ptr<DeviceStreamManager> stream_manager
    );

    ~DeviceResourceManager();

    GPUContextHandle context(DeviceId device_id);

    DeviceEvent submit(
        DeviceId device_id,
        DeviceEventSet deps,
        DeviceResourceOperation& op,
        std::vector<BufferAccessor> accessors
    );

  private:
    struct Device;

    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    std::vector<std::unique_ptr<Device>> m_devices;
};

}  // namespace kmm
