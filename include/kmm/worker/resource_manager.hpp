#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/resource.hpp"
#include "kmm/core/system_info.hpp"
#include "kmm/utils/gpu.hpp"
#include "kmm/worker/stream_manager.hpp"

namespace kmm {

class DeviceResourceOperation {
  public:
    virtual ~DeviceResourceOperation() = default;
    virtual void submit(DeviceResource& resource, std::vector<BufferAccessor> accessors) = 0;
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
    struct State;

    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    std::vector<std::unique_ptr<State>> m_devices;
};

}  // namespace kmm