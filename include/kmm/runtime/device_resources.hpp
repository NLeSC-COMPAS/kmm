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

    size_t num_contexts() const;
    GPUContextHandle context(DeviceId device_id);

    DeviceEvent submit(
        DeviceId device_id,
        DeviceStreamSet stream_hint,
        DeviceEventSet deps,
        DeviceResourceOperation& op,
        std::vector<BufferAccessor> accessors
    );

  private:
    struct Device;
    struct Stream;

    Stream* select_stream_for_operation(
        DeviceId device_id,
        DeviceStreamSet stream_hint,
        const DeviceEventSet& deps
    );

    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    size_t m_streams_per_device;
    std::vector<std::unique_ptr<Device>> m_devices;
};

}  // namespace kmm
