#include "kmm/runtime/resource_manager.hpp"

namespace kmm {

struct DeviceResourceManager::Device {
    KMM_NOT_COPYABLE_OR_MOVABLE(Device)

  public:
    Device(
        DeviceId device_id,
        DeviceStream stream,
        GPUContextHandle context,
        GPUstream_t gpu_stream
    ) :
        context(context),
        resource(DeviceInfo(device_id, context), context, gpu_stream),
        stream(stream) {}

    GPUContextHandle context;
    DeviceResource resource;
    DeviceStream stream;
    DeviceEvent last_event;
};

DeviceResourceManager::DeviceResourceManager(
    std::vector<GPUContextHandle> contexts,
    std::shared_ptr<DeviceStreamManager> stream_manager
) :
    m_stream_manager(stream_manager) {
    for (size_t i = 0; i < contexts.size(); i++) {
        auto stream = stream_manager->create_stream(contexts[i]);

        m_devices.emplace_back(
            std::make_unique<Device>(DeviceId(i), stream, contexts[i], stream_manager->get(stream))
        );
    }
}

DeviceResourceManager::~DeviceResourceManager() {}

GPUContextHandle DeviceResourceManager::context(DeviceId device_id) {
    KMM_ASSERT(device_id < m_devices.size());
    return m_devices[device_id]->context;
}

DeviceEvent DeviceResourceManager::submit(
    DeviceId device_id,
    DeviceEventSet deps,
    DeviceResourceOperation& op,
    std::vector<BufferAccessor> accessors
) {
    KMM_ASSERT(device_id < m_devices.size());
    auto& state = *m_devices.at(device_id);

    try {
        GPUContextGuard guard {state.context};
        m_stream_manager->wait_for_events(state.stream, deps);

        op.execute(state.resource, std::move(accessors));

        m_stream_manager->wait_on_default_stream(state.stream);
        auto event = m_stream_manager->record_event(state.stream);

        state.last_event = event;
        return event;
    } catch (const std::exception& e) {
        try {
            m_stream_manager->wait_until_ready(state.stream);
        } catch (...) {
            KMM_PANIC_FMT("fatal error: {}", e.what());
        }

        throw;
    }
}

}  // namespace kmm
