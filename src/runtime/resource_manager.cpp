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
    size_t streams_per_context,
    std::shared_ptr<DeviceStreamManager> stream_manager
) :
    m_stream_manager(stream_manager),
    m_streams_per_device(streams_per_context) {
    KMM_ASSERT(m_streams_per_device > 0);

    for (size_t i = 0; i < contexts.size(); i++) {
        for (size_t j = 0; j < m_streams_per_device; j++) {
            auto stream = stream_manager->create_stream(contexts[i]);

            m_streams.emplace_back(std::make_unique<Device>(
                DeviceId(i),
                stream,
                contexts[i],
                stream_manager->get(stream)
            ));
        }
    }
}

DeviceResourceManager::~DeviceResourceManager() {}

GPUContextHandle DeviceResourceManager::context(DeviceId device_id) {
    KMM_ASSERT(device_id * m_streams_per_device < m_streams.size());
    return m_streams[device_id * m_streams_per_device]->context;
}

size_t DeviceResourceManager::select_stream_for_operation(
    DeviceId device_id,
    const DeviceEventSet& deps
) {
    KMM_ASSERT(device_id * m_streams_per_device < m_streams.size());
    size_t offset = device_id * m_streams_per_device;

    // Push the first stream down to the bottom
    for (size_t i = 1; i < m_streams_per_device; i++) {
        std::swap(m_streams[offset + i - 1], m_streams[offset + i]);
    }

    // Find a stream that contains one of the dependencies
    for (size_t i = 0; i < m_streams_per_device; i++) {
        auto e = m_streams[offset + i]->last_event;

        if (std::find(deps.begin(), deps.end(), e) != deps.end()) {
            return offset + i;
        }
    }

    // Just return the first stream;
    return offset;
}

DeviceEvent DeviceResourceManager::submit(
    DeviceId device_id,
    DeviceEventSet deps,
    DeviceResourceOperation& op,
    std::vector<BufferAccessor> accessors
) {
    auto& state = *m_streams[select_stream_for_operation(device_id, deps)];

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
