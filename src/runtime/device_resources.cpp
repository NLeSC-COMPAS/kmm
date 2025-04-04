#include <algorithm>

#include "kmm/runtime/device_resources.hpp"

namespace kmm {

struct DeviceResources::Device {
    KMM_NOT_COPYABLE_OR_MOVABLE(Device)

  public:
    Device(GPUContextHandle context) : context(context) {}

    GPUContextHandle context;
    size_t last_selected_stream = 0;
};

struct DeviceResources::Stream {
    KMM_NOT_COPYABLE_OR_MOVABLE(Stream)

  public:
    Stream(
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

DeviceResources::DeviceResources(
    std::vector<GPUContextHandle> contexts,
    size_t streams_per_context,
    std::shared_ptr<DeviceStreamManager> stream_manager
) :
    m_stream_manager(stream_manager),
    m_streams_per_device(streams_per_context) {
    KMM_ASSERT(m_streams_per_device > 0);

    for (size_t i = 0; i < contexts.size(); i++) {
        m_devices.emplace_back(std::make_unique<Device>(contexts[i]));

        for (size_t j = 0; j < m_streams_per_device; j++) {
            auto stream = stream_manager->create_stream(contexts[i]);

            m_streams.emplace_back(std::make_unique<Stream>(
                DeviceId(i),
                stream,
                contexts[i],
                stream_manager->get(stream)
            ));
        }
    }
}

DeviceResources::~DeviceResources() {
    for (const auto& e : m_streams) {
        m_stream_manager->wait_until_ready(e->stream);
    }
}

GPUContextHandle DeviceResources::context(DeviceId device_id) {
    KMM_ASSERT(device_id < m_devices.size());
    return m_devices[device_id]->context;
}

size_t DeviceResources::select_stream_for_operation(
    DeviceId device_id,
    const DeviceEventSet& deps
) {
    KMM_ASSERT(device_id * m_streams_per_device < m_streams.size());
    size_t offset = device_id * m_streams_per_device;

    // Find a stream that contains one of the dependencies
    for (size_t i = 0; i < m_streams_per_device; i++) {
        auto e = m_streams[offset + i]->last_event;

        if (std::find(deps.begin(), deps.end(), e) != deps.end()) {
            return offset + i;
        }
    }

    // Go over the streams round-robin
    auto i = m_devices[device_id]->last_selected_stream++;
    return offset + (i % m_streams_per_device);
}

DeviceEvent DeviceResources::submit(
    DeviceId device_id,
    std::optional<uint64_t> stream_index,
    DeviceEventSet deps,
    DeviceResourceOperation& op,
    std::vector<BufferAccessor> accessors
) {
    if (!stream_index.has_value()) {
        stream_index = select_stream_for_operation(device_id, deps);
    }

    auto& state = *m_streams[(*stream_index) % m_streams_per_device];

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
