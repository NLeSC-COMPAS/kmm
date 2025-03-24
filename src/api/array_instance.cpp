#include "kmm/api/array_instance.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

template<size_t N>
static std::vector<BufferDescriptor> allocate_buffers(
    Runtime& rt,
    const Distribution<N>& dist,
    DataType dtype
) {
    size_t num_chunks = dist.num_chunks();
    std::vector<BufferDescriptor> buffers(num_chunks);

    for (size_t i = 0; i < num_chunks; i++) {
        auto chunk = dist.chunk(i);
        auto num_elements = chunk.size.volume();
        auto layout = BufferLayout::for_type(dtype, num_elements);
        auto buffer_id = rt.create_buffer(layout);

        buffers[i] = BufferDescriptor {.id = buffer_id, .layout = layout};
    }

    return buffers;
}

template<size_t N>
ArrayInstance<N>::ArrayInstance(
    TaskGraphStage& stage,
    Runtime& rt,
    Distribution<N> dist,
    DataType dtype
) :
    ArrayDescriptor<N>(stage, dist, dtype),
    m_rt(rt.shared_from_this()) {}

template<size_t N>
std::shared_ptr<ArrayInstance<N>> ArrayInstance<N>::create(
    Runtime& rt,
    Distribution<N> dist,
    DataType dtype
) {
    std::shared_ptr<ArrayInstance<N>> result;
    rt.schedule([&](auto& stage) {
        result = std::shared_ptr<ArrayInstance<N>>(
            new ArrayInstance<N>(stage, rt, std::move(dist), dtype)
        );
    });

    return result;
}

template<size_t N>
ArrayInstance<N>::~ArrayInstance() {
    m_rt->schedule([&](TaskGraphStage& stage) { this->destroy(stage); });
}

template<size_t N>
void ArrayInstance<N>::copy_bytes_into(void* dst_data) {
    auto event_id = m_rt->schedule([&](TaskGraphStage& stage) {
        this->copy_bytes_into_buffer(stage, dst_data);
    });

    m_rt->query_event(event_id, std::chrono::system_clock::time_point::max());
}

template<size_t N>
void ArrayInstance<N>::copy_bytes_from(const void* src_data) {
    auto event_id = m_rt->schedule([&](TaskGraphStage& stage) {
        this->copy_bytes_from_buffer(stage, src_data);
    });

    m_rt->query_event(event_id, std::chrono::system_clock::time_point::max());
}

template<size_t N>
void ArrayInstance<N>::synchronize() const {
    EventId event_id;
    m_rt->schedule([&](TaskGraphStage& stage) { event_id = this->join_events(stage); });

    m_rt->query_event(event_id, std::chrono::system_clock::time_point::max());
}

[[noreturn]] void throw_uninitialized_array_exception() {
    throw std::runtime_error(
        "attempted to access an uninitialized array, "
        "no associated instance was found"
    );
}

}  // namespace kmm