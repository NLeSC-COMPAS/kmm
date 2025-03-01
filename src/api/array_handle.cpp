#include "spdlog/spdlog.h"

#include "kmm/api/array.hpp"
#include "kmm/memops/host_copy.hpp"
#include "kmm/utils/integer_fun.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {

template<size_t N>
ArrayHandle<N>::~ArrayHandle() {
    m_worker->with_task_graph([&](auto& builder) {
        for (const auto& buffer_id : this->m_buffers) {
            builder.delete_buffer(buffer_id);
        }
    });
}

template<size_t N>
BufferId ArrayHandle<N>::buffer(size_t index) const {
    return m_buffers.at(index);
}

template<size_t N>
void ArrayHandle<N>::synchronize() const {
    auto event_id = m_worker->with_task_graph([&](TaskGraph& graph) {
        auto deps = EventList {};

        for (const auto& buffer_id : m_buffers) {
            deps.insert_all(graph.extract_buffer_dependencies(buffer_id));
        }

        return graph.join_events(deps);
    });

    m_worker->query_event(event_id, std::chrono::system_clock::time_point::max());

    // Access each buffer once to check for errors.
    for (size_t i = 0; i < m_buffers.size(); i++) {
        //        auto memory_id = this->chunk(i).owner_id;
        //        m_worker->access_buffer(m_buffers[i], memory_id, AccessMode::Read);
        //        KMM_TODO();
    }
}

template<size_t N>
class CopyOutTask: public ComputeTask {
  public:
    CopyOutTask(void* data, size_t element_size, Dim<N> array_size, Bounds<N> region) :
        m_dst_addr(data) {
        size_t src_stride = 1;
        size_t dst_stride = 1;

        m_copy = CopyDef(element_size);

        for (size_t j = 0; compare_less(j, N); j++) {
            size_t i = N - j - 1;

            m_copy.add_dimension(
                checked_cast<size_t>(region.size(i)),
                checked_cast<size_t>(0),
                checked_cast<size_t>(region.begin(i)),
                src_stride,
                dst_stride
            );

            src_stride *= checked_cast<size_t>(region.size(i));
            dst_stride *= checked_cast<size_t>(array_size.get_or_default(i));
        }
    }

    void execute(Resource& proc, TaskContext context) override {
        KMM_ASSERT(context.accessors.size() == 1);
        const void* src_addr = context.accessors[0].address;
        execute_copy(src_addr, m_dst_addr, m_copy);
    }

  private:
    void* m_dst_addr;
    CopyDef m_copy;
};

template<size_t N>
void ArrayHandle<N>::copy_bytes(void* dest_addr, size_t element_size) const {
    auto dest_mem = MemoryId::host();

    if (auto ordinal = get_gpu_device_by_address(dest_addr)) {
        dest_mem = m_worker->system_info().device_by_ordinal(*ordinal).memory_id();
    }

    auto event_id = m_worker->with_task_graph([&](TaskGraph& graph) {
        EventList deps;

        for (size_t i = 0; i < m_buffers.size(); i++) {
            auto chunk = m_distribution.chunk(i);
            auto region = Bounds<N>::from_offset_size(chunk.offset, chunk.size);

            auto task = std::make_shared<CopyOutTask<N>>(
                dest_addr,
                element_size,
                m_distribution.array_size(),
                region
            );

            auto buffer_id = m_buffers[i];
            auto buffer = BufferRequirement {
                .buffer_id = buffer_id,
                .memory_id = MemoryId::host(),
                .access_mode = AccessMode::Read,
            };

            auto event_id =
                graph.insert_compute_task(ProcessorId::host(), std::move(task), {buffer});
            deps.push_back(event_id);
        }

        return graph.join_events(std::move(deps));
    });

    m_worker->query_event(event_id, std::chrono::system_clock::time_point::max());
}

template<size_t N>
ArrayHandle<N>::ArrayHandle(
    Worker& worker,
    std::pair<DataDistribution<N>, std::vector<BufferId>> distribution
) :
    m_distribution(std::move(distribution.first)),
    m_worker(worker.shared_from_this()),
    m_buffers(std::move(distribution.second)) {}

KMM_INSTANTIATE_ARRAY_IMPL(ArrayHandle)

}  // namespace kmm