#include "kmm/memops/host_copy.hpp"
#include "kmm/planner/array_instance.hpp"
#include "kmm/planner/read_planner.hpp"
#include "kmm/planner/write_planner.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

template<size_t N>
ArrayDescriptor<N>::ArrayDescriptor(
    Distribution<N> distribution,
    DataType dtype,
    std::vector<BufferDescriptor> buffers
) :
    m_distribution(std::move(distribution)),
    m_dtype(dtype),
    m_buffers(std::move(buffers)) {}

template<size_t N>
EventId ArrayDescriptor<N>::join_events(TaskGraphStage& stage) const {
    EventList deps;

    for (const auto& buffer : m_buffers) {
        deps.insert_all(buffer.last_access_events);
    }

    return stage.join_events(deps);
}

template<size_t N>
void ArrayDescriptor<N>::destroy(TaskGraphStage& stage) {
    KMM_ASSERT(m_num_readers == 0 && m_num_writers == 0);

    for (BufferDescriptor& meta : m_buffers) {
        stage.delete_buffer(meta.id, std::move(meta.last_access_events));
    }

    m_distribution = Distribution<N>();
    m_buffers.clear();
}

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
ArrayInstance<N>::ArrayInstance(Runtime& rt, Distribution<N> dist, DataType dtype) :
    ArrayDescriptor<N>(dist, dtype, allocate_buffers(rt, dist, dtype)),
    m_rt(rt.shared_from_this()) {}

template<size_t N>
ArrayInstance<N>::~ArrayInstance() {
    m_rt->schedule([&](TaskGraphStage& stage) { this->destroy(stage); });
}

template<size_t N>
CopyDef build_copy_operation(
    Index<N> src_offset,
    Dim<N> src_dims,
    Index<N> dst_offset,
    Dim<N> dst_dims,
    Dim<N> counts,
    size_t element_size
) {
    auto copy_def = CopyDef(element_size);

    size_t src_stride = 1;
    size_t dst_stride = 1;

    for (size_t i = 0; i < N; i++) {
        copy_def.add_dimension(  //
            checked_cast<size_t>(counts[i]),
            checked_cast<size_t>(src_offset[i]),
            checked_cast<size_t>(dst_offset[i]),
            src_stride,
            dst_stride
        );

        src_stride *= checked_cast<size_t>(src_dims[i]);
        dst_stride *= checked_cast<size_t>(dst_dims[i]);
    }

    copy_def.simplify();
    return copy_def;
}

template<size_t N>
class CopyIntoTask: public ComputeTask {
  public:
    CopyIntoTask(void* dst_buffer, CopyDef copy_def) {
        m_dst_buffer = dst_buffer;
        m_copy = copy_def;
    }

    void execute(Resource& resource, TaskContext context) {
        KMM_ASSERT(context.accessors.size() == 1);
        const void* src_buffer = context.accessors[0].address;
        execute_copy(src_buffer, m_dst_buffer, m_copy);
    }

  private:
    void* m_dst_buffer;
    CopyDef m_copy;
};

template<size_t N>
class CopyFromTask: public ComputeTask {
  public:
    CopyFromTask(const void* src_buffer, CopyDef copy_def) {
        m_src_buffer = src_buffer;
        m_copy = copy_def;
    }

    void execute(Resource& resource, TaskContext context) {
        KMM_ASSERT(context.accessors.size() == 1);
        KMM_ASSERT(context.accessors[0].is_writable);
        void* dst_buffer = context.accessors[0].address;
        execute_copy(m_src_buffer, dst_buffer, m_copy);
    }

  private:
    const void* m_src_buffer;
    CopyDef m_copy;
};

template<size_t N>
void ArrayInstance<N>::copy_bytes_into(void* dst_data) {
    auto& dist = this->distribution();
    auto data_type = this->data_type();
    auto num_chunks = dist.num_chunks();

    auto event_id = m_rt->schedule([&](TaskGraphStage& stage) {
        auto planner = ArrayReadPlanner<N> {this->shared_from_this()};

        for (size_t i = 0; i < num_chunks; i++) {
            auto deps = EventList {};
            auto chunk = dist.chunk(i);
            auto chunk_region = Bounds<N>::from_offset_size(chunk.offset, chunk.size);

            auto buffer_req = planner.prepare_access(  //
                stage,
                MemoryId::host(),
                chunk_region,
                deps
            );

            auto copy_def = build_copy_operation(  //
                Index<N>::zero(),
                chunk_region.sizes(),
                chunk_region.begin(),
                dist.array_size(),
                chunk_region.sizes(),
                data_type.size_in_bytes()
            );

            auto task = std::make_unique<CopyIntoTask<N>>(dst_data, copy_def);

            auto event_id = stage.insert_compute_task(
                ProcessorId::host(),
                std::move(task),
                {buffer_req},
                std::move(deps)
            );

            planner.finalize_access(stage, event_id);
        }

        planner.commit(stage);
    });

    m_rt->query_event(event_id, std::chrono::system_clock::time_point::max());
}

template<size_t N>
void ArrayInstance<N>::copy_bytes_from(const void* src_data) {
    auto& dist = this->distribution();
    auto data_type = this->data_type();
    auto num_chunks = dist.num_chunks();

    auto event_id = m_rt->schedule([&](TaskGraphStage& stage) {
        auto planner = ArrayWritePlanner<N> {this->shared_from_this()};

        for (size_t i = 0; i < num_chunks; i++) {
            auto deps = EventList {};
            auto chunk = dist.chunk(i);
            auto chunk_region = Bounds<N>::from_offset_size(chunk.offset, chunk.size);

            auto buffer_req = planner.prepare_access(  //
                stage,
                MemoryId::host(),
                chunk_region,
                deps
            );

            auto copy_def = build_copy_operation(  //
                Index<N>::zero(),
                chunk_region.sizes(),
                chunk_region.begin(),
                dist.array_size(),
                chunk_region.sizes(),
                data_type.size_in_bytes()
            );

            auto task = std::make_unique<CopyFromTask<N>>(src_data, copy_def);

            auto event_id = stage.insert_compute_task(
                ProcessorId::host(),
                std::move(task),
                {buffer_req},
                std::move(deps)
            );

            planner.finalize_access(stage, event_id);
        }

        planner.commit(stage);
    });

    m_rt->query_event(event_id, std::chrono::system_clock::time_point::max());
}

template<size_t N>
void ArrayInstance<N>::synchronize() const {
    EventId event_id;
    m_rt->schedule([&](TaskGraphStage& stage) { event_id = this->join_events(stage); });

    m_rt->query_event(event_id, std::chrono::system_clock::time_point::max());
}

}  // namespace kmm
