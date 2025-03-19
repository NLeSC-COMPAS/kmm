#include "kmm/memops/host_copy.hpp"
#include "kmm/planner/array_descriptor.hpp"
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
EventId ArrayDescriptor<N>::copy_bytes_into_buffer(TaskGraphStage& stage, void* dst_data) {
    auto& dist = this->distribution();
    auto data_type = this->data_type();
    auto num_chunks = dist.num_chunks();
    auto new_read_events = EventList {};

    for (size_t i = 0; i < num_chunks; i++) {
        auto& buffer = m_buffers[i];

        auto buffer_req = BufferRequirement {
            .buffer_id = buffer.id,
            .memory_id = MemoryId::host(),
            .access_mode = AccessMode::Read};

        auto chunk = dist.chunk(i);
        auto chunk_region = Bounds<N>::from_offset_size(chunk.offset, chunk.size);

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
            {buffer.last_write_event}
        );

        new_read_events.push_back(event_id);
    }

    for (size_t i = 0; i < num_chunks; i++) {
        m_buffers[i].last_access_events.push_back(new_read_events[i]);
    }

    return stage.join_events(new_read_events);
}

template<size_t N>
EventId ArrayDescriptor<N>::copy_bytes_from_buffer(TaskGraphStage& stage, const void* src_data) {
    auto& dist = this->distribution();
    auto data_type = this->data_type();
    auto num_chunks = dist.num_chunks();
    auto new_write_events = EventList {};

    for (size_t i = 0; i < num_chunks; i++) {
        auto& buffer = m_buffers[i];

        auto buffer_req = BufferRequirement {
            .buffer_id = buffer.id,  //
            .memory_id = MemoryId::host(),
            .access_mode = AccessMode::ReadWrite};

        auto chunk = dist.chunk(i);
        auto chunk_region = Bounds<N>::from_offset_size(chunk.offset, chunk.size);

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
            buffer.last_access_events
        );

        new_write_events.push_back(event_id);
    }

    for (size_t i = 0; i < num_chunks; i++) {
        auto& buffer = m_buffers[i];
        buffer.last_write_event = new_write_events[i];
        buffer.last_access_events = {new_write_events[i]};
    }

    return stage.join_events(new_write_events);
}

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

}  // namespace kmm
