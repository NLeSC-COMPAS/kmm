#include <algorithm>

#include "spdlog/spdlog.h"

#include "kmm/dag/task_graph.hpp"

namespace kmm {

TaskGraph::TaskGraph() = default;
TaskGraph::~TaskGraph() = default;

BufferId TaskGraph::create_buffer(DataLayout layout) {
    auto buffer_id = BufferId(m_next_buffer_id);
    m_next_buffer_id++;

    auto event_id = insert_node(CommandBufferCreate {
        .id = buffer_id,
        .layout = layout,
    });

    auto meta = std::make_unique<BufferMeta>(event_id);
    m_staged_delta_buffers.emplace(buffer_id, meta.get());
    m_staged_new_buffers.emplace_back(buffer_id, std::move(meta));

    return buffer_id;
}

EventId TaskGraph::delete_buffer(BufferId id, EventList deps) {
    // Erase the buffer
    m_staged_buffer_deletions.push_back(id);
}

const EventList& TaskGraph::extract_buffer_dependencies(BufferId id) {
    return find_buffer_update(id).meta->accesses;
}

EventId TaskGraph::join_events(EventList deps) {
    if (deps.size() == 0) {
        return EventId(0);
    }

    // Check if all dependencies are the same event ID
    if (std::equal(deps.begin() + 1, deps.end(), deps.begin())) {
        return deps[0];
    }

    return insert_node(CommandEmpty {}, std::move(deps));
}

EventId TaskGraph::insert_copy(
    BufferId src_buffer,
    MemoryId src_memory,
    BufferId dst_buffer,
    MemoryId dst_memory,
    CopyDef spec,
    EventList deps
) {
    pre_access_buffer(src_buffer, AccessMode::Read, src_memory, deps);
    pre_access_buffer(dst_buffer, AccessMode::ReadWrite, dst_memory, deps);

    auto event_id = insert_node(
        CommandCopy {
            src_buffer,  //
            src_memory,
            dst_buffer,
            dst_memory,
            spec},
        std::move(deps)
    );

    post_access_buffer(src_buffer, AccessMode::Read, src_memory, event_id);
    post_access_buffer(dst_buffer, AccessMode::ReadWrite, dst_memory, event_id);

    return event_id;
}

EventId TaskGraph::insert_prefetch(BufferId buffer_id, MemoryId memory_id, EventList deps) {
    pre_access_buffer(buffer_id, AccessMode::Read, memory_id, deps);

    auto event_id = insert_node(
        CommandPrefetch {
            .buffer_id = buffer_id,  //
            .memory_id = memory_id},
        std::move(deps)
    );

    post_access_buffer(buffer_id, AccessMode::Read, memory_id, event_id);
    return event_id;
}

EventId TaskGraph::insert_compute_task(
    ProcessorId processor_id,
    std::shared_ptr<ComputeTask> task,
    const std::vector<BufferRequirement>& buffers,
    EventList deps
) {
    for (const auto& buffer : buffers) {
        pre_access_buffer(buffer.buffer_id, buffer.access_mode, buffer.memory_id, deps);
    }

    auto event_id = insert_node(
        CommandExecute {.processor_id = processor_id, .task = std::move(task), .buffers = buffers},
        std::move(deps)
    );

    for (const auto& buffer : buffers) {
        post_access_buffer(buffer.buffer_id, buffer.access_mode, buffer.memory_id, event_id);
    }

    return event_id;
}

EventId TaskGraph::insert_reduction(
    BufferId src_buffer,
    MemoryId src_memory,
    BufferId dst_buffer,
    MemoryId dst_memory,
    ReductionDef reduction,
    EventList deps
) {
    pre_access_buffer(src_buffer, AccessMode::Read, src_memory, deps);
    pre_access_buffer(dst_buffer, AccessMode::ReadWrite, dst_memory, deps);
    EventId event_id;

    if (reduction.num_inputs_per_output == 1) {
        auto dtype = reduction.data_type;
        auto src_offset = reduction.input_offset_elements;
        auto dst_offset = reduction.output_offset_elements;
        size_t num_elements = reduction.num_outputs;

        auto copy = CopyDef(dtype.size_in_bytes());
        copy.add_dimension(num_elements, src_offset, dst_offset);

        event_id = insert_node(
            CommandCopy {
                .src_buffer = src_buffer,
                .src_memory = src_memory,
                .dst_buffer = dst_buffer,
                .dst_memory = dst_memory,
                .definition = copy},
            std::move(deps)
        );
    } else {
        event_id = insert_node(
            CommandReduction {
                .src_buffer = src_buffer,
                .dst_buffer = dst_buffer,
                .memory_id = dst_memory,
                .definition = reduction},
            std::move(deps)
        );
    }

    post_access_buffer(src_buffer, AccessMode::Read, src_memory, event_id);
    post_access_buffer(dst_buffer, AccessMode::ReadWrite, dst_memory, event_id);

    return event_id;
}

EventId TaskGraph::insert_fill(
    MemoryId memory_id,
    BufferId buffer_id,
    FillDef fill,
    EventList deps
) {
    pre_access_buffer(buffer_id, AccessMode::ReadWrite, memory_id, deps);

    auto event_id = insert_node(
        CommandFill {
            .dst_buffer = buffer_id,
            .memory_id = memory_id,
            .definition = std::move(fill)},
        std::move(deps)
    );

    post_access_buffer(buffer_id, AccessMode::ReadWrite, memory_id, event_id);
    return event_id;
}

EventId TaskGraph::insert_barrier() {
    EventList deps = std::move(m_events_since_last_barrier);
    return join_events(std::move(deps));
}

EventId TaskGraph::shutdown() {
    rollback();

    for (auto& [id, buffer] : m_buffers) {
        insert_node(CommandBufferDelete {id}, buffer->accesses);
    }

    return insert_barrier();
}

void TaskGraph::rollback() {
    m_events.clear();
    m_staged_new_buffers.clear();
    m_staged_delta_buffers.clear();
    m_staged_buffer_deletions.clear();
}

EventId TaskGraph::commit() {
    for (auto& p : m_staged_new_buffers) {
        m_buffers.emplace(std::move(p));
    }

    for (auto& [id, delta] : m_staged_delta_buffers) {
        auto& meta = delta.meta;

        meta->accesses.insert_all(std::move(delta.new_accesses));

        if (!delta.new_writes.is_empty()) {
            meta->last_write = join_events(std::move(delta.new_writes));
        }
    }

    for (auto& id : m_staged_buffer_deletions) {
        auto it = m_buffers.find(id);
        KMM_ASSERT(it != m_buffers.end());

        auto meta = std::move(it->second);
        m_buffers.erase(it);

        insert_node(CommandBufferDelete {id}, std::move(meta->accesses));
    }

    m_staged_new_buffers.clear();
    m_staged_delta_buffers.clear();
    m_staged_buffer_deletions.clear();

    return insert_barrier();
}

std::vector<TaskNode> TaskGraph::flush() {
    rollback();
    return std::move(m_events);
}

EventId TaskGraph::insert_node(Command command, EventList deps) {
    auto event_id = EventId(m_next_event_id);
    m_next_event_id++;

    if (deps.size() > 1) {
        std::sort(deps.begin(), deps.end());
        auto* mid = std::unique(deps.begin(), deps.end());
        deps.truncate(mid - deps.begin());
    }

    m_events.push_back({event_id, std::move(command), std::move(deps)});
    m_events_since_last_barrier.push_back(event_id);

    return event_id;
}

TaskGraph::BufferMetaUpdate& TaskGraph::find_buffer_update(BufferId id) {
    if (auto it = m_staged_delta_buffers.find(id); KMM_LIKELY(it != m_staged_delta_buffers.end())) {
        return it->second;
    }

    if (auto it = m_buffers.find(id); it != m_buffers.end()) {
        auto result =
            m_staged_delta_buffers.emplace(id, BufferMetaUpdate {.meta = it->second.get()});

        return result.first->second;
    }

    throw std::runtime_error(fmt::format("buffer with identifier {} was not found", id));
}

void TaskGraph::pre_access_buffer(
    BufferId buffer_id,
    AccessMode mode,
    MemoryId memory_id,
    EventList& deps_out
) {
    auto& delta = find_buffer_update(buffer_id);
    deps_out.push_back(delta.meta->last_write);

    if (mode != AccessMode::Read) {
        deps_out.insert_all(delta.meta->accesses);
    }
}

void TaskGraph::post_access_buffer(
    BufferId buffer_id,
    AccessMode mode,
    MemoryId memory_id,
    EventId new_event_id
) {
    auto& delta = find_buffer_update(buffer_id);
    delta.new_accesses.push_back(new_event_id);

    if (mode != AccessMode::Read) {
        delta.new_writes.push_back(new_event_id);
    }
}

}  // namespace kmm