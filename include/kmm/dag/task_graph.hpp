#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "kmm/core/buffer.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/dag/commands.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {

struct TaskNode {
    EventId id;
    Command command;
    EventList dependencies;
};

class TaskGraph {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskGraph)

  public:
    TaskGraph();
    ~TaskGraph();

    BufferId create_buffer(DataLayout layout);

    EventId delete_buffer(BufferId id, EventList deps = {});

    const EventList& extract_buffer_dependencies(BufferId id);

    EventId join_events(EventList deps);

    EventId insert_copy(
        BufferId src_buffer,
        MemoryId src_memory,
        BufferId dst_buffer,
        MemoryId dst_memory,
        CopyDef spec,
        EventList deps = {}
    );

    EventId insert_prefetch(BufferId buffer_id, MemoryId memory_id, EventList deps = {});

    EventId insert_compute_task(
        ProcessorId processor_id,
        std::shared_ptr<ComputeTask> task,
        const std::vector<BufferRequirement>& buffers,
        EventList deps = {}
    );

    EventId insert_reduction(
        BufferId src_buffer,
        MemoryId src_memory,
        BufferId dst_buffer,
        MemoryId dst_memory,
        ReductionDef reduction,
        EventList deps
    );

    EventId insert_fill(MemoryId memory_id, BufferId buffer_id, FillDef fill, EventList deps = {});

    EventId insert_barrier();

    EventId shutdown();

    void rollback();

    EventId commit();

    std::vector<TaskNode> flush();

  private:
    EventId insert_node(Command command, EventList deps = {});

    struct BufferMeta {
        BufferMeta(EventId epoch_event) : last_write(epoch_event), accesses {epoch_event} {}

        MemoryId owner_id = MemoryId::host();
        EventId last_write;
        EventList accesses;
    };

    struct BufferMetaUpdate {
        BufferMeta* meta;
        EventList new_writes;
        EventList new_accesses;
    };

    BufferMetaUpdate& find_buffer_update(BufferId id);

    void pre_access_buffer(
        BufferId buffer_id,
        AccessMode mode,
        MemoryId memory_id,
        EventList& deps_out
    );

    void post_access_buffer(
        BufferId buffer_id,
        AccessMode mode,
        MemoryId memory_id,
        EventId new_event_id
    );

    uint64_t m_next_buffer_id = 1;
    uint64_t m_next_event_id = 1;
    EventList m_events_since_last_barrier;
    std::unordered_map<BufferId, std::unique_ptr<BufferMeta>> m_buffers;
    std::vector<TaskNode> m_events;

    std::vector<std::pair<BufferId, std::unique_ptr<BufferMeta>>> m_staged_new_buffers;
    std::unordered_map<BufferId, BufferMetaUpdate> m_staged_delta_buffers;
    std::vector<BufferId> m_staged_buffer_deletions;
};

}  // namespace kmm