#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "kmm/core/buffer.hpp"
#include "kmm/core/commands.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {

class TaskGraphState;

class TaskGraph {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskGraph)

  public:
    friend class TaskGraphState;

    struct Node {
        EventId id;
        Command command;
        EventList dependencies;
    };

    TaskGraph(TaskGraphState* state);

    BufferId create_buffer(BufferLayout layout);
    EventId delete_buffer(BufferId id, EventList deps = {});
    EventId insert_barrier();
    EventId insert_compute_task(
        ResourceId process_id,
        std::unique_ptr<ComputeTask> task,
        std::vector<BufferRequirement> buffers,
        EventList deps = {}
    );

    EventId join_events(EventList deps);
    EventId insert_node(Command command, EventList deps = {});

  private:
    TaskGraphState* m_state;
    EventList m_events_since_last_barrier;
    std::vector<Node> m_staged_nodes;
    std::vector<std::pair<BufferId, BufferLayout>> m_staged_buffers;
};

class TaskGraphState {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskGraphState)
  public:
    friend class TaskGraph;

    TaskGraphState() = default;
    EventId commit(
        TaskGraph& g,
        std::vector<TaskGraph::Node>& staged_nodes,
        std::vector<std::pair<BufferId, BufferLayout>>& staged_buffers
    );

  private:
    BufferId m_next_buffer_id = BufferId(1);
    EventId m_next_event_id = EventId(1);
    EventId m_last_barrier_id = EventId(1);
};

}  // namespace kmm
