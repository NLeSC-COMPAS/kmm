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

class TaskGraph;
class TaskGraphStage;

struct TaskNode {
    EventId id;
    Command command;
    EventList dependencies;
};

class TaskGraph {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskGraph)
  public:
    friend TaskGraphStage;

    TaskGraph() = default;
    std::pair<std::vector<std::pair<BufferId, BufferLayout>>, std::vector<TaskNode>> flush();

  private:
    std::mutex m_mutex;
    BufferId m_next_buffer_id = BufferId(1);
    EventId m_next_event_id = EventId(1);
    EventId m_last_stage_id = EventId(1);
    std::vector<TaskNode> m_nodes;
    std::vector<std::pair<BufferId, BufferLayout>> m_buffers;
};

class TaskGraphStage {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskGraphStage)

  public:
    TaskGraphStage(TaskGraph* state);
    EventId commit();

    BufferId create_buffer(BufferLayout layout);
    EventId delete_buffer(BufferId id, EventList deps = {});

    EventId insert_barrier();

    EventId insert_compute_task(
        ProcessorId process_id,
        std::unique_ptr<ComputeTask> task,
        std::vector<BufferRequirement> buffers,
        EventList deps = {}
    );

    EventId join_events(EventList deps);

    EventId insert_node(Command command, EventList deps = {});

  private:
    std::lock_guard<std::mutex> m_guard;
    TaskGraph* m_state;
    BufferId m_next_buffer_id;
    EventId m_next_event_id;
    EventList m_events_since_last_barrier;
    std::vector<TaskNode> m_staged_nodes;
    std::vector<std::pair<BufferId, BufferLayout>> m_staged_buffers;
};

}  // namespace kmm
