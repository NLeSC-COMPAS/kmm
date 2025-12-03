#include <algorithm>

#include "spdlog/spdlog.h"

#include "kmm/runtime/task_graph.hpp"

namespace kmm {

EventId TaskGraphState::commit(
    TaskGraph& g,
    std::vector<TaskGraph::Node>& staged_nodes,
    std::vector<std::pair<BufferId, BufferLayout>>& staged_buffers
) {
    KMM_ASSERT(g.m_state == this);
    m_last_barrier_id = g.insert_barrier();
    staged_nodes = std::move(g.m_staged_nodes);
    staged_buffers = std::move(g.m_staged_buffers);
    return m_last_barrier_id;
}

TaskGraph::TaskGraph(TaskGraphState* state) : m_state(state) {}

EventId TaskGraph::join_events(EventList deps) {
    if (deps.size() == 0) {
        return EventId();
    }

    if (std::equal(deps.begin() + 1, deps.end(), deps.begin())) {
        return deps[0];
    }

    return insert_node(CommandEmpty {}, std::move(deps));
}

BufferId TaskGraph::create_buffer(BufferLayout layout) {
    auto buffer_id = m_state->m_next_buffer_id;
    m_state->m_next_buffer_id = BufferId(buffer_id + 1);
    m_staged_buffers.emplace_back(buffer_id, layout);
    return buffer_id;
}

EventId TaskGraph::delete_buffer(BufferId id, EventList deps) {
    return insert_node(CommandBufferDelete {id}, std::move(deps));
}

EventId TaskGraph::insert_barrier() {
    if (m_events_since_last_barrier.is_empty()) {
        return m_state->m_last_barrier_id;
    }

    EventList deps = std::move(m_events_since_last_barrier);
    deps.push_back(m_state->m_last_barrier_id);

    return join_events(std::move(deps));
}

EventId TaskGraph::insert_compute_task(
    ResourceId process_id,
    std::unique_ptr<ComputeTask> task,
    std::vector<BufferRequirement> buffers,
    EventList deps
) {
    return insert_node(
        CommandExecute {
            .processor_id = process_id,
            .task = std::move(task),
            .buffers = std::move(buffers)
        },
        std::move(deps)
    );
}

EventId TaskGraph::insert_node(Command command, EventList deps) {
    auto id = EventId(m_state->m_next_event_id.get());
    m_state->m_next_event_id = EventId(id.get() + 1);

    m_events_since_last_barrier.push_back(id);
    m_staged_nodes.push_back(
        Node {.id = id, .command = std::move(command), .dependencies = std::move(deps)}
    );

    return id;
}
}  // namespace kmm
