#include <algorithm>

#include "spdlog/spdlog.h"

#include "kmm/runtime/task_graph.hpp"

namespace kmm {

std::vector<std::pair<BufferId, BufferLayout>> TaskGraph::flush_buffers() {
    std::lock_guard guard {m_mutex};
    auto temp = std::move(m_buffers);
    return temp;
}

std::vector<TaskNode> TaskGraph::flush_tasks() {
    std::lock_guard guard {m_mutex};
    auto temp = std::move(m_nodes);
    return temp;
}

TaskGraphStage::TaskGraphStage(TaskGraph* state) : m_guard(state->m_mutex), m_state(state) {
    m_next_event_id = state->m_next_event_id;
    m_events_since_last_barrier = {state->m_last_stage_id};
}

EventId TaskGraphStage::commit() {
    auto barrier_id = insert_barrier();

    m_state->m_last_stage_id = barrier_id;
    m_state->m_next_event_id = m_next_event_id;
    m_state->m_next_buffer_id = m_next_buffer_id;
    m_state->m_nodes.insert(
        m_state->m_nodes.end(),
        std::make_move_iterator(m_staged_nodes.begin()),
        std::make_move_iterator(m_staged_nodes.end())
    );

    m_staged_nodes.clear();
    return barrier_id;
}

EventId TaskGraphStage::join_events(EventList deps) {
    if (deps.size() == 0) {
        return EventId();
    }

    if (std::equal(deps.begin() + 1, deps.end(), deps.begin())) {
        return deps[0];
    }

    return insert_node(CommandEmpty {}, std::move(deps));
}

BufferId TaskGraphStage::create_buffer(BufferLayout layout) {
    auto buffer_id = m_next_buffer_id;
    m_next_buffer_id = BufferId(buffer_id + 1);
    m_staged_buffers.emplace_back(buffer_id, layout);
    return buffer_id;
}

EventId TaskGraphStage::delete_buffer(BufferId id, EventList deps) {
    return insert_node(CommandBufferDelete {id}, std::move(deps));
}

EventId TaskGraphStage::insert_barrier() {
    if (m_events_since_last_barrier.size() == 1) {
        return m_events_since_last_barrier[0];
    }

    EventList deps = std::move(m_events_since_last_barrier);
    return insert_node(CommandEmpty {}, std::move(deps));
}

EventId TaskGraphStage::insert_compute_task(
    ProcessorId process_id,
    std::unique_ptr<ComputeTask> task,
    std::vector<BufferRequirement> buffers,
    EventList deps
) {
    return insert_node(
        CommandExecute {
            .processor_id = process_id,
            .task = std::move(task),
            .buffers = std::move(buffers)},
        std::move(deps)
    );
}

EventId TaskGraphStage::insert_node(Command command, EventList deps) {
    auto id = EventId(m_next_event_id.get());
    m_next_event_id = EventId(id.get() + 1);

    m_events_since_last_barrier.push_back(id);
    m_staged_nodes.push_back(
        TaskNode {.id = id, .command = std::move(command), .dependencies = std::move(deps)}
    );

    return id;
}
}  // namespace kmm
