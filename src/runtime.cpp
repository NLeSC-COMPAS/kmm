#include <utility>

#include "spdlog/spdlog.h"

#include "kmm/runtime.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {

Runtime::Runtime(
    std::vector<std::shared_ptr<DeviceHandle>> devices,
    std::unique_ptr<Memory> memory) :
    m_worker(std::make_shared<Worker>(devices, std::move(memory))),
    m_thread(m_worker) {
    for (const auto& device : devices) {
        m_devices.push_back(device->info());
    }
}

Runtime::~Runtime() {
    m_worker->shutdown();
    m_thread.join();
}

BlockId Runtime::create_block(
    MemoryId memory_id,
    std::unique_ptr<BlockHeader> header,
    const void* src_data,
    size_t num_bytes) const {
    auto event_id = EventId(m_next_event++);
    auto block_id = BlockId(event_id, 0);
    m_worker->create_block(block_id, memory_id, std::move(header), src_data, num_bytes);

    m_block_accesses.insert({block_id, {event_id}});
    return block_id;
}

std::shared_ptr<BlockHeader> Runtime::read_block_header(BlockId block_id) const {
    query_event(block_id.event(), std::chrono::system_clock::time_point::max());
    return m_worker->read_block_header(block_id);
}

std::shared_ptr<BlockHeader> Runtime::read_block(BlockId block_id, void* dst_data, size_t num_bytes)
    const {
    query_event(block_id.event(), std::chrono::system_clock::time_point::max());
    return m_worker->read_block(block_id, std::nullopt, dst_data, num_bytes);
}

EventId Runtime::submit_task(std::shared_ptr<Task> task, TaskRequirements reqs) const {
    std::lock_guard guard {m_mutex};

    spdlog::info("submit task");
    auto event_id = EventId(m_next_event++);
    auto num_outputs = reqs.outputs.size();

    for (auto& input : reqs.inputs) {
        auto& accesses = m_block_accesses.at(input.block_id);
        reqs.dependencies.push_back(input.block_id.event());
        accesses.push_back(event_id);
    }

    for (size_t output_index = 0; output_index < num_outputs; output_index++) {
        auto block_id = BlockId(event_id, static_cast<uint8_t>(output_index));
        m_block_accesses.insert({block_id, {event_id}});
    }

    m_worker->submit_command(
        event_id,
        ExecuteCommand {
            .device_id = reqs.device_id,
            .task = std::move(task),
            .inputs = std::move(reqs.inputs),
            .outputs = std::move(reqs.outputs),
        },
        std::move(reqs.dependencies));

    return event_id;
}

EventId Runtime::delete_block(BlockId block_id, EventList deps) const {
    std::lock_guard guard {m_mutex};

    auto& accesses = m_block_accesses[block_id];
    deps.extend(accesses);

    auto id = EventId(m_next_event++);
    m_worker->submit_command(id, BlockDeleteCommand {block_id}, std::move(deps));

    m_block_accesses.erase(block_id);
    return id;
}

EventId Runtime::join_events(EventList deps) const {
    std::lock_guard guard {m_mutex};

    auto id = EventId(m_next_event++);
    m_worker->submit_command(id, EmptyCommand {}, std::move(deps));
    return id;
}

EventId Runtime::submit_barrier() const {
    std::lock_guard guard {m_mutex};
    auto id = EventId(m_next_event++);
    m_worker->submit_barrier(id);
    return id;
}

EventId Runtime::submit_block_barrier(BlockId block_id) const {
    std::lock_guard guard {m_mutex};
    auto id = EventId(m_next_event++);
    EventList deps;

    auto it = m_block_accesses.find(block_id);
    if (it != m_block_accesses.end()) {
        auto& accesses = it->second;
        deps.extend(accesses);
        accesses = {id};
    }

    m_worker->submit_command(id, EmptyCommand {}, std::move(deps));
    return id;
}

EventId Runtime::submit_block_prefetch(BlockId block_id, MemoryId memory_id, EventList deps) const {
    std::lock_guard guard {m_mutex};
    auto id = EventId(m_next_event++);

    // Add the event that created the block as a dependency
    deps.push_back(block_id.event());

    // Try to find the block. If we cannot find it, it was probably deleted and
    // thus prefetching has no effect. We can just create an event that joins all dependencies.
    auto it = m_block_accesses.find(block_id);
    if (it == m_block_accesses.end()) {
        return join_events(std::move(deps));
    }

    // Add prefetch event as access
    auto& accesses = it->second;
    accesses.push_back(id);

    // Submit the command!
    m_worker->submit_command(id, BlockPrefetchCommand {memory_id, block_id}, std::move(deps));
    return id;
}

bool Runtime::query_event(EventId id, std::chrono::time_point<std::chrono::system_clock> deadline)
    const {
    // No need to get the lock, the worker has its own internal lock
    /* std::lock_guard guard {m_mutex}; */

    return m_worker->query_event(id, deadline);
}

size_t Runtime::num_devices() const {
    return m_devices.size();
}

const DeviceInfo& Runtime::device_info(DeviceId id) const {
    return *m_devices.at(id);
}

}  // namespace kmm