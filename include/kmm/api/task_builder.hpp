#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/identifiers.hpp"

namespace kmm {

class Worker;
class TaskGraph;

struct TaskGroupInfo {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskGroupInfo)

  public:
    Worker& worker;
    TaskGraph& graph;
    const WorkPartition& partition;
};

struct TaskInstance {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskInstance)

  public:
    Worker& worker;
    TaskGraph& graph;
    WorkChunk chunk;
    MemoryId memory_id;
    std::vector<BufferRequirement> buffers;
    EventList dependencies;

    size_t add_buffer_requirement(BufferRequirement req) {
        size_t index = buffers.size();
        buffers.push_back(std::move(req));
        return index;
    }
};

struct TaskGroupResult {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskGroupResult)

  public:
    Worker& worker;
    TaskGraph& graph;
    EventList events;
};

}  // namespace kmm