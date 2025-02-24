#pragma once

#include "kmm/core/identifiers.hpp"
#include "kmm/dag/work_distribution.hpp"

namespace kmm {

class Worker;
class TaskGraph;

struct TaskGroupInit {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskGroupInit)

  public:
    Worker& worker;
    TaskGraph& graph;
    const WorkDistribution& partition;
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

struct TaskGroupFinalize {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskGroupFinalize)

  public:
    Worker& worker;
    TaskGraph& graph;
    EventList events;
};

}  // namespace kmm