#pragma once

#include "kmm/core/identifiers.hpp"
#include "kmm/dag/domain.hpp"

namespace kmm {

class Worker;
class TaskGraph;

struct TaskGroupInit {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskGroupInit)

  public:
    Worker& worker;
    const Domain& domain;
};

struct TaskInstance {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskInstance)

  public:
    Worker& worker;
    TaskGraph& graph;
    DomainChunk chunk;
    MemoryId memory_id;
    std::vector<BufferRequirement> buffers;
    EventList dependencies;

    size_t add_buffer_requirement(BufferRequirement req) {
        size_t index = buffers.size();
        buffers.push_back(std::move(req));
        return index;
    }
};

struct TaskSubmissionResult {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskSubmissionResult)

  public:
    Worker& worker;
    TaskGraph& graph;
    EventId event_id;
    EventList& dependencies;
};

}  // namespace kmm