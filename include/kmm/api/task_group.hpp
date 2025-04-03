#pragma once

#include "kmm/core/domain.hpp"
#include "kmm/core/identifiers.hpp"

namespace kmm {

class Runtime;
class TaskGraph;

struct TaskGroupInit {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskGroupInit)

  public:
    Runtime& runtime;
    const Domain& domain;
};

struct TaskInstance {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskInstance)

  public:
    Runtime& runtime;
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
    Runtime& runtime;
    TaskGraph& graph;
    EventId event_id;
};

struct TaskGroupCommit {
    Runtime& runtime;
    TaskGraph& graph;
};

}  // namespace kmm