#pragma once

#include <variant>

#include "fmt/ostream.h"

#include "kmm/core/buffer.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/core/resource.hpp"
#include "kmm/memops/types.hpp"

namespace kmm {

struct CommandBufferDelete {
    BufferId id;
};

struct CommandPrefetch {
    BufferId buffer_id;
    MemoryId memory_id;
};

struct CommandCopy {
    BufferId src_buffer;
    MemoryId src_memory;
    BufferId dst_buffer;
    MemoryId dst_memory;
    CopyDef definition;
};

struct CommandExecute {
    ResourceId processor_id;
    std::unique_ptr<ComputeTask> task;
    std::vector<BufferRequirement> buffers;
};

struct CommandReduction {
    BufferId src_buffer;
    BufferId dst_buffer;
    MemoryId memory_id;
    ReductionDef definition;
};

struct CommandFill {
    BufferId dst_buffer;
    MemoryId memory_id;
    FillDef definition;
};

struct CommandEmpty {};

using Command = std::variant<
    CommandEmpty,
    CommandBufferDelete,
    CommandPrefetch,
    CommandCopy,
    CommandExecute,
    CommandReduction,
    CommandFill>;

inline const char* command_name(const Command& cmd) {
    static constexpr const char* names[] = {
        "CommandEmpty",
        "CommandBufferDelete",
        "CommandPrefetch",
        "CommandCopy",
        "CommandExecute",
        "CommandReduction",
        "CommandFill"};

    return names[cmd.index()];
}

inline std::ostream& operator<<(std::ostream& f, const Command& cmd) {
    return f << command_name(cmd);
}

}  // namespace kmm

template<>
struct fmt::formatter<kmm::Command>: fmt::ostream_formatter {};
