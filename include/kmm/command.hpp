#include <functional>
#include <future>
#include <variant>

#include "kmm/executor.hpp"
#include "kmm/types.hpp"

namespace kmm {

struct BufferRequirement {
    PhysicalBufferId buffer_id;
    DeviceId memory_id;
    bool is_write;
};

struct CommandExecute {
    std::optional<ObjectId> output_object_id;
    DeviceId device_id;
    std::shared_ptr<Task> task;
    std::vector<BufferRequirement> buffers;
};

struct CommandNoop {};

struct CommandBufferCreate {
    PhysicalBufferId id;
    BufferLayout description;
};

struct CommandBufferDelete {
    PhysicalBufferId id;
};

struct CommandObjectDelete {
    ObjectId id;
};

struct CommandPromise {
    mutable std::promise<void> promise;
};

using Command = std::variant<
    CommandNoop,
    CommandPromise,
    CommandExecute,
    CommandBufferCreate,
    CommandBufferDelete,
    CommandObjectDelete>;

struct CommandPacket {
    CommandPacket(OperationId id, Command command, std::vector<OperationId> dependencies = {}) :
        id(id),
        command(std::move(command)),
        dependencies(std::move(dependencies)) {}

    OperationId id;
    Command command;
    std::vector<OperationId> dependencies;
};
}  // namespace kmm