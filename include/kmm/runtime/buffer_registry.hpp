#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "kmm/runtime/memory_manager.hpp"

namespace kmm {

class PoisonException;
using BufferRequest = std::shared_ptr<MemoryManager::Request>;
using BufferRequestList = std::vector<BufferRequest>;

class BufferRegistry {
  public:
    BufferRegistry(std::shared_ptr<MemoryManager> memory_manager);

    BufferId add(BufferId id, BufferLayout layout);

    std::shared_ptr<MemoryManager::Buffer> get(BufferId id);

    void remove(BufferId buffer_id);

    BufferRequestList create_requests(const std::vector<BufferRequirement>& buffers);

    Poll poll_requests(const BufferRequestList& requests, DeviceEventSet& dependencies_out);

    std::vector<BufferAccessor> access_requests(const BufferRequestList& requests);

    void release_requests(BufferRequestList& requests, DeviceEvent event = {});

    void poison(BufferId id, PoisonException reason);

    void poison_all(const std::vector<BufferRequirement>& buffers, PoisonException reason);

  private:
    struct BufferMeta {
        std::shared_ptr<MemoryManager::Buffer> buffer;
        std::unique_ptr<PoisonException> poison_reason_opt = nullptr;
    };

    std::shared_ptr<MemoryManager> m_memory_manager;
    std::unordered_map<BufferId, BufferMeta> m_buffers;
};

class PoisonException final: public std::exception {
  public:
    PoisonException(const std::string& error);
    const char* what() const noexcept override;

  private:
    std::string m_message;
};

}  // namespace kmm
