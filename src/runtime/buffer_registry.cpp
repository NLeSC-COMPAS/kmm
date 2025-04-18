#include "fmt/format.h"
#include "spdlog/spdlog.h"

#include "kmm/runtime/buffer_registry.hpp"

namespace kmm {

BufferRegistry::BufferRegistry(std::shared_ptr<MemoryManager> memory_manager) :
    m_memory_manager(memory_manager) {
    KMM_ASSERT(m_memory_manager);
}

BufferId BufferRegistry::add(BufferId buffer_id, BufferLayout layout) {
    auto [it, success] = m_buffers.emplace(buffer_id, BufferMeta {});

    if (!success) {
        throw std::runtime_error(
            fmt::format("could not add buffer {}: buffer already exists", buffer_id)
        );
    }

    auto buffer = m_memory_manager->create_buffer(layout, std::to_string(buffer_id));
    it->second.buffer = buffer;

    return buffer_id;
}

void BufferRegistry::remove(BufferId buffer_id) {
    auto it = m_buffers.find(buffer_id);

    if (it == m_buffers.end()) {
        throw std::runtime_error(
            fmt::format("could not remove buffer {}: buffer not found", buffer_id)
        );
    }

    m_memory_manager->delete_buffer(it->second.buffer);
    m_buffers.erase(it);
}

std::shared_ptr<MemoryManager::Buffer> BufferRegistry::get(BufferId id) {
    auto it = m_buffers.find(id);

    // Buffer not found, ignore
    if (it == m_buffers.end()) {
        throw std::runtime_error(fmt::format("could not retrieve buffer {}: buffer not found", id));
    }

    auto& meta = it->second;

    // If poisoned, throw exception
    if (meta.poison_reason_opt != nullptr) {
        throw PoisonException(*meta.poison_reason_opt);
    }

    return meta.buffer;
}

void BufferRegistry::poison(BufferId id, PoisonException reason) {
    auto it = m_buffers.find(id);

    // Buffer not found, ignore
    if (it == m_buffers.end()) {
        return;
    }

    auto& meta = it->second;

    // Buffer already poisoned, ignore
    if (meta.poison_reason_opt != nullptr) {
        return;
    }

    spdlog::warn("buffer {} was poisoned: {}", id, reason.what());
    meta.poison_reason_opt = std::make_unique<PoisonException>(std::move(reason));
}

BufferRequestList BufferRegistry::create_requests(const std::vector<BufferRequirement>& buffers) {
    auto parent = m_memory_manager->create_transaction();
    auto requests = BufferRequestList {};

    try {
        for (const auto& r : buffers) {
            auto buffer = this->get(r.buffer_id);
            auto req = m_memory_manager->create_request(buffer, r.memory_id, r.access_mode, parent);
            requests.push_back(BufferRequest {req});
        }

        return requests;
    } catch (...) {
        // Release the requests that have been created so far.
        for (const auto& r : requests) {
            m_memory_manager->release_request(r);
        }

        throw;
    }
}

Poll BufferRegistry::poll_requests(
    const BufferRequestList& requests,
    DeviceEventSet& dependencies_out
) {
    Poll result = Poll::Ready;

    for (const auto& req : requests) {
        if (m_memory_manager->poll_request(*req, dependencies_out) != Poll::Ready) {
            result = Poll::Pending;
        }
    }

    return result;
}

std::vector<BufferAccessor> BufferRegistry::access_requests(const BufferRequestList& requests) {
    auto accessors = std::vector<BufferAccessor> {};

    for (const auto& req : requests) {
        accessors.push_back(m_memory_manager->get_accessor(*req));
    }

    return accessors;
}

void BufferRegistry::release_requests(BufferRequestList& requests, DeviceEvent event) {
    for (auto& req : requests) {
        m_memory_manager->release_request(req, event);
    }

    requests.clear();
}

void BufferRegistry::poison_all(
    const std::vector<BufferRequirement>& buffers,
    PoisonException reason
) {
    for (const auto& r : buffers) {
        if (r.access_mode != AccessMode::Read) {
            this->poison(r.buffer_id, reason);
        }
    }
}

PoisonException::PoisonException(const std::string& error) {
    m_message = error;
}

const char* PoisonException::what() const noexcept {
    return m_message.c_str();
}
}  // namespace kmm
