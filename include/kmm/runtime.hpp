#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "kmm/executor.hpp"
#include "kmm/types.hpp"
#include "kmm/utils.hpp"

namespace kmm {

class Event {
  public:
    Event(OperationId id, std::shared_ptr<RuntimeImpl> runtime) :
        m_id(id),
        m_runtime(std::move(runtime)) {}

  private:
    OperationId id() const;
    void wait() const;

  private:
    OperationId m_id;
    std::shared_ptr<RuntimeImpl> m_runtime;
};

class Buffer {
  public:
    Buffer(BufferId id, std::shared_ptr<RuntimeImpl> rt);
    Buffer(Buffer&&) = delete;
    Buffer(const Buffer&) = default;

    BufferId id() const;
    Runtime runtime() const;
    Event barrier() const;
    void destroy() const;

  private:
    class Lifetime;
    std::shared_ptr<Lifetime> m_lifetime;
};

template<typename T, size_t N = 1>
class Array {
  public:
    Array(Buffer buffer, std::array<index_t, N> sizes) : m_buffer(buffer), m_sizes(sizes) {}

    std::array<index_t, N> sizes() const {
        return m_sizes;
    }

    index_t size(size_t axis) const {
        return m_sizes.at(axis);
    }

    index_t size() const {
        return checked_product(m_sizes.begin(), m_sizes.end());
    }

    Buffer buffer() const {
        return m_buffer;
    }

    BufferId id() const {
        return m_buffer.id();
    }

    Event barrier() const {
        return m_buffer.barrier();
    }

    Runtime runtime() const;

  private:
    Buffer m_buffer;
    std::array<index_t, N> m_sizes;
};

class Runtime {
  public:
    Runtime(std::shared_ptr<RuntimeImpl> impl);

    Buffer allocate_buffer(
        size_t num_elements,
        size_t element_size,
        size_t element_align,
        DeviceId home = DeviceId(0)) const;

    void submit_task(
        DeviceId device_id,
        std::shared_ptr<Task> task,
        std::vector<VirtualBufferRequirement> buffers = {},
        std::vector<OperationId> dependencies = {}) const;

    Event barrier() const;
    Event barrier_buffer(BufferId) const;

    template<typename T, size_t N = 1>
    Array<T, N> allocate(std::array<index_t, N> sizes, DeviceId home = DeviceId(0)) const {
        size_t total_size = checked_product(sizes.begin(), sizes.end());
        return {allocate_buffer(total_size, sizeof(T), alignof(T), home), sizes};
    }

    template<typename Device, typename Fun, typename... Args>
    void submit(const Device& device, Fun fun, Args... args) {
        KMM_TODO();
        DeviceId device_id = device.find_id(*this);
        std::shared_ptr<Task> task;
        std::vector<VirtualBufferRequirement> buffers;

        submit_task(device_id, std::move(task), std::move(buffers));
    }

    std::shared_ptr<RuntimeImpl> inner() const {
        return m_impl;
    }

  private:
    std::shared_ptr<RuntimeImpl> m_impl;
};

Runtime build_runtime();

}  // namespace kmm