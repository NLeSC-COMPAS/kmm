#pragma once

#include <memory>
#include <vector>

#include "kmm/api/array.hpp"
#include "kmm/api/launcher.hpp"
#include "kmm/api/parallel_submit.hpp"
#include "kmm/api/struct_argument.hpp"
#include "kmm/core/config.hpp"
#include "kmm/core/system_info.hpp"
#include "kmm/core/view.hpp"
#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/panic.hpp"
#include "kmm/utils/range.hpp"

namespace kmm {

class Runtime;

class RuntimeHandle {
    struct Impl;
    RuntimeHandle(std::shared_ptr<Impl> impl);

  public:
    RuntimeHandle(std::shared_ptr<Runtime> rt);
    RuntimeHandle(Runtime& rt);

    /**
     * Submit a single task to the runtime system.
     *
     * @param index_space The index space defining the task dimensions.
     * @param target The target processor for the task.
     * @param launcher The task launcher.
     * @param args The arguments that are forwarded to the launcher.
     * @return The event identifier for the submitted task.
     */
    template<typename L, typename... Args>
    EventId submit(ResourceId target, L&& launcher, Args&&... args) const {
        DomainChunk chunk = {
            .owner_id = target,  //
            .offset = DomainPoint::zero(),
            .size = DomainDim::one()
        };

        return kmm::parallel_submit(
            worker(),
            info(),
            Domain {{chunk}},
            std::forward<L>(launcher),
            std::forward<Args>(args)...
        );
    }

    /**
     * Submit a set of tasks to the runtime systems.
     *
     * @param dist The domain describing how the work is split.
     * @param launcher The task launcher.
     * @param args The arguments that are forwarded to the launcher.
     * @return The event identifier for the submitted task.
     */
    template<typename D, typename L, typename... Args>
    EventId parallel_submit(D&& domain, L&& launcher, Args&&... args) const {
        return kmm::parallel_submit(
            worker(),
            info(),
            IntoDomain<std::decay_t<D>>::call(
                std::forward<D>(domain),
                info(),
                std::decay_t<L>::execution_space
            ),
            std::forward<L>(launcher),
            std::forward<Args>(args)...
        );
    }

    /**
     * Submit a set of tasks to the runtime systems.
     *
     * @param domain_size The index space defining the domain dimensions.
     * @param partitioner The partitioner describing how the work is split.
     * @param launcher The task launcher.
     * @param args The arguments that are forwarded to the launcher.
     * @return The event identifier for the submitted task.
     */
    template<typename L, typename... Args>
    EventId parallel_submit(
        DomainDim domain_size,
        DomainDim chunk_size,
        L&& launcher,
        Args&&... args
    ) const {
        return this->parallel_submit(
            TileDomain(domain_size, chunk_size),
            std::forward<L>(launcher),
            std::forward<Args>(args)...
        );
    }

    /**
     * Allocates an array in memory with the given shape and memory affinity.
     *
     * The pointer to the given buffer should contain `shape[0] * shape[1] * shape[2]...`
     * elements.
     *
     * @param data Pointer to the array data.
     * @param shape Shape of the array.
     * @param memory_id Identifier of the memory region.
     * @return The allocated Array object.
     */
    template<size_t N = 1, typename T>
    Array<T, N> allocate(const T* data, Dim<N> shape, MemoryId memory_id) const {
        auto handle = ArrayInstance<N>::create(
            worker(),
            Distribution<N> {shape, shape, {memory_id}},
            DataType::of<T>()
        );

        handle->copy_bytes_from(data);
        return Array<T, N> {std::move(handle)};
    }

    /**
     * Allocates an array in memory with the given shape.
     *
     * The pointer to the given buffer should contain `shape[0] * shape[1] * shape[2]...`
     * elements.
     *
     * In which memory the data will be allocated is determined by `memory_affinity_for_address`.
     *
     * @param data Pointer to the array data.
     * @param shape Shape of the array.
     * @return The allocated Array object.
     */
    template<size_t N = 1, typename T>
    Array<T, N> allocate(const T* data, Dim<N> shape) const {
        return allocate(data, shape, memory_affinity_for_address(data));
    }

    /**
     * Alias for `allocate(v.data(), v.sizes())`
     */
    template<typename T, size_t N = 1>
    Array<T, N> allocate(View<T, N> v) const {
        return allocate(v.data(), v.sizes());
    }

    /**
     * Alias for `allocate(data, Dim<N>{sizes...})`
     */
    template<typename T, typename... Is>
    Array<T, sizeof...(Is)> allocate(const T* data, const Is&... num_elements) const {
        return allocate(data, Dim<sizeof...(Is)> {checked_cast<int64_t>(num_elements)...});
    }

    /**
     * Alias for `allocate(v.data(), v.size())`
     */
    template<typename T>
    Array<T> allocate(const std::vector<T>& v) const {
        return allocate(v.data(), v.size());
    }

    /**
     * Alias for `allocate(v.begin(), v.size())`
     */
    template<typename T>
    Array<T> allocate(std::initializer_list<T> v) const {
        return allocate(v.begin(), v.size());
    }

    /**
     * Returns the memory affinity for a given address.
     */
    MemoryId memory_affinity_for_address(const void* address) const;

    /**
     * Returns a new event that triggers when all the given events have triggered.
     */
    EventId join(EventList events) const;

    /**
     * Returns a new event that triggers when all the given events have triggered. Each argument
     * must be convertible to an `EventId`.
     */
    template<typename... Es>
    EventId join(Es... events) const {
        return join(EventList(EventId(events)...));
    }

    /**
     * Returns `true` if the event with the provided identifier has finished, or `false` otherwise.
     */
    bool is_done(EventId) const;

    /**
     * Block the current thread until the event with the provided identifier completes.
     */
    void wait(EventId id) const;

    /**
     * Block the current thread until the event with the provided id completes. Blocks until
     * either the event completes or the deadline is exceeded, whatever comes first.
     *
     * @return `true` if the event with the provided id has finished, otherwise returns `false`.
     */
    bool wait_until(EventId id, std::chrono::system_clock::time_point deadline) const;

    /**
     * Block the current thread until the event with the provided id completes. Blocks until
     * either the event completes or the duration is exceeded, whatever comes first.
     *
     * @return `true` if the event with the provided id has finished, otherwise returns `false`.
     */
    bool wait_for(EventId id, std::chrono::system_clock::duration duration) const;

    /**
     * Submit a barrier the runtime system. The barrier completes once all the tasks submitted
     * to the runtime system so far have finished.
     *
     * @return The identifier of the barrier.
     */
    EventId barrier() const;

    /**
     * Blocks until all the tasks submitted to the runtime system have finished and the
     * system has become idle.
     */
    void synchronize() const;

    /**
     * Return a new `RuntimeHandle` that is constrained to the given set of resources. In other
     * words, it only can submit work onto those resources.
     */
    RuntimeHandle constrain_to(std::vector<ResourceId> resources) const;

    /**
     * Return a new `RuntimeHandle` that is constrained to the given device. In other
     * words, it only can submit work onto that device.
     */
    RuntimeHandle constrain_to(DeviceId device) const;

    /**
     * Return a new `RuntimeHandle` that is constrained to the given resource. In other
     * words, it only can submit work onto that resource.
     */
    RuntimeHandle constrain_to(ResourceId resource) const;

    /**
     * Returns information about the current system.
     */
    const SystemInfo& info() const;

    /**
     * Returns the inner `Worker`.
     */
    Runtime& worker() const;

  private:
    std::shared_ptr<Impl> m_data;
};

RuntimeHandle make_runtime(const RuntimeConfig& config = default_config_from_environment());

}  // namespace kmm
