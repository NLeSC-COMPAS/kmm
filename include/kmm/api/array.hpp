#pragma once

#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <vector>

#include "spdlog/spdlog.h"

#include "kmm/api/access.hpp"
#include "kmm/api/argument.hpp"
#include "kmm/api/array_handle.hpp"
#include "kmm/api/view_argument.hpp"
#include "kmm/dag/array_planner.hpp"
#include "kmm/dag/reduction_planner.hpp"

namespace kmm {

class ArrayBase {
  public:
    virtual ~ArrayBase() = default;
    virtual const std::type_info& type_info() const = 0;
    virtual size_t rank() const = 0;
    virtual int64_t size(size_t axis) const = 0;
    virtual const Worker& worker() const = 0;
    virtual void synchronize() const = 0;
    virtual void copy_bytes_to(void* output, size_t num_bytes) const = 0;
};

template<typename T, size_t N = 1>
class Array: public ArrayBase {
  public:
    Array(Dim<N> shape = {}) : m_shape(shape) {}

    explicit Array(std::shared_ptr<const ArrayHandle<N>> b) :
        m_handle(b),
        m_shape(m_handle->distribution().array_size()) {}

    const std::type_info& type_info() const final {
        return typeid(T);
    }

    size_t rank() const final {
        return N;
    }

    Dim<N> shape() const {
        return m_shape;
    }

    int64_t size(size_t axis) const final {
        return m_shape.get_or_default(axis);
    }

    int64_t size() const {
        return m_shape.volume();
    }

    bool is_empty() const {
        return m_shape.is_empty();
    }

    bool is_valid() const {
        return m_handle != nullptr;
    }

    const ArrayHandle<N>& handle() const {
        KMM_ASSERT(m_handle != nullptr);
        return *m_handle;
    }

    const Distribution<N>& distribution() const {
        return handle().distribution();
    }

    Dim<N> chunk_size() const {
        return distribution().chunk_size();
    }

    int64_t chunk_size(size_t axis) const {
        return chunk_size().get_or_default(axis);
    }

    const Worker& worker() const final {
        return handle().worker();
    }

    void synchronize() const final {
        if (m_handle) {
            m_handle->synchronize();
        }
    }

    void reset() {
        m_handle = nullptr;
    }

    template<typename M = All>
    Read<Array<T, N>, M> access(M mapper = {}) {
        return {*this, {std::move(mapper)}};
    }

    template<typename M = All>
    Read<const Array<T, N>, M> access(M mapper = {}) const {
        return {*this, {std::move(mapper)}};
    }

    template<typename M>
    auto operator[](M first_index) {
        return MultiIndexAccess<Array<T, N>, Read, N>(*this)[first_index];
    }

    template<typename M>
    auto operator[](M first_index) const {
        return MultiIndexAccess<const Array<T, N>, Read, N>(*this)[first_index];
    }

    template<typename... Is>
    Read<Array<T, N>, MultiIndexMap<N>> operator()(const Is&... index) {
        return access(bounds(index...));
    }

    template<typename... Is>
    Read<const Array<T, N>, MultiIndexMap<N>> operator()(const Is&... index) const {
        return access(bounds(index...));
    }

    void copy_bytes_to(void* output, size_t num_bytes) const {
        KMM_ASSERT(num_bytes % sizeof(T) == 0);
        KMM_ASSERT(compare_equal(num_bytes / sizeof(T), size()));

        handle().copy_bytes(output, sizeof(T));
    }

    void copy_to(T* output) const {
        handle().copy_bytes(output, sizeof(T));
    }

    template<typename I>
    void copy_to(T* output, I num_elements) const {
        KMM_ASSERT(compare_equal(num_elements, size()));
        handle().copy_bytes(output, sizeof(T));
    }

    void copy_to(std::vector<T>& output) const {
        output.resize(checked_cast<size_t>(size()));
        handle().copy_bytes(output.data(), sizeof(T));
    }

    std::vector<T> copy() const {
        std::vector<T> output;
        copy_to(output);
        return output;
    }

  private:
    std::shared_ptr<const ArrayHandle<N>> m_handle;
    Index<N> m_offset;  // Unused for now, always zero
    Dim<N> m_shape;
};

template<typename T>
using Scalar = Array<T, 0>;

template<typename T, size_t N>
struct ArgumentHandler<Read<const Array<T, N>>> {
    using type = ViewArgument<const T, views::dynamic_domain<N>>;

    ArgumentHandler(Read<const Array<T, N>> access) :
        m_handle(access.argument.handle().shared_from_this()),
        m_planner(&m_handle->instance()),
        m_array_shape(access.argument.shape()) {}

    void initialize(const TaskGroupInit& init) {}

    type before_submit(TaskInstance& task) {
        auto region = Bounds<N>(m_array_shape);
        size_t buffer_index = task.add_buffer_requirement(  //
            m_planner.prepare_access(task.graph, task.memory_id, region)
        );

        auto domain = views::dynamic_domain<N> {region.sizes()};
        return {buffer_index, domain};
    }

    void after_submit(const TaskSubmissionResult& result) {
        m_planner.finalize_access(result.graph, result.event_id);
    }

  private:
    std::shared_ptr<const ArrayHandle<N>> m_handle;
    ArrayReadPlanner<N> m_planner;
    Dim<N> m_array_shape;
};

template<typename T, size_t N>
struct ArgumentHandler<Array<T, N>>: ArgumentHandler<Read<const Array<T, N>>> {
    ArgumentHandler(Array<T, N> array) : ArgumentHandler<Read<const Array<T, N>>>(read(array)) {}
};

template<typename T, size_t N, typename M>
struct ArgumentHandler<Read<const Array<T, N>, M>> {
    using type = ViewArgument<const T, views::dynamic_subdomain<N>>;

    static_assert(
        is_dimensionality_accepted_by_mapper<M, N>,
        "mapper of 'read' must return N-dimensional region"
    );

    ArgumentHandler(Read<const Array<T, N>, M> access) :
        m_handle(access.argument.handle().shared_from_this()),
        m_planner(&m_handle->instance()),
        m_array_shape(access.argument.shape()),
        m_access_mapper(access.access_mapper) {}

    void initialize(const TaskGroupInit& init) {}

    type before_submit(TaskInstance& task) {
        Bounds<N> region = m_access_mapper(task.chunk, Bounds<N>(m_array_shape));
        auto buffer_index = task.add_buffer_requirement(  //
            m_planner.prepare_access(task.graph, task.memory_id, region)
        );

        auto domain = views::dynamic_subdomain<N> {region.begin(), region.sizes()};
        return {buffer_index, domain};
    }

    void after_submit(const TaskSubmissionResult& result) {
        m_planner.finalize_access(result.graph, result.event_id);
    }

  private:
    std::shared_ptr<const ArrayHandle<N>> m_handle;
    ArrayReadPlanner<N> m_planner;
    Dim<N> m_array_shape;
    M m_access_mapper;
};

template<typename T, size_t N>
struct ArgumentHandler<Write<Array<T, N>>> {
    using type = ViewArgument<T, views::dynamic_domain<N>>;

    ArgumentHandler(Write<Array<T, N>> access) : m_array(access.argument) {}

    void initialize(const TaskGroupInit& init) {
        if (!m_array.is_valid()) {
            m_handle = ArrayHandle<N>::instantiate(  //
                init.worker,
                map_domain_to_distribution(m_array.shape(), init.domain, All()),
                DataLayout::for_type<T>()
            );

            m_array = Array<T, N>(m_handle);
        }

        m_handle = m_array.handle().shared_from_this();
        m_planner = ArrayWritePlanner<N>(&m_handle->instance());
    }

    type before_submit(TaskInstance& task) {
        auto access_region = Bounds<N>(m_array.shape());
        auto buffer_index = task.add_buffer_requirement(
            m_planner.prepare_access(task.graph, task.memory_id, access_region)
        );

        auto domain = views::dynamic_domain<N> {access_region.sizes()};
        return {buffer_index, domain};
    }

    void after_submit(const TaskSubmissionResult& result) {
        m_planner.finalize_access(result.graph, result.event_id);
    }

  private:
    Array<T, N>& m_array;
    std::shared_ptr<const ArrayHandle<N>> m_handle;
    ArrayWritePlanner<N> m_planner;
};

template<typename T, size_t N, typename M>
struct ArgumentHandler<Write<Array<T, N>, M>> {
    using type = ViewArgument<T, views::dynamic_subdomain<N>>;

    static_assert(
        is_dimensionality_accepted_by_mapper<M, N>,
        "mapper of 'write' must return N-dimensional region"
    );

    ArgumentHandler(Write<Array<T, N>, M> access) :
        m_array(access.argument),
        m_shape(m_array.shape()),
        m_access_mapper(access.access_mapper) {}

    void initialize(const TaskGroupInit& init) {
        if (!m_array.is_valid()) {
            m_handle = ArrayHandle<N>::instantiate(  //
                init.worker,
                map_domain_to_distribution(m_shape, init.domain, m_access_mapper),
                DataLayout::for_type<T>()
            );

            m_array = Array<T, N>(m_handle);
        }

        m_handle = m_array.handle().shared_from_this();
        m_planner = ArrayWritePlanner<N>(&m_handle->instance());
    }

    type before_submit(TaskInstance& task) {
        auto access_region = m_access_mapper(task.chunk, Bounds<N>(m_shape));
        auto buffer_index = task.add_buffer_requirement(
            m_planner.prepare_access(task.graph, task.memory_id, access_region)
        );

        auto domain = views::dynamic_subdomain<N> {access_region.begin(), access_region.sizes()};
        return {buffer_index, domain};
    }

    void after_submit(const TaskSubmissionResult& result) {
        m_planner.finalize_access(result.graph, result.event_id);
    }

  private:
    Array<T, N>& m_array;
    std::shared_ptr<const ArrayHandle<N>> m_handle;
    Dim<N> m_shape;
    M m_access_mapper;
    ArrayWritePlanner<N> m_planner;
};

template<typename T, size_t N>
struct ArgumentHandler<Reduce<Array<T, N>>> {
    using type = ViewArgument<T, views::dynamic_domain<N>>;

    ArgumentHandler(Reduce<Array<T, N>> access) :
        m_array(access.argument),
        m_operation(access.op) {}

    void initialize(const TaskGroupInit& init) {
        if (!m_array.is_valid()) {
            Distribution<N> dist = map_domain_to_distribution(  //
                m_array.shape(),
                init.domain,
                All(),
                true
            );

            m_handle = ArrayHandle<N>::instantiate(  //
                init.worker,
                std::move(dist),
                DataLayout::for_type<T>()
            );

            m_array = Array<T, N>(m_handle);
        }

        m_handle = m_array.handle();
        m_planner = ReductionPlanner<N>(&m_handle->instance(), DataType::of<T>(), m_operation);
        m_remaining = init.domain.chunks.size();
    }

    type before_submit(TaskInstance& task) {
        auto access_region = m_access_mapper(task.chunk, Bounds<N>(m_array.shape()));

        size_t buffer_index = task.add_buffer_requirement(
            m_planner.prepare_access(task.graph, task.memory_id, access_region)
        );

        views::dynamic_domain<N> domain = {access_region.begin(), access_region.sizes()};

        return {buffer_index, domain};
    }

    void after_submit(const TaskSubmissionResult& result) {
        m_planner.finalize_access(result.graph, result.event_id);

        if (--m_remaining == 0) {
            m_planner.finalize(result.graph);
        }
    }

  private:
    Array<T, N>& m_array;
    std::shared_ptr<const ArrayHandle<N>> m_handle;
    Reduction m_operation;
    ReductionPlanner<N> m_planner;
    size_t m_remaining = 0;
};

template<typename T, size_t N, typename M, typename P>
struct ArgumentHandler<Reduce<Array<T, N>, M, P>> {
    static constexpr size_t K = mapper_dimensionality<P>;
    using type = ViewArgument<T, views::dynamic_subdomain<K + N>>;

    static_assert(
        is_dimensionality_accepted_by_mapper<M, N>,
        "mapper of 'reduce' must return N-dimensional region"
    );

    static_assert(
        is_dimensionality_accepted_by_mapper<P, K>,
        "private mapper of 'reduce' must return K-dimensional region"
    );

    ArgumentHandler(Reduce<Array<T, N>, M, P> access) :
        m_array(access.argument),
        m_operation(access.op),
        m_access_mapper(access.access_mapper),
        m_private_mapper(access.private_mapper) {}

    void initialize(const TaskGroupInit& init) {
        if (!m_array.is_valid()) {
            m_handle = ArrayHandle<N>::instantiate(  //
                init.worker,
                map_domain_to_distribution(  //
                    m_array.shape(),
                    init.domain,
                    All(),
                    true
                ),
                DataLayout::for_type<T>()
            );

            m_array = Array<T, N>(m_handle);
        }

        m_handle = m_array.handle().shared_from_this();
        m_planner = ReductionPlanner<N>(&m_handle->instance(), DataType::of<T>(), m_operation);
        m_remaining = init.domain.chunks.size();
    }

    type before_submit(TaskInstance& task) {
        auto access_region = m_access_mapper(task.chunk, Bounds<N>(m_array.shape()));
        auto private_region = m_private_mapper(task.chunk);

        auto rep = checked_cast<size_t>(private_region.size());
        size_t buffer_index = task.add_buffer_requirement(
            m_planner.prepare_access(task.graph, task.memory_id, access_region, rep)
        );

        views::dynamic_subdomain<K + N> domain = {
            concat(private_region, access_region).begin(),
            concat(private_region, access_region).sizes()};

        return {buffer_index, domain};
    }

    void after_submit(const TaskSubmissionResult& result) {
        m_planner.finalize_access(result.graph, result.event_id);

        if (--m_remaining == 0) {
            m_planner.finalize(result.graph);
        }
    }

  private:
    Array<T, N>& m_array;
    std::shared_ptr<const ArrayHandle<N>> m_handle;
    Reduction m_operation;
    ReductionPlanner<N> m_planner;
    M m_access_mapper;
    P m_private_mapper;
    size_t m_remaining = 0;
};

}  // namespace kmm