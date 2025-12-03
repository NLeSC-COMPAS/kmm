#pragma once

#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <vector>

#include "spdlog/spdlog.h"

#include "kmm/api/access.hpp"
#include "kmm/api/argument.hpp"
#include "kmm/api/array_instance.hpp"
#include "kmm/api/view_argument.hpp"
#include "kmm/planner/read_planner.hpp"
#include "kmm/planner/reduction_planner.hpp"
#include "kmm/planner/write_planner.hpp"

namespace kmm {

class ArrayBase {
  public:
    virtual ~ArrayBase() = default;
    virtual const std::type_info& type_info() const = 0;
    virtual size_t rank() const = 0;
    virtual int64_t size(size_t axis) const = 0;
    virtual const Runtime& runtime() const = 0;
    virtual void synchronize() const = 0;
    virtual void copy_bytes_to(void* output, size_t num_bytes) const = 0;
};

template<typename T, size_t N = 1>
class Array: public ArrayBase {
  public:
    Array(Dim<N> shape = {}) : m_shape(shape) {}

    explicit Array(std::shared_ptr<ArrayInstance<N>> b) :
        m_instance(b),
        m_shape(m_instance->distribution().array_size()) {}

    const std::type_info& type_info() const final {
        return typeid(T);
    }

    size_t rank() const final {
        return N;
    }

    Dim<N> size() const {
        return m_shape;
    }

    int64_t size(size_t axis) const final {
        return m_shape.get_or_default(axis);
    }

    int64_t volume() const {
        return m_shape.volume();
    }

    bool is_empty() const {
        return m_shape.is_empty();
    }

    bool has_instance() const {
        return m_instance != nullptr;
    }

    ArrayInstance<N>& instance() const {
        if (m_instance == nullptr) {
            throw_uninitialized_array_exception();
        }

        return *m_instance;
    }

    const Distribution<N>& distribution() const {
        return instance().distribution();
    }

    Dim<N> chunk_size() const {
        return distribution().chunk_size();
    }

    int64_t chunk_size(size_t axis) const {
        return chunk_size().get_or_default(axis);
    }

    Runtime& runtime() const final {
        return instance().runtime();
    }

    void synchronize() const final {
        if (m_instance) {
            m_instance->synchronize();
        }
    }

    void reset() {
        m_instance = nullptr;
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
        KMM_ASSERT(is_equal(num_bytes / sizeof(T), volume()));
        instance().copy_bytes_into(output);
    }

    void copy_to(T* output) const {
        instance().copy_bytes_into(output);
    }

    template<typename I>
    void copy_to(T* output, I num_elements) const {
        KMM_ASSERT(is_equal(num_elements, volume()));
        instance().copy_bytes_into(output);
    }

    void copy_to(std::vector<T>& output) const {
        output.resize(checked_cast<size_t>(volume()));
        instance().copy_bytes_into(output.data());
    }

    std::vector<T> copy_to_vector() const {
        std::vector<T> output(volume());
        copy_to(output);
        return output;
    }

    void copy_bytes_from(const void* input, size_t num_bytes) const {
        KMM_ASSERT(num_bytes % sizeof(T) == 0);
        KMM_ASSERT(is_equal(num_bytes / sizeof(T), volume()));
        instance().copy_bytes_from(input);
    }

    void copy_from(T* input) const {
        instance().copy_bytes_from(input);
    }

    template<typename I>
    void copy_from(T* input, I num_elements) const {
        KMM_ASSERT(is_equal(num_elements, volume()));
        instance().copy_bytes_from(input);
    }

    void copy_from(const std::vector<T>& input) const {
        copy_from(input.data(), input.size());
    }

  private:
    std::shared_ptr<ArrayInstance<N>> m_instance;
    Point<N> m_offset;  // Unused for now, always zero
    Dim<N> m_shape;
};

template<typename T>
using Scalar = Array<T, 0>;

template<typename T, size_t N>
struct ArgumentHandler<Read<const Array<T, N>>> {
    using type = ViewArgument<const T, views::dynamic_domain<N>>;

    ArgumentHandler(Read<const Array<T, N>> access) :
        m_planner(access.argument.instance().shared_from_this()),
        m_array_shape(access.argument.size()) {}

    void initialize(const TaskGroupInit& init) {}

    type before_submit(TaskInstance& task) {
        auto region = Bounds<N>(m_array_shape);
        size_t buffer_index = task.add_buffer_requirement(  //
            m_planner.prepare_access(task.graph, task.memory_id, region, task.dependencies)
        );

        auto domain = views::dynamic_domain<N> {region.size()};
        return {buffer_index, domain};
    }

    void after_submit(const TaskSubmissionResult& result) {
        m_planner.finalize_access(result.graph, result.event_id);
    }

    void commit(const TaskGroupCommit& commit) {
        m_planner.commit(commit.graph);
    }

  private:
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
        m_planner(access.argument.instance().shared_from_this()),
        m_array_shape(access.argument.size()),
        m_access_mapper(access.access_mapper) {}

    void initialize(const TaskGroupInit& init) {}

    type before_submit(TaskInstance& task) {
        Bounds<N> region = m_access_mapper(task.chunk, Bounds<N>(m_array_shape));
        auto buffer_index = task.add_buffer_requirement(  //
            m_planner.prepare_access(task.graph, task.memory_id, region, task.dependencies)
        );

        auto domain = views::dynamic_subdomain<N> {region.begin(), region.size()};
        return {buffer_index, domain};
    }

    void after_submit(const TaskSubmissionResult& result) {
        m_planner.finalize_access(result.graph, result.event_id);
    }

    void commit(const TaskGroupCommit& commit) {
        m_planner.commit(commit.graph);
    }

  private:
    ArrayReadPlanner<N> m_planner;
    Dim<N> m_array_shape;
    M m_access_mapper;
};

template<typename T, size_t N>
struct ArgumentHandler<Write<Array<T, N>>> {
    using type = ViewArgument<T, views::dynamic_domain<N>>;

    ArgumentHandler(Write<Array<T, N>> access) : m_array(access.argument) {}

    void initialize(const TaskGroupInit& init) {
        if (!m_array.has_instance()) {
            auto instance = ArrayInstance<N>::create(  //
                init.runtime,
                map_domain_to_distribution(m_array.size(), init.domain, All()),
                DataType::of<T>()
            );

            m_array = Array<T, N>(instance);
        }

        m_planner = std::make_unique<ArrayWritePlanner<N>>(m_array.instance().shared_from_this());
    }

    type before_submit(TaskInstance& task) {
        auto access_region = Bounds<N>(m_array.size());
        auto buffer_index = task.add_buffer_requirement(
            m_planner->prepare_access(task.graph, task.memory_id, access_region, task.dependencies)
        );

        auto domain = views::dynamic_domain<N> {access_region.size()};
        return {buffer_index, domain};
    }

    void after_submit(const TaskSubmissionResult& result) {
        m_planner->finalize_access(result.graph, result.event_id);
    }

    void commit(const TaskGroupCommit& commit) {
        m_planner->commit(commit.graph);
    }

  private:
    Array<T, N>& m_array;
    std::unique_ptr<ArrayWritePlanner<N>> m_planner;
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
        m_shape(m_array.size()),
        m_access_mapper(access.access_mapper) {}

    void initialize(const TaskGroupInit& init) {
        if (!m_array.has_instance()) {
            auto instance = ArrayInstance<N>::create(  //
                init.runtime,
                map_domain_to_distribution(m_array.size(), init.domain, m_access_mapper),
                DataType::of<T>()
            );

            m_array = Array<T, N>(instance);
        }

        m_planner = std::make_unique<ArrayWritePlanner<N>>(m_array.instance().shared_from_this());
    }

    type before_submit(TaskInstance& task) {
        auto access_region = m_access_mapper(task.chunk, Bounds<N>(m_shape));
        auto buffer_index = task.add_buffer_requirement(
            m_planner->prepare_access(task.graph, task.memory_id, access_region, task.dependencies)
        );

        auto domain = views::dynamic_subdomain<N> {access_region.begin(), access_region.size()};
        return {buffer_index, domain};
    }

    void after_submit(const TaskSubmissionResult& result) {
        m_planner->finalize_access(result.graph, result.event_id);
    }

    void commit(const TaskGroupCommit& commit) {
        m_planner->commit(commit.graph);
    }

  private:
    Array<T, N>& m_array;
    Dim<N> m_shape;
    M m_access_mapper;
    std::unique_ptr<ArrayWritePlanner<N>> m_planner;
};

template<typename T, size_t N>
struct ArgumentHandler<Reduce<Array<T, N>>> {
    using type = ViewArgument<T, views::dynamic_domain<N>>;

    ArgumentHandler(Reduce<Array<T, N>> access) :
        m_array(access.argument),
        m_operation(access.op) {}

    void initialize(const TaskGroupInit& init) {
        if (!m_array.has_instance()) {
            auto instance = ArrayInstance<N>::create(  //
                init.runtime,
                map_domain_to_distribution(  //
                    m_array.size(),
                    init.domain,
                    All(),
                    true
                ),
                DataType::of<T>()
            );

            m_array = Array<T, N>(instance);
        }

        m_planner = std::make_unique<ArrayReductionPlanner<N>>(
            m_array.instance().shared_from_this(),
            m_operation
        );
    }

    type before_submit(TaskInstance& task) {
        auto access_region = Bounds<N>(m_array.size());

        size_t buffer_index = task.add_buffer_requirement(
            m_planner
                ->prepare_access(task.graph, task.memory_id, access_region, 1, task.dependencies)
        );

        views::dynamic_domain<N> domain = {access_region.size()};

        return {buffer_index, domain};
    }

    void after_submit(const TaskSubmissionResult& result) {
        m_planner->finalize_access(result.graph, result.event_id);
    }

    void commit(const TaskGroupCommit& commit) {
        m_planner->commit(commit.graph);
    }

  private:
    Array<T, N>& m_array;
    Reduction m_operation;
    std::unique_ptr<ArrayReductionPlanner<N>> m_planner;
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
        if (!m_array.has_instance()) {
            auto instance = ArrayInstance<N>::create(  //
                init.runtime,
                map_domain_to_distribution(  //
                    m_array.size(),
                    init.domain,
                    m_access_mapper,
                    true
                ),
                DataType::of<T>()
            );

            m_array = Array<T, N>(instance);
        }

        m_planner = std::make_unique<ArrayReductionPlanner<N>>(
            m_array.instance().shared_from_this(),
            m_operation
        );
    }

    type before_submit(TaskInstance& task) {
        auto access_region = m_access_mapper(task.chunk, Bounds<N>(m_array.size()));
        auto private_region = m_private_mapper(task.chunk);

        auto rep = checked_cast<size_t>(private_region.volume());
        size_t buffer_index = task.add_buffer_requirement(
            m_planner
                ->prepare_access(task.graph, task.memory_id, access_region, rep, task.dependencies)
        );

        views::dynamic_subdomain<K + N> domain = {
            concat(private_region, access_region).begin(),
            concat(private_region, access_region).size()
        };

        return {buffer_index, domain};
    }

    void after_submit(const TaskSubmissionResult& result) {
        m_planner->finalize_access(result.graph, result.event_id);
    }

    void commit(const TaskGroupCommit& commit) {
        m_planner->commit(commit.graph);
    }

  private:
    Array<T, N>& m_array;
    Reduction m_operation;
    std::unique_ptr<ArrayReductionPlanner<N>> m_planner;
    M m_access_mapper;
    P m_private_mapper;
};

}  // namespace kmm