#pragma once

#include "kmm/api/task_group.hpp"
#include "kmm/core/resource.hpp"

namespace kmm {

template<typename T, typename Enabled = void>
struct ArgumentHandler;

template<typename T>
struct ArgumentHandler<const T&>: ArgumentHandler<T> {
    ArgumentHandler(const T& arg) : ArgumentHandler<T>(arg) {}
};

template<typename T>
struct ArgumentHandler<T&>: ArgumentHandler<const T&> {
    ArgumentHandler(T& arg) : ArgumentHandler<const T&>(arg) {}
};

template<typename T>
struct ArgumentHandler<T&&>: ArgumentHandler<T> {
    ArgumentHandler(T&& arg) : ArgumentHandler<T>(std::move(arg)) {}
};

template<typename T>
using packed_argument_t = typename ArgumentHandler<T>::type;

template<typename T>
packed_argument_t<T> pack_argument(TaskInstance& task, T&& arg) {
    return ArgumentHandler<T>(std::forward<T>(arg)).before_submit(task);
}

template<ExecutionSpace, typename T>
struct ArgumentUnpack;

template<ExecutionSpace execution_space, typename T>
auto unpack_argument(TaskContext& context, T&& arg) {
    return ArgumentUnpack<execution_space, std::decay_t<T>>::call(context, std::forward<T>(arg));
}

template<typename T, typename = void>
struct Argument {
    Argument(T value) : m_value(std::move(value)) {}

    static Argument pack(TaskInstance& builder, T value) {
        return Argument {std::move(value)};
    }

    template<ExecutionSpace Space>
    T unpack(TaskContext& context) {
        return m_value;
    }

  private:
    T m_value;
};

template<typename T, typename Enabled>
struct ArgumentHandler {
    using type = Argument<T>;

    ArgumentHandler(T value) : m_value(std::move(value)) {}

    void initialize(const TaskGroupInit& init) {
        // Nothing to do
    }

    type before_submit(TaskInstance& builder) {
        return Argument<T>::pack(builder, m_value);
    }

    void after_submit(const TaskSubmissionResult& result) {
        // Nothing to do
    }

    void commit(const TaskGroupCommit& commit) {
        // Nothing to do
    }

  private:
    T m_value;
};

template<ExecutionSpace Space, typename T>
struct ArgumentUnpack<Space, Argument<T>> {
    static auto call(TaskContext& context, Argument<T>& data) {
        return data.template unpack<Space>(context);
    }
};

}  // namespace kmm