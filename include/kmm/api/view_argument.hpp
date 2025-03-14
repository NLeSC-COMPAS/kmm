#pragma once

#include "kmm/api/argument.hpp"
#include "kmm/core/view.hpp"

namespace kmm {

template<typename T, typename D, typename L = views::default_layout<D::rank>>
struct ViewArgument {
    using value_type = T;
    using domain_type = D;
    using layout_type = L;

    ViewArgument(size_t buffer_index, D domain, L layout) :
        buffer_index(buffer_index),
        domain(domain),
        layout(layout) {}

    ViewArgument(size_t buffer_index, D domain) :
        ViewArgument(buffer_index, domain, L::from_domain(domain)) {}

    size_t buffer_index;
    D domain;
    L layout;
};

template<typename T, typename D, typename L>
struct ArgumentUnpack<ExecutionSpace::Host, ViewArgument<T, D, L>> {
    using type = AbstractView<T, D, L, views::host_accessor>;

    static type call(const TaskContext& context, ViewArgument<T, D, L> arg) {
        T* data = static_cast<T*>(context.accessors.at(arg.buffer_index).address);
        return {data, arg.domain, arg.layout};
    }
};

template<typename T, typename D, typename L>
struct ArgumentUnpack<ExecutionSpace::Host, ViewArgument<const T, D, L>> {
    using type = AbstractView<const T, D, L, views::host_accessor>;

    static type call(const TaskContext& context, ViewArgument<const T, D, L> arg) {
        const T* data = static_cast<const T*>(context.accessors.at(arg.buffer_index).address);
        return {data, arg.domain, arg.layout};
    }
};

template<typename T, typename D, typename L>
struct ArgumentUnpack<ExecutionSpace::Device, ViewArgument<T, D, L>> {
    using type = AbstractView<T, D, L, views::device_accessor>;

    static type call(const TaskContext& context, ViewArgument<T, D, L> arg) {
        T* data = static_cast<T*>(context.accessors.at(arg.buffer_index).address);
        return {data, arg.domain, arg.layout};
    }
};

template<typename T, typename D, typename L>
struct ArgumentUnpack<ExecutionSpace::Device, ViewArgument<const T, D, L>> {
    using type = AbstractView<const T, D, L, views::device_accessor>;

    static type call(const TaskContext& context, ViewArgument<const T, D, L> arg) {
        const T* data = static_cast<const T*>(context.accessors.at(arg.buffer_index).address);
        return {data, arg.domain, arg.layout};
    }
};

}  // namespace kmm