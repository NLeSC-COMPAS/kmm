#pragma once

#include "kmm/api/mapper.hpp"

namespace kmm {
template<typename T, typename I = All>
struct Read {
    T argument;
    I access_mapper = {};
};

template<typename T, typename I = All>
struct Write {
    T& argument;
    I access_mapper = {};
};

template<typename I>
struct Privatize {
    I access_mapper;
};

template<typename T, typename I = All, typename R = MultiIndexMap<0>>
struct Reduce {
    T& argument;
    Reduction op;
    I access_mapper = {};
    R private_mapper = {};
};

template<typename I = All, typename T>
Read<T, I> read(T argument, I access_mapper = {}) {
    return {argument, access_mapper};
}

template<typename I = All, typename T>
Write<T, I> write(T& argument, I access_mapper = {}) {
    return {argument, access_mapper};
}

template<typename... Is>
Privatize<MultiIndexMap<sizeof...(Is)>> privatize(const Is&... slices) {
    return {into_index_map(slices)...};
}

template<typename T>
Reduce<T> reduce(Reduction op, T& argument) {
    return {argument, op};
}

template<typename T, typename I>
Reduce<T, I> reduce(Reduction op, T& argument, I access_mapper) {
    return {argument, op, access_mapper};
}

template<typename T, typename I, typename P>
Reduce<T, I, P> reduce(Reduction op, T& argument, Privatize<P> private_mapper, I access_mapper) {
    return {argument, op, access_mapper, private_mapper.access_mapper};
}

template<typename T, typename P>
Reduce<T, All, P> reduce(Reduction op, T& argument, Privatize<P> private_mapper) {
    return {argument, op, All {}, private_mapper.access_mapper};
}

}  // namespace kmm