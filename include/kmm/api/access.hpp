#pragma once

#include "kmm/api/argument.hpp"
#include "kmm/api/mapper.hpp"

namespace kmm {

template<typename Arg, typename Mode>
struct Access {
    Arg& argument;
    Mode mode = {};

    Access(Arg& argument, Mode mode = {}) : argument(argument), mode(mode) {}

    template<typename U>
    Access(Access<U, Mode> that = {}) : argument(that.argument), mode(that.mode) {}
};

template<typename M = All>
struct Read {
    M access_mapper = {};
};

template<typename M = All>
struct Write {
    M access_mapper = {};
};

template<typename M = All, typename P = MultiIndexMap<0>>
struct Reduce {
    Reduction op;
    M access_mapper = {};
    P private_mapper = {};
};

template<typename M = All, typename Arg>
Access<const Arg, Read<M>> read(const Arg& argument, M access_mapper = {}) {
    return {argument, {access_mapper}};
}

template<typename Arg, typename M>
Access<const Arg, Read<M>> read(Access<Arg, Read<M>> access) {
    return access;
}

template<typename M = All, typename Arg>
Access<Arg, Write<M>> write(Arg& argument, M access_mapper = {}) {
    return {argument, {access_mapper}};
}

template<typename Arg, typename M>
Access<Arg, Write<M>> write(Access<Arg, Read<M>> access) {
    return {access.argument, {access.mode.access_mapper}};
}

template<typename M>
struct Privatize {
    M access_mapper;

    explicit Privatize(M access_mapper) :  //
        access_mapper(std::move(access_mapper)) {}
};

template<typename... Is>
Privatize<MultiIndexMap<sizeof...(Is)>> privatize(const Is&... slices) {
    return Privatize {bounds(slices...)};
}

template<typename M = All, typename Arg>
Access<Arg, Reduce<M>> reduce(Reduction op, Arg& argument, M access_mapper = {}) {
    return {argument, {op, access_mapper}};
}

template<typename M = All, typename Arg, typename P>
Access<Arg, Reduce<M, P>> reduce(
    Reduction op,
    Privatize<P> private_mapper,
    Arg& argument,
    M access_mapper = {}
) {
    return {argument, {op, access_mapper, private_mapper.access_mapper}};
}

template<typename M = All, typename Arg>
Access<Arg, Reduce<M>> reduce(Reduction op, Access<Arg, Read<M>> access) {
    return {access.argument, {op, access.mode.access_mapper}};
}

template<typename M = All, typename Arg, typename P>
Access<Arg, Reduce<M, P>> reduce(
    Reduction op,
    Privatize<P> private_mapper,
    Access<Arg, Read<M>> access
) {
    return {access.argument, {op, access.mode.access_mapper, private_mapper.access_mapper}};
}

// Forward `Access<Arg, Read<M>>` to `Access<const Arg, Read<M>>` only if `Arg` is not const
template<typename Arg, typename M>
struct ArgumentHandler<Access<Arg, Read<M>>, std::enable_if_t<!std::is_const_v<Arg>>>:
    ArgumentHandler<Access<const Arg, Read<M>>> {
    ArgumentHandler(Access<Arg, Read<M>> access) :
        ArgumentHandler<Access<const Arg, Read<M>>>(access) {}
};

}  // namespace kmm