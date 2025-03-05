#pragma once

#include "kmm/api/argument.hpp"
#include "kmm/api/mapper.hpp"

namespace kmm {

/**
 * Encapsulates read access to an argument.
 */
template<typename Arg, typename M = All>
struct Read {
    Arg& argument;
    M access_mapper = {};
};

template<typename M = All, typename Arg>
Read<const Arg, M> read(const Arg& argument, M access_mapper = {}) {
    return {argument, access_mapper};
}

template<typename Arg, typename M>
Read<const Arg, M> read(Read<Arg, M> access) {
    return access;
}

/**
 * Encapsulates write access to an argument.
 */
template<typename Arg, typename M = All>
struct Write {
    Arg& argument;
    M access_mapper = {};
};

template<typename M = All, typename Arg>
Write<Arg, M> write(Arg& argument, M access_mapper = {}) {
    return {argument, access_mapper};
}

template<typename Arg, typename M>
Write<Arg, M> write(Read<Arg, M> access) {
    return {access.argument, access.access_mapper};
}

/**
 * Encapsulates reduce access to an argument.
 */
template<typename Arg, typename M = All, typename P = MultiIndexMap<0>>
struct Reduce {
    Arg& argument;
    Reduction op;
    M access_mapper = {};
    P private_mapper = {};
};

template<typename M>
struct Privatize {
    M access_mapper;

    explicit Privatize(M access_mapper) :  //
        access_mapper(std::move(access_mapper)) {}
};

template<typename M>
Privatize<M> privatize(const M& mapper) {
    return Privatize {mapper};
}

template<typename... Is>
Privatize<MultiIndexMap<sizeof...(Is)>> privatize(const Is&... slices) {
    return Privatize {bounds(slices...)};
}

template<typename M = All, typename Arg>
Reduce<Arg, M> reduce(Reduction op, Arg& argument, M access_mapper = {}) {
    return {argument, op, access_mapper};
}

template<typename M = All, typename Arg, typename P>
Reduce<Arg, M, P> reduce(
    Reduction op,
    Privatize<P> private_mapper,
    Arg& argument,
    M access_mapper = {}
) {
    return {argument, op, access_mapper, private_mapper.access_mapper};
}

template<typename M, typename Arg>
Reduce<Arg, M> reduce(Reduction op, Read<Arg, M> access) {
    return {access.argument, op, access.access_mapper};
}

template<typename M, typename Arg, typename P>
Reduce<Arg, M, P> reduce(Reduction op, Privatize<P> private_mapper, Read<Arg, M> access) {
    return {access.argument, op, access.access_mapper, private_mapper.access_mapper};
}

template<typename Arg, template<typename, typename> typename Mode, size_t N, size_t I = 0>
struct MultiIndexAccess {
    MultiIndexAccess(Arg& m_argument, MultiIndexMap<N> m_mapper = {}) :
        m_argument(m_argument),
        m_mapper(m_mapper) {}

    template<typename M>
    auto operator[](const M& index) {
        m_mapper.axes[I] = into_index_map(index);

        if constexpr (I + 1 == N) {
            return Mode<Arg, MultiIndexMap<N>> {m_argument, {m_mapper}};
        } else {
            return MultiIndexAccess<Arg, Mode, N, I + 1>(m_argument, m_mapper);
        }
    }

  private:
    Arg& m_argument;
    MultiIndexMap<N> m_mapper = {};
};

// Forward `Read<Arg, M>` to `Read<const Arg, M>` only if `Arg` is not const
template<typename Arg, typename M>
struct ArgumentHandler<Read<Arg, M>, std::enable_if_t<!std::is_const_v<Arg>>>:
    ArgumentHandler<Read<const Arg, M>> {
    ArgumentHandler(Read<Arg, M> access) :
        ArgumentHandler<Read<const Arg, M>>({access.argument, access.access_mapper}) {}
};

}  // namespace kmm