#pragma once

#include "partition.hpp"
#include "spdlog/spdlog.h"

#include "kmm/api/argument.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/utils/geometry.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

struct All {
    template<size_t N>
    Bounds<N> operator()(WorkChunk chunk, Bounds<N> bounds) const {
        return bounds;
    }
};

struct Axis {
    constexpr Axis() : m_axis(0) {}
    explicit constexpr Axis(size_t axis) : m_axis(axis) {}

    Bounds<1> operator()(WorkChunk chunk) const {
        return Bounds<1>::from_offset_size(
            chunk.offset.get_or_default(m_axis),
            chunk.size.get_or_default(m_axis)
        );
    }

    Bounds<1> operator()(WorkChunk chunk, Bounds<1> bounds) const {
        return (*this)(chunk).intersection(bounds);
    }

    size_t get() const {
        return m_axis;
    }

    explicit operator size_t() const {
        return get();
    }

  private:
    size_t m_axis = 0;
};

struct IdentityMap {
    template<size_t N>
    Bounds<N> operator()(WorkChunk chunk, Bounds<N> bounds) const {
        return Bounds<N>::from_offset_size(Index<N>::from(chunk.offset), Dim<N>::from(chunk.size));
    }
};

// (scale * variable + offset + [0...length]) / divisor
struct IndexMap {
    constexpr IndexMap(Axis variable = {}) : m_variable(variable) {}

    IndexMap(
        Axis variable,
        int64_t scale,
        int64_t offset = 0,
        int64_t length = 1,
        int64_t divisor = 1
    );

    static IndexMap range(IndexMap begin, IndexMap end);
    IndexMap offset_by(int64_t offset) const;
    IndexMap scale_by(int64_t factor) const;
    IndexMap divide_by(int64_t divisor) const;
    IndexMap negate() const;
    Bounds<1> apply(WorkChunk chunk) const;

    Bounds<1> operator()(WorkChunk chunk) const {
        return apply(chunk);
    }

    Bounds<1> operator()(WorkChunk chunk, Bounds<1> bounds) const {
        return apply(chunk).intersection(bounds);
    }

    friend std::ostream& operator<<(std::ostream& f, const IndexMap& that);

  private:
    Axis m_variable = {};
    int64_t m_scale = 1;
    int64_t m_offset = 0;
    int64_t m_length = 1;
    int64_t m_divisor = 1;
};

inline IndexMap range(IndexMap begin, IndexMap end) {
    return IndexMap::range(begin, end);
}

inline IndexMap range(int64_t begin, int64_t end) {
    return {Axis {}, 0, begin, end - begin};
}

inline IndexMap range(int64_t end) {
    return range(0, end);
}

inline IndexMap operator+(IndexMap a) {
    return a;
}

inline IndexMap operator+(IndexMap a, int64_t b) {
    return a.offset_by(b);
}

inline IndexMap operator+(int64_t a, IndexMap b) {
    return b.offset_by(a);
}

inline IndexMap operator-(IndexMap a) {
    return a.negate();
}

inline IndexMap operator-(IndexMap a, int64_t b) {
    return a + (-b);
}

inline IndexMap operator-(int64_t a, IndexMap b) {
    return a + (-b);
}

inline IndexMap operator*(IndexMap a, int64_t b) {
    return a.scale_by(b);
}

inline IndexMap operator*(int64_t a, IndexMap b) {
    return b.scale_by(a);
}

inline IndexMap operator/(IndexMap a, int64_t b) {
    return a.divide_by(b);
}

template<size_t N>
struct MultiIndexMap {
    Bounds<N> operator()(WorkChunk chunk) const {
        Bounds<N> result;

        for (size_t i = 0; i < N; i++) {
            result[i] = (this->axes[i])(chunk);
        }

        return result;
    }

    Bounds<N> operator()(WorkChunk chunk, Bounds<N> bounds) const {
        Bounds<N> result;

        for (size_t i = 0; i < N; i++) {
            result[i] = (this->axes[i])(chunk, Bounds<1> {bounds[i]});
        }

        return result;
    }

    IndexMap axes[N];
};

template<>
struct MultiIndexMap<0> {
    Bounds<0> operator()(WorkChunk chunk, Bounds<0> bounds = {}) const {
        return {};
    }
};

inline IndexMap into_index_map(int64_t m) {
    return {Axis {}, 0, m};
}

inline IndexMap into_index_map(Axis m) {
    return m;
}

inline IndexMap into_index_map(IndexMap m) {
    return m;
}

inline IndexMap into_index_map(All m) {
    return {Axis(), 0, 0, std::numeric_limits<int64_t>::max()};
}

template<typename... Is>
MultiIndexMap<sizeof...(Is)> bounds(const Is&... slices) {
    return {into_index_map(slices)...};
}

template<typename... Is>
MultiIndexMap<sizeof...(Is)> tile(const Is&... length) {
    size_t variable = 0;
    return {IndexMap(
        Axis {variable++},
        checked_cast<int64_t>(length),
        0,
        checked_cast<int64_t>(length)
    )...};
}

namespace placeholders {
static constexpr All _;

static constexpr Axis _x = Axis(0);
static constexpr Axis _y = Axis(1);
static constexpr Axis _z = Axis(2);

static constexpr Axis _i = Axis(0);
static constexpr Axis _j = Axis(1);
static constexpr Axis _k = Axis(2);

static constexpr Axis _0 = Axis(0);
static constexpr Axis _1 = Axis(1);
static constexpr Axis _2 = Axis(2);

static constexpr MultiIndexMap<2> _xy = {_x, _y};
static constexpr MultiIndexMap<3> _xyz = {_x, _y, _z};

static constexpr MultiIndexMap<2> _ij = {_x, _y};
static constexpr MultiIndexMap<3> _ijk = {_x, _y, _z};

static constexpr IdentityMap one_to_one;
static constexpr All all;
}  // namespace placeholders

template<>
struct ArgumentHandler<IndexMap> {
    using type = Argument<Range<default_index_type>>;

    ArgumentHandler(IndexMap mapper) : m_mapper(mapper) {}

    void initialize(const TaskGroupInfo& init) {}

    type process_chunk(TaskInstance& builder) {
        return m_mapper(builder.chunk).get_or_default(0);
    }

    void finalize(const TaskGroupResult& result) {}

  private:
    IndexMap m_mapper;
};

template<>
struct ArgumentHandler<Axis>: ArgumentHandler<IndexMap> {
    ArgumentHandler(Axis mapper) : ArgumentHandler<IndexMap>(mapper) {}
};

template<size_t N>
struct ArgumentHandler<MultiIndexMap<N>> {
    using type = Argument<Bounds<N>>;

    ArgumentHandler(MultiIndexMap<N> mapper) : m_mapper(mapper) {}

    void initialize(const TaskGroupInfo& init) {}

    type process_chunk(TaskInstance& builder) {
        return m_mapper(builder.chunk);
    }

    void finalize(const TaskGroupResult& result) {}

  private:
    MultiIndexMap<N> m_mapper;
};

namespace detail {
template<typename T>
struct RangeDim: std::integral_constant<size_t, ~size_t(0)> {};

template<size_t N>
struct RangeDim<Bounds<N>>: std::integral_constant<size_t, N> {};
}  // namespace detail

template<typename F>
static constexpr size_t mapper_dimensionality =
    detail::RangeDim<std::invoke_result_t<F, WorkChunk>>::value;

template<typename F, size_t N>
static constexpr bool is_dimensionality_accepted_by_mapper =
    detail::RangeDim<std::invoke_result_t<F, WorkChunk, Bounds<N>>>::value == N;

}  // namespace kmm

template<>
struct fmt::formatter<kmm::IndexMap>: fmt::ostream_formatter {};