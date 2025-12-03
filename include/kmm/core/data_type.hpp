#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "fmt/ostream.h"

#include "kmm/utils/key_value.hpp"

namespace kmm {

enum struct ScalarType : uint8_t {
    Invalid = 0,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Float16,
    Float32,
    Float64,
    BFloat16,
    Complex16,
    Complex32,
    Complex64,
    KeyAndInt64,
    KeyAndFloat64,
};

template<typename T>
struct DataTypeOf;

struct DataType {
    DataType() = default;

    static DataType of(ScalarType kind);

    template<typename T>
    static DataType of() {
        static_assert(DataTypeOf<T>::value.size_in_bytes == sizeof(T));
        static_assert(DataTypeOf<T>::value.alignment >= alignof(T));
        return DataType {&DataTypeOf<T>::value};
    }

    const std::type_info& type_info() const;

    ScalarType as_scalar() const;

    size_t size_in_bytes() const;

    size_t alignment() const;

    const char* name() const;

    const char* c_name() const;

  public:
    struct Info {
        size_t size_in_bytes;
        size_t alignment;
        const char* name = nullptr;
        const char* c_name = nullptr;
        const std::type_info& type_id;
        ScalarType scalar_type = ScalarType::Invalid;
    };

  private:
    explicit DataType(const Info* info) : m_info(info) {}
    const Info* m_info = nullptr;
};

template<typename T>
struct DataTypeOf {
    static constexpr DataType::Info value =
        {.size_in_bytes = sizeof(T), .alignment = alignof(T), .type_id = typeid(T)};
};

std::ostream& operator<<(std::ostream& f, ScalarType p);
std::ostream& operator<<(std::ostream& f, DataType p);

}  // namespace kmm

#define KMM_DEFINE_SCALAR_TYPE(S, T)                     \
    template<>                                           \
    struct kmm::DataTypeOf<T> {                          \
        static constexpr ::kmm::DataType::Info value = { \
            .size_in_bytes = sizeof(T),                  \
            .alignment = alignof(T),                     \
            .name = #S,                                  \
            .c_name = #T,                                \
            .type_id = typeid(T),                        \
            .scalar_type = ::kmm::ScalarType::S          \
        };                                               \
    };

KMM_DEFINE_SCALAR_TYPE(Int8, int8_t)
KMM_DEFINE_SCALAR_TYPE(Int16, int16_t)
KMM_DEFINE_SCALAR_TYPE(Int32, int32_t)
KMM_DEFINE_SCALAR_TYPE(Int64, int64_t)
KMM_DEFINE_SCALAR_TYPE(Uint8, uint8_t)
KMM_DEFINE_SCALAR_TYPE(Uint16, uint16_t)
KMM_DEFINE_SCALAR_TYPE(Uint32, uint32_t)
KMM_DEFINE_SCALAR_TYPE(Uint64, uint64_t)
KMM_DEFINE_SCALAR_TYPE(Float32, float)
KMM_DEFINE_SCALAR_TYPE(Float64, double)
KMM_DEFINE_SCALAR_TYPE(Complex32, ::std::complex<float>)
KMM_DEFINE_SCALAR_TYPE(Complex64, ::std::complex<double>)
KMM_DEFINE_SCALAR_TYPE(KeyAndInt64, ::kmm::KeyValue<int64_t>)
KMM_DEFINE_SCALAR_TYPE(KeyAndFloat64, ::kmm::KeyValue<double>)

template<>
struct fmt::formatter<kmm::ScalarType>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::DataType>: fmt::ostream_formatter {};