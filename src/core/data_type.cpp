#include <complex>

#include "fmt/format.h"

#include "kmm/core/data_type.hpp"
#include "kmm/utils/panic.hpp"

namespace kmm {

const char* scalar_name(ScalarType kind) {
    switch (kind) {
        case ScalarType::Int8:
            return "Int8";
        case ScalarType::Int16:
            return "Int16";
        case ScalarType::Int32:
            return "Int32";
        case ScalarType::Int64:
            return "Int64";
        case ScalarType::Uint8:
            return "Uint8";
        case ScalarType::Uint16:
            return "Uint16";
        case ScalarType::Uint32:
            return "Uint32";
        case ScalarType::Uint64:
            return "Uint64";
        case ScalarType::Float16:
            return "Float16";
        case ScalarType::Float32:
            return "Float32";
        case ScalarType::Float64:
            return "Float64";
        case ScalarType::BFloat16:
            return "BFloat16";
        case ScalarType::Complex16:
            return "Float64";
        case ScalarType::Complex32:
            return "Complex32";
        case ScalarType::Complex64:
            return "Complex64";
        case ScalarType::KeyAndInt64:
            return "KeyAndInt64";
        case ScalarType::KeyAndFloat64:
            return "KeyAndFloat64";
        default:
            return "(unknown type)";
    }
}

DataType DataType::of(ScalarType kind) {
    switch (kind) {
        case ScalarType::Int8:
            return DataType::of<int8_t>();
        case ScalarType::Int16:
            return DataType::of<int16_t>();
        case ScalarType::Int32:
            return DataType::of<int32_t>();
        case ScalarType::Int64:
            return DataType::of<int64_t>();
        case ScalarType::Uint8:
            return DataType::of<uint8_t>();
        case ScalarType::Uint16:
            return DataType::of<uint16_t>();
        case ScalarType::Uint32:
            return DataType::of<uint32_t>();
        case ScalarType::Uint64:
            return DataType::of<uint64_t>();
            //        case ScalarType::Float16:
            //            return DataType::of<uint64_t>();
        case ScalarType::Float32:
            return DataType::of<float>();
        case ScalarType::Float64:
            return DataType::of<double>();
            //        case ScalarType::BFloat16:
            //            break;
            //        case ScalarType::Complex16:
            //            break;
        case ScalarType::Complex32:
            return DataType::of<std::complex<float>>();
        case ScalarType::Complex64:
            return DataType::of<std::complex<double>>();
        case ScalarType::KeyAndInt64:
            return DataType::of<KeyValue<int64_t>>();
        case ScalarType::KeyAndFloat64:
            return DataType::of<KeyValue<double>>();
        case ScalarType::Invalid:
            return DataType();
    }

    throw std::runtime_error(fmt::format("unknown scalar type: {}", scalar_name(kind)));
}

const std::type_info& DataType::type_info() const {
    KMM_ASSERT(m_info != nullptr);
    return m_info->type_id;
}

ScalarType DataType::as_scalar() const {
    return m_info != nullptr ? m_info->scalar_type : ScalarType::Invalid;
}

size_t DataType::size_in_bytes() const {
    KMM_ASSERT(m_info != nullptr);
    return m_info->size_in_bytes;
}

size_t DataType::alignment() const {
    KMM_ASSERT(m_info != nullptr);
    return m_info->alignment;
}

const char* DataType::name() const {
    if (m_info == nullptr) {
        return "Invalid";
    } else if (const auto* name = m_info->name) {
        return name;
    } else {
        return m_info->type_id.name();
    }
}

const char* DataType::c_name() const {
    if (m_info != nullptr && m_info->c_name != nullptr) {
        return m_info->c_name;
    }

    return name();
}

std::ostream& operator<<(std::ostream& f, ScalarType p) {
    return f << scalar_name(p);
}

std::ostream& operator<<(std::ostream& f, DataType p) {
    return f << p.name();
}

}  // namespace kmm