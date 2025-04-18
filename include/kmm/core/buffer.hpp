#pragma once

#include <iosfwd>
#include <memory>
#include <vector>

#include "kmm/core/data_type.hpp"
#include "kmm/core/identifiers.hpp"

namespace kmm {

/**
 * Represents the layout of a buffer. For now, this is just its size and alignment.
 */
struct BufferLayout {
    BufferLayout repeat(size_t n) {
        size_t remainder = size_in_bytes % alignment;
        size_t padding = remainder != 0 ? alignment - remainder : 0;
        return {(size_in_bytes + padding) * n, alignment};
    }

    template<typename T>
    static BufferLayout for_type(size_t n = 1) {
        return BufferLayout {sizeof(T), alignof(T)}.repeat(n);
    }

    static BufferLayout for_type(DataType dtype, size_t n = 1) {
        return BufferLayout {dtype.size_in_bytes(), dtype.alignment()}.repeat(n);
    }

    size_t size_in_bytes = 0;
    size_t alignment = 1;
};

/**
 * This enum is used to specify how a buffer can be accessed: read-only, read-write, or exclusive.
 */
enum struct AccessMode {
    Read,  ///< Read-only access to the buffer.
    ReadWrite,  ///< Read and write access to the buffer.
    Exclusive  ///< Exclusive access, implying full control over the buffer.
};

/**
 *  Represents the requirements for accessing a buffer.
 */
struct BufferRequirement {
    BufferId buffer_id;
    MemoryId memory_id;
    AccessMode access_mode;
};

/**
 * Provides access to a buffer with specific properties.
 */
struct BufferAccessor {
    MemoryId memory_id;
    BufferLayout layout;
    bool is_writable;
    void* address;
};

inline std::ostream& operator<<(std::ostream& f, AccessMode mode) {
    switch (mode) {
        case AccessMode::Read:
            return f << "Read";
        case AccessMode::ReadWrite:
            return f << "ReadWrite";
        case AccessMode::Exclusive:
            return f << "Exclusive";
    }

    return f;
}

}  // namespace kmm

template<>
struct fmt::formatter<kmm::AccessMode>: fmt::ostream_formatter {};