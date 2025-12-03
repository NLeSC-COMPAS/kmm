#pragma once

#include <iosfwd>
#include <optional>
#include <utility>

#include "fmt/ostream.h"

#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/macros.hpp"
#include "kmm/utils/panic.hpp"
#include "kmm/utils/small_vector.hpp"

#define KMM_IMPL_COMPARISON_OPS(T)                              \
    KMM_INLINE constexpr bool operator!=(const T& that) const { \
        return !(*this == that);                                \
    }                                                           \
    KMM_INLINE constexpr bool operator<=(const T& that) const { \
        return !(*this > that);                                 \
    }                                                           \
    KMM_INLINE constexpr bool operator>(const T& that) const {  \
        return that < *this;                                    \
    }                                                           \
    KMM_INLINE constexpr bool operator>=(const T& that) const { \
        return that <= *this;                                   \
    }

namespace kmm {

using index_t = int;

struct NodeId {
    explicit constexpr NodeId(uint8_t v) : m_value(v) {}
    explicit constexpr NodeId(size_t v) : m_value(checked_cast<uint8_t>(v)) {}

    KMM_INLINE constexpr uint8_t get() const {
        return m_value;
    }

    KMM_INLINE operator uint8_t() const {
        return get();
    }

    KMM_INLINE constexpr bool operator==(const NodeId& that) const {
        return m_value == that.m_value;
    }

    KMM_INLINE constexpr bool operator<(const NodeId& that) const {
        return m_value < that.m_value;
    }

    KMM_IMPL_COMPARISON_OPS(NodeId)

    friend std::ostream& operator<<(std::ostream&, const NodeId&);

  private:
    uint8_t m_value;
};

// Maximum of 8 devices per node
static constexpr size_t MAX_DEVICES = 8;

struct DeviceId {
    template<typename T>
    explicit constexpr DeviceId(T v) : m_value(static_cast<uint8_t>(v)) {
        if (!in_range(v, MAX_DEVICES)) {
            throw std::runtime_error("device index out of range");
        }
    }

    KMM_INLINE constexpr uint8_t get() const {
        if (m_value >= MAX_DEVICES) {
            __builtin_unreachable();
        }

        return m_value;
    }

    KMM_INLINE operator uint8_t() const {
        return get();
    }

    KMM_INLINE constexpr bool operator==(const DeviceId& that) const {
        return m_value == that.m_value;
    }

    KMM_INLINE constexpr bool operator<(const DeviceId& that) const {
        return m_value < that.m_value;
    }

    KMM_IMPL_COMPARISON_OPS(DeviceId)

    friend std::ostream& operator<<(std::ostream&, const DeviceId&);

  private:
    uint8_t m_value;
};

struct DeviceStreamSet {
    static constexpr size_t MAX_SIZE = 64;
    using value_type = uint64_t;

    constexpr DeviceStreamSet(size_t index) {
        if (index < MAX_SIZE) {
            this->m_value |= value_type(1) << value_type(index);
        }
    }

    constexpr DeviceStreamSet() = default;

    constexpr DeviceStreamSet(std::initializer_list<size_t> indices) {
        for (auto index : indices) {
            this->m_value |= DeviceStreamSet(index).m_value;
        }
    }

    static constexpr DeviceStreamSet range(size_t begin, size_t end) {
        auto result = DeviceStreamSet {};

        for (size_t index = begin; index < end; index++) {
            result.m_value |= DeviceStreamSet(index).m_value;
        }

        return result;
    }

    static constexpr DeviceStreamSet all() {
        return range(0, MAX_SIZE);
    }

    constexpr bool is_empty() const {
        return m_value == 0;
    }

    constexpr bool contains(DeviceStreamSet that) const {
        return (this->m_value & that.m_value) == that.m_value;
    }

    constexpr bool contains(size_t index) const {
        return contains(DeviceStreamSet {index});
    }

    constexpr DeviceStreamSet& operator&=(const DeviceStreamSet& that) {
        this->m_value &= that.m_value;
        return *this;
    }

    constexpr DeviceStreamSet operator&(const DeviceStreamSet& that) const {
        return DeviceStreamSet {*this} &= that;
    }

    constexpr bool operator==(const DeviceStreamSet& that) const {
        return m_value == that.m_value;
    }

    constexpr bool operator!=(const DeviceStreamSet& that) const {
        return !(*this == that);
    }

  private:
    value_type m_value = 0;
};

struct MemoryId {
  public:
    enum struct Type : uint8_t { Host, Device };

    KMM_INLINE constexpr MemoryId(Type kind, uint8_t v) : m_type(kind), m_value(v) {}

    KMM_INLINE constexpr MemoryId(DeviceId device) : MemoryId(Type::Device, device.get()) {}

    KMM_INLINE static constexpr MemoryId host(DeviceId device_affinity = DeviceId(0)) {
        return MemoryId {Type::Host, device_affinity.get()};
    }

    KMM_INLINE constexpr bool is_host() const noexcept {
        return m_type == Type::Host;
    }

    KMM_INLINE constexpr bool is_device() const noexcept {
        return m_type == Type::Device;
    }

    KMM_INLINE constexpr DeviceId as_device() const {
        KMM_ASSERT(is_device());
        return DeviceId(m_value);
    }

    KMM_INLINE constexpr DeviceId device_affinity() const {
        return DeviceId(m_value % MAX_DEVICES);
    }

    KMM_INLINE constexpr bool operator==(const MemoryId& that) const {
        if (m_type == Type::Device && m_type == Type::Device) {
            return m_value == that.m_value;
        } else {
            return m_type == that.m_type;
        }
    }

    KMM_INLINE constexpr bool operator<(const MemoryId& that) const {
        if (m_type == Type::Device && m_type == Type::Device) {
            return m_value < that.m_value;
        } else {
            return m_type < that.m_type;
        }
    }

    KMM_IMPL_COMPARISON_OPS(MemoryId)

    friend std::ostream& operator<<(std::ostream&, const MemoryId&);

  private:
    Type m_type;
    uint8_t m_value;
};

class ResourceId {
  public:
    enum struct Type : uint8_t { Host, Device };

    KMM_INLINE constexpr ResourceId() : m_type(Type::Host) {}

    KMM_INLINE constexpr ResourceId(
        DeviceId device,
        DeviceStreamSet stream_affinity = DeviceStreamSet::all()
    ) :
        m_type(Type::Device),
        m_device(device),
        m_stream_affinity(stream_affinity) {}

    KMM_INLINE static constexpr ResourceId host(DeviceId affinity = DeviceId(0)) {
        ResourceId result;
        result.m_device = affinity;
        return result;
    }

    KMM_INLINE constexpr bool is_host() const {
        return m_type == Type::Host;
    }

    KMM_INLINE constexpr bool is_device() const {
        return m_type == Type::Device;
    }

    KMM_INLINE constexpr DeviceId as_device() const {
        KMM_ASSERT(is_device());
        return m_device;
    }

    KMM_INLINE constexpr DeviceId device_affinity() const {
        return m_device;
    }

    KMM_INLINE constexpr DeviceStreamSet stream_affinity() const {
        if (m_type == Type::Device && m_stream_affinity != DeviceStreamSet {}) {
            return m_stream_affinity;
        } else {
            return DeviceStreamSet::all();
        }
    }

    KMM_INLINE constexpr MemoryId as_memory() const {
        return is_host() ? MemoryId::host(m_device) : MemoryId(m_device);
    }

    KMM_INLINE constexpr bool contains(const ResourceId& that) const {
        if (m_type == Type::Host && that.m_type == Type::Host) {
            return true;
        } else if (m_type == Type::Device && that.m_type == Type::Device) {
            return m_device == that.m_device && m_stream_affinity.contains(that.m_stream_affinity);
        } else {
            return false;
        }
    }

    friend std::ostream& operator<<(std::ostream&, const ResourceId&);

  private:
    Type m_type = Type::Host;
    DeviceId m_device = DeviceId(0);
    DeviceStreamSet m_stream_affinity;  // only for Type::Device
};

struct BufferId {
    KMM_INLINE explicit constexpr BufferId(uint64_t v = ~uint64_t(0)) : m_value(v) {}

    KMM_INLINE constexpr uint64_t get() const {
        return m_value;
    }

    KMM_INLINE operator uint64_t() const {
        return get();
    }

    KMM_INLINE constexpr bool operator==(const BufferId& that) const {
        return m_value == that.m_value;
    }

    KMM_INLINE constexpr bool operator<(const BufferId& that) const {
        return m_value < that.m_value;
    }

    KMM_IMPL_COMPARISON_OPS(BufferId)

    friend std::ostream& operator<<(std::ostream&, const BufferId&);

  private:
    uint64_t m_value;
};

struct EventId {
    KMM_INLINE constexpr EventId() = default;

    KMM_INLINE explicit constexpr EventId(uint64_t v) : m_value(v) {}

    KMM_INLINE constexpr uint64_t get() const {
        return m_value;
    }

    KMM_INLINE operator uint64_t() const {
        return get();
    }

    KMM_INLINE constexpr bool operator==(const EventId& that) const {
        return m_value == that.m_value;
    }

    KMM_INLINE constexpr bool operator<(const EventId& that) const {
        return m_value < that.m_value;
    }

    KMM_IMPL_COMPARISON_OPS(EventId)

    friend std::ostream& operator<<(std::ostream&, const EventId&);

  private:
    uint64_t m_value = 0;
};

using EventList = small_vector<EventId, 2>;
std::ostream& operator<<(std::ostream&, const EventList&);

}  // namespace kmm

template<>
struct std::hash<kmm::NodeId>: std::hash<uint8_t> {};
template<>
struct std::hash<kmm::DeviceId>: std::hash<uint8_t> {};
template<>
struct std::hash<kmm::BufferId>: std::hash<uint64_t> {};
template<>
struct std::hash<kmm::EventId>: std::hash<uint64_t> {};

template<>
struct fmt::formatter<kmm::NodeId>: fmt::formatter<uint64_t> {};
template<>
struct fmt::formatter<kmm::DeviceId>: fmt::formatter<uint64_t> {};
template<>
struct fmt::formatter<kmm::BufferId>: fmt::formatter<uint64_t> {};
template<>
struct fmt::formatter<kmm::EventId>: fmt::formatter<uint64_t> {};
template<>
struct fmt::formatter<kmm::MemoryId>: fmt::ostream_formatter {};
template<>
struct fmt::formatter<kmm::ResourceId>: fmt::ostream_formatter {};
template<>
struct fmt::formatter<kmm::EventList>: fmt::ostream_formatter {};
