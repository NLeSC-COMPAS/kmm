#pragma once

#include "kmm/planner/array_descriptor.hpp"

namespace kmm {

class Runtime;

template<size_t N>
class ArrayInstance:
    public ArrayDescriptor<N>,
    public std::enable_shared_from_this<ArrayInstance<N>> {
    KMM_NOT_COPYABLE_OR_MOVABLE(ArrayInstance)

    ArrayInstance(TaskGraph& stage, Runtime& rt, Distribution<N> dist, DataType dtype);

  public:
    static std::shared_ptr<ArrayInstance> create(Runtime& rt, Distribution<N> dist, DataType dtype);
    ~ArrayInstance();

    void copy_bytes_into(void* data);
    void copy_bytes_from(const void* data);
    void synchronize() const;

    Runtime& runtime() const {
        return *m_rt;
    }

  private:
    std::shared_ptr<Runtime> m_rt;
};

[[noreturn]] void throw_uninitialized_array_exception();

}  // namespace kmm