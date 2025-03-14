#pragma once

#include "kmm/core/identifiers.hpp"
#include "kmm/dag/array_planner.hpp"
#include "kmm/dag/distribution.hpp"
#include "kmm/utils/geometry.hpp"

namespace kmm {

class Runtime;

template<size_t N>
class ArrayHandle: public std::enable_shared_from_this<ArrayHandle<N>> {
    KMM_NOT_COPYABLE_OR_MOVABLE(ArrayHandle)

  public:
    ArrayHandle(Runtime& rt, std::unique_ptr<ArrayInstance<N>> instance);
    ~ArrayHandle();

    static std::shared_ptr<ArrayHandle> instantiate(
        Runtime& rt,
        Distribution<N> dist,
        DataType dtype
    );

    BufferId buffer(size_t index) const;
    void copy_bytes(void* dest_addr, size_t element_size) const;
    void synchronize() const;

    const ArrayInstance<N>& instance() const {
        return *m_instance;
    }

    const Distribution<N>& distribution() const {
        return m_instance->distribution();
    }

    const Runtime& runtime() const {
        return *m_rt;
    }

  private:
    std::shared_ptr<Runtime> m_rt;
    std::unique_ptr<ArrayInstance<N>> m_instance;
};

}  // namespace kmm