#pragma once

#include "kmm/core/identifiers.hpp"
#include "kmm/dag/array_planner.hpp"
#include "kmm/dag/distribution.hpp"
#include "kmm/utils/geometry.hpp"

namespace kmm {

class Worker;

template<size_t N>
class ArrayHandle: public std::enable_shared_from_this<ArrayHandle<N>> {
    KMM_NOT_COPYABLE_OR_MOVABLE(ArrayHandle)

  public:
    ArrayHandle(Worker& worker, ArrayInstance<N> instance);
    ~ArrayHandle();

    static std::shared_ptr<ArrayHandle> instantiate(
        Worker& worker,
        Distribution<N> dist,
        DataLayout element_layout
    );

    BufferId buffer(size_t index) const;
    void copy_bytes(void* dest_addr, size_t element_size) const;
    void synchronize() const;

    const ArrayInstance<N>& instance() const {
        return m_instance;
    }

    const Distribution<N>& distribution() const {
        return m_instance.distribution();
    }

    const Worker& worker() const {
        return *m_worker;
    }

  private:
    std::shared_ptr<Worker> m_worker;
    ArrayInstance<N> m_instance;
};

}  // namespace kmm