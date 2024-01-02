#include "kmm/memory.hpp"
#include "kmm/utils.hpp"

namespace kmm {

MemoryCompletion::MemoryCompletion(std::shared_ptr<IMemoryCompletion> c) :
    m_completion(std::move(c)) {
    KMM_ASSERT(m_completion != nullptr);
}

MemoryCompletion::~MemoryCompletion() {
    if (m_completion) {
        KMM_PANIC("deleted `MemoryCompletion` without calling `complete()`");
    }
}

void MemoryCompletion::complete(Result<void> result) {
    KMM_ASSERT(m_completion != nullptr);
    auto completion = std::exchange(m_completion, nullptr);
    completion->complete(std::move(result));
}

}  // namespace kmm