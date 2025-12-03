#include <stdexcept>

#include "kmm/utils/small_vector.hpp"

namespace kmm {
[[noreturn]] __attribute__((noinline)) void throw_small_vector_out_of_capacity() {
    throw std::overflow_error("small_vector exceeds capacity");
}
}  // namespace kmm