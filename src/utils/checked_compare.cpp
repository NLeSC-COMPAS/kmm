#include <stdexcept>

#include "kmm/utils/checked_compare.hpp"

namespace kmm {

void throw_overflow_exception() {
    throw std::overflow_error("integer overflow occurred");
}

}  // namespace kmm