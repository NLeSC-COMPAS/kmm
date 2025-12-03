#pragma once

#include "macros.hpp"

#define KMM_PANIC(...)                                 \
    do {                                               \
        ::kmm::panic(__FILE__, __LINE__, __VA_ARGS__); \
        while (1)                                      \
            ;                                          \
    } while (0)

#define KMM_ASSERT(...)                                      \
    do {                                                     \
        if (KMM_UNLIKELY(!static_cast<bool>(__VA_ARGS__))) { \
            KMM_PANIC("assertion failed: " #__VA_ARGS__);    \
        }                                                    \
    } while (0)

#define KMM_DEBUG_ASSERT(...) KMM_ASSERT(__VA_ARGS__)
#define KMM_TODO()            KMM_PANIC("not implemented")

#if !KMM_IS_RTC
    #include "fmt/format.h"

    #define KMM_PANIC_FMT(...)                                                    \
        do {                                                                      \
            ::kmm::panic(__FILE__, __LINE__, ::fmt::format(__VA_ARGS__).c_str()); \
            while (1)                                                             \
                ;                                                                 \
        } while (0)
#endif

namespace kmm {

#if !KMM_IS_DEVICE
/**
 *  Logs a fatal error, prints relevant debugging info, and aborts the program.
 *
 * @param file      Source filename where the panic occurred.
 * @param line      Line number where the panic occurred.
 * @param function  Function name where the panic occurred.
 * @param message   Reason for the panic.
 */
[[noreturn]] void panic(const char* filename, int lineno, const char* message);
#else
KMM_DEVICE void panic(const char* filename, int lineno, const char* message) {
    printf(
        "[block=(%u,%u,%u) thread=(%u,%u,%u)] PANIC at %s:%d: %s\n",
        blockIdx.x,
        blockIdx.y,
        blockIdx.z,
        threadIdx.x,
        threadIdx.y,
        threadIdx.z,
        filename,
        lineno,
        message
    );

    while (true) {
        asm volatile("trap;");
    }
}
#endif

}  // namespace kmm