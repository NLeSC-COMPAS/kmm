#pragma once

#include <ostream>

#define KMM_PANIC(...)                                 \
    do {                                               \
        ::kmm::panic(__FILE__, __LINE__, __VA_ARGS__); \
        while (1)                                      \
            ;                                          \
    } while (0);

#define KMM_ASSERT(...)                                   \
    do {                                                  \
        if (!static_cast<bool>(__VA_ARGS__)) {            \
            KMM_PANIC("assertion failed: " #__VA_ARGS__); \
        }                                                 \
    } while (0);

#define KMM_DEBUG_ASSERT(...) KMM_ASSERT(__VA_ARGS__)
#define KMM_TODO()            KMM_PANIC("not implemented")

#define KMM_PANIC_FMT(...)                                                    \
    do {                                                                      \
        ::kmm::panic(__FILE__, __LINE__, ::fmt::format(__VA_ARGS__).c_str()); \
        while (1)                                                             \
            ;                                                                 \
    } while (0);

namespace kmm {

/**
 *  Logs a fatal error, prints relevant debugging info, and aborts the program.
 *
 * @param file      Source filename where the panic occurred.
 * @param line      Line number where the panic occurred.
 * @param function  Function name where the panic occurred.
 * @param message   Reason for the panic.
 */
[[noreturn]] void panic(const char* filename, int lineno, const char* message);

}  // namespace kmm