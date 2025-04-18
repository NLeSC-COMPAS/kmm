#include <stdio.h>
#include <stdlib.h>

// For POSIX stack trace (Linux, macOS, etc.)
// Remove if not available on your platform
#ifdef __GNUC__
    #include <execinfo.h>
#endif

namespace kmm {

void panic(const char* file, int line, const char* message) {
    fprintf(stderr, "\nPANIC TRIGGERED\n");
    fprintf(stderr, "  location:  %s:%d\n", file, line);
    fprintf(stderr, "  message:   %s\n", message);

#ifdef __GNUC__
    // Attempt to capture and print a backtrace
    void* callstack[128];
    int nframes = backtrace(callstack, 128);
    char** symbols = backtrace_symbols(callstack, nframes);

    if (symbols != NULL) {
        fprintf(stderr, "  stack trace:\n");
        for (int i = 0; i < nframes; i++) {
            fprintf(stderr, "    %s\n", symbols[i]);
        }
    } else {
        fprintf(stderr, "  stack trace:\n");
        fprintf(stderr, "    ??? <backtrace_symbols() failed>\n");
    }
#else
    fprintf(stderr, "  stack trace:\n");
    fprintf(stderr, "    ??? <backtrace_symbols() not supported>\n");
#endif

    fflush(stderr);

    // Use abort() to generate a core dump
    abort();
}

}  // namespace kmm