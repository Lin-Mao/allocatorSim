#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <cxxabi.h>  // for __cxa_demangle
#include <sstream>

#include "utils/unwind_utils.h"

std::string get_backtrace() {
    unw_cursor_t cursor;
    unw_context_t context;

    // Initialize cursor to current frame for local unwinding.
    unw_getcontext(&context);
    unw_init_local(&cursor, &context);

    std::stringstream ss;

    // Unwind frames one by one, going up the frame stack.
    unw_step(&cursor);
    unw_step(&cursor);
    unw_step(&cursor); // skip the first three frames
    while (unw_step(&cursor) > 0) {
        unw_word_t offset, pc;
        unw_get_reg(&cursor, UNW_REG_IP, &pc);
        if (pc == 0) {
            break;
        }
        // std::printf("0x%lx:", pc);
        // ss << "0x" << std::hex << pc << ":" << std::dec;
        ss << "0x" << ":" << std::dec;

        char sym[256];
        if (unw_get_proc_name(&cursor, sym, sizeof(sym), &offset) == 0) {
            // std::printf(" (%s+0x%lx)\n", sym, offset);
            ss << " (" << sym << "+0x" << std::hex << offset << std::dec << ")\n";
        } else {
            // std::printf(" -- error: unable to obtain symbol name for this frame\n");
            ss << " -- error: unable to obtain symbol name for this frame\n";
        }
    }

    return ss.str();
}

std::string get_demangled_backtrace() {
    unw_cursor_t cursor;
    unw_context_t context;

    // Initialize cursor to current frame for local unwinding.
    unw_getcontext(&context);
    unw_init_local(&cursor, &context);

    std::stringstream ss;

    // Unwind frames one by one, going up the frame stack.
    unw_step(&cursor);
    unw_step(&cursor);
    unw_step(&cursor); // skip the first three frames
    while (unw_step(&cursor) > 0) {
        unw_word_t offset, pc;
        unw_get_reg(&cursor, UNW_REG_IP, &pc);
        if (pc == 0) {
            break;
        }
        // std::printf("0x%lx:", pc);
        ss << "0x" << std::hex << pc << ":" << std::dec;

        char sym[256];
        if (unw_get_proc_name(&cursor, sym, sizeof(sym), &offset) == 0) {
            char* nameptr = sym;
            int status;
            char* demangled = abi::__cxa_demangle(sym, nullptr, nullptr, &status);
            if (status == 0) {
                nameptr = demangled;
            }
            // std::printf(" (%s+0x%lx)\n", nameptr, offset);
            ss << " (" << nameptr << "+0x" << std::hex << offset << std::dec << ")\n";
            std::free(demangled);
        } else {
            // std::printf(" -- error: unable to obtain symbol name for this frame\n");
            ss << " -- error: unable to obtain symbol name for this frame\n";
        }
    }

    return ss.str();
}