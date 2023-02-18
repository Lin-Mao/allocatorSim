#ifndef UNWIND_UTILS_H
#define UNWIND_UTILS_H
#include <string>

std::string get_backtrace();

std::string get_demangled_backtrace();

#endif  // UNWIND_UTILS_H