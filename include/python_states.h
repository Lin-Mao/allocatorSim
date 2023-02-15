#ifndef PYTHON_STATES_H
#define PYTHON_STATES_H

#include <Python.h>
#include <pybind11/pybind11.h>

#include <string>
#include <vector>

namespace c10 {
namespace cuda {
namespace AllocatorSim {

typedef struct python_state {
    const char *file_name;
    const char *function_name;
    size_t function_first_lineno;
    size_t lineno;
}python_state_t;

struct PythonState {
    std::string file_name;
    std::string function_name;
    size_t function_first_lineno;
    size_t lineno;

    PythonState(const std::string &file_name, const std::string &function_name,
                size_t function_first_lineno, size_t lineno)
            : file_name(file_name),
            function_name(function_name),
            function_first_lineno(function_first_lineno),
            lineno(lineno) {}
};

class PythonStateMonitor {
public:
    // Return the current python states with a query or using the previous cached states
    std::vector<PythonState> &get_states(bool cached = false);

    // Get the singleton instance
    static PythonStateMonitor &instance();

private:
    PythonStateMonitor() {}

    std::string unpack_pyobject(PyObject *obj);

private:
    // Cached states for each thread
    static inline thread_local std::vector<PythonState> _states;
};

bool python_state_get(size_t max_num_states, python_state_t *states, size_t *num_states);

}  // namespace c10
}  // namespace cuda
}  // namespace AllocatorSim

#endif  // PYTHON_STATES_H