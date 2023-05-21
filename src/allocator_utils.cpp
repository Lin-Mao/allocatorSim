#include "allocator_utils.h"
#include <iostream>

namespace c10 {
namespace cuda {
namespace AllocatorSim {


/******************************************************************************/
/****************************** Common Variables ******************************/
/******************************************************************************/
namespace {
// the op_id is used in the whole code, include simulator and allocator
op_id_t global_op_id = 0;

}  // anonymous namespace for variables


/******************************************************************************/
/****************************** Common Functions ******************************/
/******************************************************************************/
op_id_t get_global_op_id() {
    return global_op_id;
}

void increase_global_op_id() {
    global_op_id++;
}

std::string format_size(size_t size) {
    std::ostringstream os;
    os.precision(2);
    os << std::fixed;
    if (size <= 1024) {
        os << size << " bytes";
    } else if (size <= 1048576) {
        os << (size / 1024.0);
        os << " KB";
    } else if (size <= 1073741824ULL) {
        os << size / 1048576.0;
        os << " MB";
    } else {
        os << size / 1073741824.0;
        os << " GB";
    }
    return os.str();
}


/******************************************************************************/
/****************************** Allocator Timer *******************************/
/******************************************************************************/
std::array<size_t, TIMER_NUMS> allocatorTimer::timers = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
std::array<std::string, TIMER_NUMS> allocatorTimer::timer_names = {"", "", "", "", "", "", "", "", "", ""};
std::array<sys_clock, TIMER_NUMS> allocatorTimer::starts = {};
std::array<sys_clock, TIMER_NUMS> allocatorTimer::ends = {};

void allocatorTimer::start_timer(int index) {
    starts[index] = std::chrono::system_clock::now();
}

void allocatorTimer::stop_timer(int index) {
    ends[index] = std::chrono::system_clock::now();
}

void allocatorTimer::log_timer(int index, std::string name) {
    if (timer_names[index] == "") {
        timer_names[index] = name;
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(ends[index] - starts[index]);
    timers[index] += double(duration.count()) * std::chrono::microseconds::period::num;
}

void allocatorTimer::print_timer(int index) {
    std::cout << std::string(timer_names[index]) << ": " << timers[index] << " us" << std::endl;
}


/******************************************************************************/
/************************** SimulatorModeController ***************************/
/******************************************************************************/
namespace sim_control{

bool disable_controller_init = false;

void print_sim_mode_controller(SimControlMode_t mode, bool enable) {
    std::string mode_name;
    switch (mode)
    {
    case ASYNC_TRACING:
        mode_name = "ASYNC_TRACING";
        break;
    case FUNCTIONALITY_CHECKING:
        mode_name = "FUNCTIONALITY_CHECKING";
        break;
    case PROFILING:
        mode_name = "PROFILING";
        break;
    case DEBUG_DUMPPING:
        mode_name = "DEBUG_DUMPPING";
        break;
    case DEBUG_POOLINFO_DUMPPING:
        mode_name = "DEBUG_POOLINFO_DUMPPING";
        break;
    case TRACE_DUMPPING:
        mode_name = "TRACE_DUMPPING";
        break;
    case CONFIG_OPTIMIZATION:
        mode_name = "CONFIG_OPTIMIZATION";
        break;
    case GROUP_OPTIMIZATION:
        mode_name = "GROUP_OPTIMIZATION";
        break;
    default:
        mode_name = "Unknown mode";
        break;
    }

    std::cout << "SimulatorModeController: " << mode_name << " is " << (enable ? "enabled" : "disabled") << std::endl;
}

void set_sim_control_mode(SimControlMode_t mode, bool enable) {
    if (disable_controller_init) {
        SimulatorModeController::init();
        disable_controller_init = true;
    }

    switch (mode)
    {
    case ASYNC_TRACING:
        SimulatorModeController::set_async_tracing(enable);
        break;
    case FUNCTIONALITY_CHECKING:
        SimulatorModeController::set_functionality_checking(enable);
        break;
    case PROFILING:
        SimulatorModeController::set_profiling(enable);
        break;
    case DEBUG_DUMPPING:
        SimulatorModeController::set_debug_dumpping(enable);
        break;
    case DEBUG_POOLINFO_DUMPPING:
        SimulatorModeController::set_debug_poolinfo_dumpping(enable);
        break;
    case TRACE_DUMPPING:    
        SimulatorModeController::set_trace_dumpping(enable);
        break;
    case CONFIG_OPTIMIZATION:
        SimulatorModeController::set_config_optimization(enable);
        break;
    case GROUP_OPTIMIZATION:
        SimulatorModeController::set_group_optimization(enable);
        break;
    default:
        std::cout << "SimulatorModeController: Unknown mode" << std::endl;
        break;
    }

    print_sim_mode_controller(mode, enable);
}

void SimulatorModeController::init() {
    if (disable_controller_init) {
        return;
    }
    enable_async_tracing = true;
    enable_functionality_checking = true;
    enable_profiling = true;
    enable_debug_dumpping = false;
    enable_debug_poolinfo_dumpping = false;
    enable_trace_dumpping = false;
    enable_config_optimization = true;
    enable_group_optimization = false;
}

bool SimulatorModeController::enable_async_tracing = true;
bool SimulatorModeController::is_async_tracing() {
    return enable_async_tracing;
}
void SimulatorModeController::set_async_tracing(bool async) {
    enable_async_tracing = async;
}

bool SimulatorModeController::enable_functionality_checking = false;
bool SimulatorModeController::is_functionality_checking() {
    return enable_functionality_checking;
}
void SimulatorModeController::set_functionality_checking(bool checking) {
    enable_functionality_checking = checking;
}

bool SimulatorModeController::enable_profiling = true;
bool SimulatorModeController::is_profiling() {
    return enable_profiling;
}
void SimulatorModeController::set_profiling(bool profiling) {
    enable_profiling = profiling;
}

// Use macro DUMP_INFO_TO_FILE_DEBUGGING for now
bool SimulatorModeController::enable_debug_dumpping = false;
bool SimulatorModeController::is_debug_dumpping() {
    return enable_debug_dumpping;
}
void SimulatorModeController::set_debug_dumpping(bool dumpping) {
    enable_debug_dumpping = dumpping;
}

bool SimulatorModeController::enable_debug_poolinfo_dumpping = false;
bool SimulatorModeController::is_debug_poolinfo_dumpping() {
    return enable_debug_poolinfo_dumpping;
}
void SimulatorModeController::set_debug_poolinfo_dumpping(bool dumpping) {
    enable_debug_poolinfo_dumpping = dumpping;
}

bool SimulatorModeController::enable_trace_dumpping = false;
bool SimulatorModeController::is_trace_dumpping() {
    return enable_trace_dumpping;
}
void SimulatorModeController::set_trace_dumpping(bool dumpping) {
    enable_trace_dumpping = dumpping;
}

bool SimulatorModeController::enable_config_optimization = true;
bool SimulatorModeController::is_config_optimization() {
    return enable_config_optimization;
}
void SimulatorModeController::set_config_optimization(bool optimization) {
    enable_config_optimization = optimization;
}

bool SimulatorModeController::enable_group_optimization = false;
bool SimulatorModeController::is_group_optimization() {
    return enable_group_optimization;
}
void SimulatorModeController::set_group_optimization(bool optimization) {
    enable_group_optimization = optimization;
}

}  // namespace sim_control

}  // namespace AllocatorSim
}  // namespace cuda
}  // namespace c10
