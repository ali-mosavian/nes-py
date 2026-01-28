//  Program:      nes-py
//  File:         lib_nes_env.cpp
//  Description:  file describes the outward facing ctypes API for Python
//
//  CHANGELOG:    - 2024-12-28: Changed from ctypes to pybind11 - Ali Mosavian
//                - 2026-01-27: Added VectorEmulator for parallel stepping
//
//  Copyright (c) 2019 Christian Kauten. All rights reserved.
//
#include "common.hpp"
#include "emulator.hpp"

#include <string>
#include <vector>
#include <thread>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>

// Platform-specific headers for CPU affinity
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <mach/thread_policy.h>
#include <mach/thread_act.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <chrono>

namespace py = pybind11;

// =============================================================================
// RAM Read Specification - for batch reading after step
// =============================================================================

enum class RamReadType {
    INT = 0,   // Single byte as integer
    BCD = 1    // Multiple bytes as BCD (Binary Coded Decimal)
};

struct RamReadSpec {
    uint16_t address;
    uint8_t size;        // 1-6 bytes
    RamReadType type;
    
    RamReadSpec(uint16_t addr, uint8_t sz, RamReadType t) 
        : address(addr), size(sz), type(t) {}
};

// =============================================================================
// CPU Affinity - Pin threads to specific cores for better cache locality
// =============================================================================

/// Pin the current thread to a specific CPU core.
/// On Linux: Uses pthread_setaffinity_np for hard affinity.
/// On macOS: Uses thread_policy_set (hint only, not guaranteed).
/// On Windows: Uses SetThreadAffinityMask.
inline void pin_thread_to_core(int core_id) {
    int num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0 || core_id < 0) return;
    
    // Wrap around if core_id exceeds available cores
    core_id = core_id % num_cores;
    
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#elif defined(__APPLE__)
    // macOS: thread affinity is only a hint, not enforced
    thread_affinity_policy_data_t policy = { core_id };
    thread_policy_set(pthread_mach_thread_np(pthread_self()),
                      THREAD_AFFINITY_POLICY,
                      (thread_policy_t)&policy,
                      THREAD_AFFINITY_POLICY_COUNT);
#elif defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), 1ULL << core_id);
#endif
}

// =============================================================================
// VectorEmulator - Parallel NES emulation with zero-copy observations
// =============================================================================
//
// Mirrors the NESEmulator interface but for multiple environments:
//   - step(actions)                 - Step all emulators 1 frame
//   - step_frames(actions, n)       - Step all emulators n frames  
//   - screen_buffer()               - Get list of zero-copy screen views
//   - screen_buffer(idx)            - Get single screen view
//   - memory_buffer()               - Get list of RAM views
//   - memory_buffer(idx)            - Get single RAM view
//
// Thread model: Persistent worker threads (one per emulator), condition
// variable synchronization. GIL is released during parallel stepping.
//
// =============================================================================

class VectorEmulator {
private:
    // Worker states - each worker only touches its own state (cache-line isolated)
    static constexpr int STATE_IDLE = 0;
    static constexpr int STATE_PENDING = 1;
    static constexpr int STATE_DONE = 2;
    static constexpr int STATE_EXIT = 3;
    
    // Cache line size for padding (typically 64 bytes on x86/ARM)
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
    // Cache-line aligned atomic to prevent false sharing
    struct alignas(CACHE_LINE_SIZE) AlignedAtomic {
        std::atomic<int> state{STATE_IDLE};
        char padding[CACHE_LINE_SIZE - sizeof(std::atomic<int>)];
        
        AlignedAtomic() : state(STATE_IDLE) {}
    };

public:
    VectorEmulator(const std::string& rom_path, int num_envs) 
        : num_envs_(num_envs), rom_path_(rom_path), ready_count_(0) {
        
        emulators_.reserve(num_envs);
        worker_states_.reserve(num_envs);
        worker_frames_.resize(num_envs, 1);
        worker_timings_.resize(num_envs);
        
        for (int i = 0; i < num_envs; i++) {
            emulators_.push_back(std::make_unique<NES::Emulator>(rom_path));
            worker_states_.push_back(std::make_unique<AlignedAtomic>());
        }
        
        workers_.reserve(num_envs);
        for (int i = 0; i < num_envs; i++) {
            workers_.emplace_back(&VectorEmulator::worker_loop, this, i);
        }
        
        // Wait for all workers to be ready (busy-wait on atomic counter)
        while (ready_count_.load(std::memory_order_acquire) < num_envs) {
            std::this_thread::yield();
        }
    }
    
    ~VectorEmulator() {
        // Signal all workers to exit
        for (int i = 0; i < num_envs_; i++) {
            worker_states_[i]->state.store(STATE_EXIT, std::memory_order_release);
        }
        
        for (auto& w : workers_) {
            if (w.joinable()) {
                w.join();
            }
        }
    }
    
    int num_envs() const { return num_envs_; }
    
    // Reset all emulators
    void reset() {
        for (auto& emu : emulators_) {
            emu->reset();
        }
    }
    
    // Reset single emulator
    void reset_env(int idx) {
        check_idx(idx);
        while (worker_states_[idx]->state.load(std::memory_order_acquire) != STATE_IDLE) {
            std::this_thread::yield();
        }
        emulators_[idx]->reset();
    }
    
    // Step all emulators 1 frame in parallel (like NESEmulator.step())
    void step(py::array_t<uint8_t> actions) {
        step_impl(actions, 1);
    }
    
    // Step a single emulator (synchronous, no threading)
    void step_single(int idx, uint8_t action) {
        check_idx(idx);
        // Wait for any pending work on this emulator
        while (worker_states_[idx]->state.load(std::memory_order_acquire) != STATE_IDLE) {
            std::this_thread::yield();
        }
        // Set action and step directly (no threading)
        *emulators_[idx]->get_controller(0) = action;
        emulators_[idx]->step();
    }
    
    // Get screen buffer for all emulators as list of zero-copy views
    py::list screen_buffer() {
        py::list result;
        for (int i = 0; i < num_envs_; i++) {
            result.append(screen_buffer_single(i));
        }
        return result;
    }
    
    // Get screen buffer for single emulator as zero-copy view
    py::array_t<uint8_t> screen_buffer_single(int idx) {
        check_idx(idx);
        const int HEIGHT = NES::Emulator::HEIGHT;
        const int WIDTH = NES::Emulator::WIDTH;
        
        #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            return py::array_t<uint8_t>(
                {HEIGHT, WIDTH, 3},
                {WIDTH * 4, 4, -1},  // negative stride: BGR -> RGB
                reinterpret_cast<uint8_t*>(emulators_[idx]->get_screen_buffer()) + 2,
                py::capsule(emulators_[idx].get(), [](void*) {})
            );
        #else
            return py::array_t<uint8_t>(
                {HEIGHT, WIDTH, 3},
                {WIDTH * 4, 4, 1},
                reinterpret_cast<uint8_t*>(emulators_[idx]->get_screen_buffer()) + 1,
                py::capsule(emulators_[idx].get(), [](void*) {})
            );
        #endif
    }
    
    // Get memory buffer for all emulators as list of zero-copy views
    py::list memory_buffer() {
        py::list result;
        for (int i = 0; i < num_envs_; i++) {
            result.append(memory_buffer_single(i));
        }
        return result;
    }
    
    // Get memory buffer for single emulator as zero-copy view
    py::array_t<uint8_t> memory_buffer_single(int idx) {
        check_idx(idx);
        return py::array_t<uint8_t>(
            {0x800},
            {1},
            reinterpret_cast<uint8_t*>(emulators_[idx]->get_memory_buffer()),
            py::capsule(emulators_[idx].get(), [](void*) {})
        );
    }
    
    // Get controller buffer for single emulator as zero-copy view
    py::array_t<uint8_t> controller(int idx, int port = 0) {
        check_idx(idx);
        return py::array_t<uint8_t>(
            {1},
            {1},
            reinterpret_cast<uint8_t*>(emulators_[idx]->get_controller(port)),
            py::capsule(emulators_[idx].get(), [](void*) {})
        );
    }
    
    // Dump state for single emulator
    py::array_t<uint8_t> dump_state(int idx) {
        check_idx(idx);
        
        // Create a copy of the state data
        auto* core = new NES::Core;
        memset(core, 0, sizeof(NES::Core));
        emulators_[idx]->snapshot(core);
        
        // Create capsule to own the memory
        py::capsule capsule(core, [](void* p) {
            delete static_cast<NES::Core*>(p);
        });
        
        return py::array_t<uint8_t>(
            {sizeof(NES::Core)},
            {1},
            reinterpret_cast<uint8_t*>(core),
            capsule
        );
    }
    
    // Load state for single emulator
    void load_state(int idx, const py::array_t<uint8_t>& state) {
        check_idx(idx);
        // Wait for worker to be idle before modifying emulator state
        while (worker_states_[idx]->state.load(std::memory_order_acquire) != STATE_IDLE) {
            std::this_thread::yield();
        }
        emulators_[idx]->restore(reinterpret_cast<const NES::Core*>(state.request().ptr));
        emulators_[idx]->ppu_step();
    }
    
private:
    void check_idx(int idx) const {
        if (idx < 0 || idx >= num_envs_) {
            throw std::out_of_range(
                "Environment index " + std::to_string(idx) + 
                " out of range [0, " + std::to_string(num_envs_) + ")"
            );
        }
    }
    
    void step_impl(py::array_t<uint8_t> actions, int num_frames) {
        using clock = std::chrono::high_resolution_clock;
        auto t0 = clock::now();
        
        auto actions_buf = actions.request();
        uint8_t* actions_ptr = static_cast<uint8_t*>(actions_buf.ptr);
        
        // Set actions and mark workers as pending (lock-free)
        for (int i = 0; i < num_envs_; i++) {
            *emulators_[i]->get_controller(0) = actions_ptr[i];
            worker_frames_[i] = num_frames;
            worker_states_[i]->state.store(STATE_PENDING, std::memory_order_release);
        }
        
        auto t1 = clock::now();
        
        // Busy-wait for all workers to complete (GIL released, lock-free)
        {
            py::gil_scoped_release release;
            
            // Spin until all workers are done
            while (true) {
                bool all_done = true;
                for (int i = 0; i < num_envs_; i++) {
                    if (worker_states_[i]->state.load(std::memory_order_acquire) != STATE_DONE) {
                        all_done = false;
                        break;
                    }
                }
                if (all_done) break;
                
                // Brief pause to reduce CPU spinning overhead
                std::this_thread::yield();
            }
        }
        
        auto t2 = clock::now();
        
        // Mark workers idle (ready for next step)
        for (int i = 0; i < num_envs_; i++) {
            worker_states_[i]->state.store(STATE_IDLE, std::memory_order_release);
        }
        
        auto t3 = clock::now();
        
        // Read configured RAM values (batch read in C++)
        read_ram_values();
        
        auto t4 = clock::now();
        
        // Accumulate timing stats
        timing_setup_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        timing_wait_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        timing_idle_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        timing_ram_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();
        timing_calls_++;
    }
    
    void worker_loop(int idx) {
        using clock = std::chrono::high_resolution_clock;
        
        // Pin this worker thread to spread across NUMA nodes for better memory bandwidth
        // On multi-socket systems (e.g., AMD EPYC), alternate workers between NUMA nodes
        // to avoid all workers competing for same memory controller
        int num_cores = std::thread::hardware_concurrency();
        int target_core = -1;
        if (num_cores > 0) {
            // NUMA-aware pinning: spread workers across sockets
            // Assumes 2 NUMA nodes with cores_per_numa cores each
            // Worker 0 -> NUMA 0 core 0, Worker 1 -> NUMA 1 core 0, 
            // Worker 2 -> NUMA 0 core 1, Worker 3 -> NUMA 1 core 1, etc.
            int cores_per_numa = num_cores / 2;
            if (cores_per_numa > 0) {
                int numa_node = idx % 2;
                int core_in_numa = (idx / 2) % cores_per_numa;
                target_core = numa_node * cores_per_numa + core_in_numa;
            } else {
                target_core = idx % num_cores;
            }
        }
        pin_thread_to_core(target_core);
        worker_timings_[idx].pinned_core = target_core;
        
        // Signal that this worker is ready
        ready_count_.fetch_add(1, std::memory_order_release);
        
        while (true) {
            auto t0 = clock::now();
            
            // Busy-wait for work (lock-free - each worker only checks its own state)
            int state;
            while (true) {
                state = worker_states_[idx]->state.load(std::memory_order_acquire);
                if (state == STATE_PENDING || state == STATE_EXIT) break;
                std::this_thread::yield();
            }
            
            auto t1 = clock::now();
            
            if (state == STATE_EXIT) {
                return;
            }
            
            // Step emulator (the actual work)
            for (int f = 0; f < worker_frames_[idx]; f++) {
                emulators_[idx]->step();
            }
            
            auto t2 = clock::now();
            
            // Signal completion (lock-free - just update our own state)
            worker_states_[idx]->state.store(STATE_DONE, std::memory_order_release);
            
            // Record timing
            worker_timings_[idx].wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            worker_timings_[idx].step_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            worker_timings_[idx].calls++;
        }
    }
    
    int num_envs_;
    std::string rom_path_;
    std::vector<std::unique_ptr<NES::Emulator>> emulators_;
    std::vector<std::thread> workers_;
    std::vector<std::unique_ptr<AlignedAtomic>> worker_states_;  // Cache-line aligned to prevent false sharing
    std::vector<int> worker_frames_;
    
    // Worker ready synchronization (only used during startup)
    std::atomic<int> ready_count_;
    
    // RAM read configuration (set once, used every step)
    std::vector<RamReadSpec> ram_specs_;
    std::vector<int32_t> ram_values_;  // Shape: [num_envs * num_specs], row-major
    int num_ram_specs_ = 0;
    
    // Timing instrumentation (nanoseconds) - main thread
    uint64_t timing_setup_ns_ = 0;   // Set actions + signal workers
    uint64_t timing_wait_ns_ = 0;    // Wait for workers to complete
    uint64_t timing_idle_ns_ = 0;    // Mark workers idle
    uint64_t timing_ram_ns_ = 0;     // Read RAM values
    uint64_t timing_calls_ = 0;      // Number of step calls
    
    // Per-worker timing (cache-line aligned to prevent false sharing)
    struct alignas(64) WorkerTiming {
        uint64_t wait_ns = 0;    // Time waiting for work
        uint64_t step_ns = 0;    // Time doing emulator step
        uint64_t calls = 0;      // Number of step calls
        int pinned_core = -1;    // Core this worker is pinned to
        char padding[64 - 32];   // Pad to cache line
    };
    std::vector<WorkerTiming> worker_timings_;
    
    // Read BCD value from RAM (e.g., score stored as 6 separate digits)
    inline int32_t read_bcd(const uint8_t* ram, uint16_t addr, int size) const {
        int32_t result = 0;
        for (int i = 0; i < size; i++) {
            result = result * 10 + ram[addr + i];
        }
        return result;
    }
    
    // Read RAM values for all emulators after step (called from main thread)
    void read_ram_values() {
        if (num_ram_specs_ == 0) return;
        
        for (int env = 0; env < num_envs_; env++) {
            const uint8_t* ram = reinterpret_cast<const uint8_t*>(
                emulators_[env]->get_memory_buffer());
            int base = env * num_ram_specs_;
            
            for (int s = 0; s < num_ram_specs_; s++) {
                const auto& spec = ram_specs_[s];
                if (spec.type == RamReadType::BCD) {
                    ram_values_[base + s] = read_bcd(ram, spec.address, spec.size);
                } else {
                    ram_values_[base + s] = ram[spec.address];
                }
            }
        }
    }
    
public:
    // Configure RAM addresses to read after each step
    // specs: list of (address, size, type) where type is 0=INT, 1=BCD
    void configure_ram_reads(const std::vector<std::tuple<uint16_t, uint8_t, int>>& specs) {
        ram_specs_.clear();
        ram_specs_.reserve(specs.size());
        
        for (const auto& spec : specs) {
            uint16_t addr = std::get<0>(spec);
            uint8_t size = std::get<1>(spec);
            int type = std::get<2>(spec);
            ram_specs_.emplace_back(addr, size, 
                type == 1 ? RamReadType::BCD : RamReadType::INT);
        }
        
        num_ram_specs_ = static_cast<int>(ram_specs_.size());
        ram_values_.resize(num_envs_ * num_ram_specs_);
    }
    
    // Get RAM values as numpy array, shape: (num_envs, num_specs)
    py::array_t<int32_t> ram_values() const {
        if (num_ram_specs_ == 0) {
            // Return empty array with explicit shape
            std::vector<ssize_t> shape = {static_cast<ssize_t>(num_envs_), 0};
            return py::array_t<int32_t>(shape);
        }
        
        return py::array_t<int32_t>(
            {num_envs_, num_ram_specs_},
            {num_ram_specs_ * static_cast<int>(sizeof(int32_t)), static_cast<int>(sizeof(int32_t))},
            ram_values_.data(),
            py::capsule(ram_values_.data(), [](void*) {})
        );
    }
    
    // Get timing stats as dict and reset counters
    py::dict get_timing_stats() {
        py::dict stats;
        if (timing_calls_ > 0) {
            stats["calls"] = timing_calls_;
            stats["setup_ms"] = timing_setup_ns_ / 1e6;
            stats["wait_ms"] = timing_wait_ns_ / 1e6;
            stats["idle_ms"] = timing_idle_ns_ / 1e6;
            stats["ram_ms"] = timing_ram_ns_ / 1e6;
            stats["total_ms"] = (timing_setup_ns_ + timing_wait_ns_ + timing_idle_ns_ + timing_ram_ns_) / 1e6;
        }
        // Reset main thread timing
        timing_setup_ns_ = timing_wait_ns_ = timing_idle_ns_ = timing_ram_ns_ = timing_calls_ = 0;
        return stats;
    }
    
    // Get per-worker timing stats and reset counters
    py::list get_worker_timing_stats() {
        py::list workers;
        for (int i = 0; i < num_envs_; i++) {
            py::dict w;
            w["worker"] = i;
            w["core"] = worker_timings_[i].pinned_core;
            w["calls"] = worker_timings_[i].calls;
            w["wait_ms"] = worker_timings_[i].wait_ns / 1e6;
            w["step_ms"] = worker_timings_[i].step_ns / 1e6;
            if (worker_timings_[i].calls > 0) {
                w["step_avg_us"] = (worker_timings_[i].step_ns / worker_timings_[i].calls) / 1e3;
            } else {
                w["step_avg_us"] = 0.0;
            }
            workers.append(w);
            
            // Reset
            worker_timings_[i].wait_ns = 0;
            worker_timings_[i].step_ns = 0;
            worker_timings_[i].calls = 0;
        }
        return workers;
    }
};

PYBIND11_MODULE(emulator, m) {   
    py::class_<NES::Emulator>(m, "NESEmulator")
        .def(py::init<const std::string&>())

        .def_property_readonly_static("width", [](py::object) { return NES::Emulator::WIDTH; })
        .def_property_readonly_static("height", [](py::object) { return NES::Emulator::HEIGHT; })        

        .def("reset", &NES::Emulator::reset, "Reset the emulator")
        .def("step", &NES::Emulator::step, py::call_guard<py::gil_scoped_release>(), "Perform a step on the emulator (GIL released)")
        
        .def(
            "screen_buffer", 
            [](NES::Emulator& emu) -> py::array_t<uint8_t> {
                const int HEIGHT = NES::Emulator::HEIGHT;
                const int WIDTH = NES::Emulator::WIDTH;
                
                #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
                    // On little-endian systems: BGRx -> RGB
                    return py::array_t<uint8_t>(
                        {HEIGHT, WIDTH, 3},                    // shape (3 channels)
                        {WIDTH * 4, 4, -1},                   // negative stride to reverse BGR->RGB
                        reinterpret_cast<uint8_t*>(emu.get_screen_buffer()) + 2,  // start at B
                        py::capsule(emu.get_screen_buffer(), [](void*) {})  // capsule with data pointer
                    );
                #else
                    // On big-endian systems: xRGB -> RGB
                    return py::array_t<uint8_t>(
                        {HEIGHT, WIDTH, 3},                    // shape (3 channels)
                        {WIDTH * 4, 4, 1},                    // normal stride
                        reinterpret_cast<uint8_t*>(emu.get_screen_buffer()) + 1,  // skip x
                        py::capsule(emu.get_screen_buffer(), [](void*) {})  // capsule with data pointer
                    );
                #endif
            }, 
            "Get the screen buffer as a HEIGHT x WIDTH x 3 numpy.ndarray in RGB format"
        )

        .def(
            "controller",
            [](NES::Emulator& emu, int port) -> py::array_t<uint8_t> {
                // Create a view of the controller buffer
                return py::array_t<uint8_t>(
                    {1},                                    // shape (1 controller)
                    {1},                                    // stride (1 byte per controller)
                    reinterpret_cast<uint8_t*>(emu.get_controller(port)),  // pointer to data
                    py::capsule(emu.get_controller(port), [](void*) {})    // capsule with data pointer
                );
            },
            py::arg("port"),
            "Get the controller buffer as numpy.ndarray"
        )

        .def(
            "memory_buffer", 
            [](NES::Emulator& emu) -> py::array_t<uint8_t> {
                // Create a view of the RAM buffer (0x800 bytes)
                return py::array_t<uint8_t>(
                    {0x800},                               // shape (2048 bytes)
                    {1},                                   // stride (1 byte)
                    reinterpret_cast<uint8_t*>(emu.get_memory_buffer()),  // pointer to data
                    py::capsule(emu.get_memory_buffer(), [](void*) {})    // capsule with data pointer
                );
            }, 
            "Get the memory buffer as numpy.ndarray"
        )

        .def(
            "dump_state",
            [](NES::Emulator& emu) -> py::array_t<uint8_t> {
                // Create a copy of the state data
                auto* core = new NES::Core;
                memset(core, 0, sizeof(NES::Core));
                emu.snapshot(core);
                
                return py::array_t<uint8_t>(
                    {sizeof(NES::Core)},
                    {1},
                    reinterpret_cast<uint8_t*>(core),
                    py::capsule(core, [](void* p) { delete static_cast<NES::Core*>(p); })
                );
            },
            "Dump the current state to bytes"
        )

        .def(
            "load_state",
            [](NES::Emulator& emu, const py::array_t<uint8_t>& state) {
                emu.restore(reinterpret_cast<const NES::Core*>(state.request().ptr));
                emu.ppu_step();
            },
            py::arg("state"),
            "Load state from bytes"
        )
    ;
    
    // VectorEmulator - parallel NES emulation mirroring NESEmulator interface
    py::class_<VectorEmulator>(m, "VectorEmulator")
        .def(py::init<const std::string&, int>(), 
             py::arg("rom_path"), py::arg("num_envs"),
             "Create multiple emulators for parallel stepping")
        
        .def_property_readonly("num_envs", &VectorEmulator::num_envs)
        .def_property_readonly_static("height", [](py::object) { return NES::Emulator::HEIGHT; })
        .def_property_readonly_static("width", [](py::object) { return NES::Emulator::WIDTH; })
        
        .def("reset", &VectorEmulator::reset, "Reset all emulators")
        .def("reset_env", &VectorEmulator::reset_env, py::arg("idx"), "Reset a single emulator")
        
        .def("step", &VectorEmulator::step, 
             py::arg("actions"),
             "Step all emulators 1 frame in parallel")
        
        .def("step_single", &VectorEmulator::step_single,
             py::arg("idx"), py::arg("action"),
             "Step a single emulator (synchronous, no threading)")
        
        .def("screen_buffer", 
             py::overload_cast<>(&VectorEmulator::screen_buffer),
             "Get screen buffers for all emulators as list of zero-copy views")
        
        .def("screen_buffer", 
             py::overload_cast<int>(&VectorEmulator::screen_buffer_single),
             py::arg("idx"),
             "Get screen buffer for single emulator as zero-copy view")
        
        .def("memory_buffer", 
             py::overload_cast<>(&VectorEmulator::memory_buffer),
             "Get memory buffers for all emulators as list of zero-copy views")
        
        .def("memory_buffer", 
             py::overload_cast<int>(&VectorEmulator::memory_buffer_single),
             py::arg("idx"),
             "Get memory buffer for single emulator as zero-copy view")
        
        .def("controller", &VectorEmulator::controller,
             py::arg("idx"), py::arg("port") = 0,
             "Get controller buffer for single emulator as zero-copy view")
        
        .def("dump_state", &VectorEmulator::dump_state,
             py::arg("idx"),
             "Dump state for single emulator")
        
        .def("load_state", &VectorEmulator::load_state,
             py::arg("idx"), py::arg("state"),
             "Load state for single emulator")
        
        .def("configure_ram_reads", &VectorEmulator::configure_ram_reads,
             py::arg("specs"),
             R"doc(
Configure RAM addresses to read after each step.

Args:
    specs: List of (address, size, type) tuples where:
        - address: RAM address (0x0000-0x07FF)
        - size: Number of bytes to read (1-6)
        - type: 0=INT (single byte), 1=BCD (multiple digits)

Example:
    emulator.configure_ram_reads([
        (0x07DE, 6, 1),  # score (6 BCD digits)
        (0x07F8, 3, 1),  # time (3 BCD digits)
        (0x07ED, 2, 1),  # coins (2 BCD digits)
        (0x075A, 1, 0),  # life (1 byte int)
    ])
)doc")
        
        .def("ram_values", &VectorEmulator::ram_values,
             "Get RAM values from last step as numpy array, shape: (num_envs, num_specs)")
        
        .def("get_timing_stats", &VectorEmulator::get_timing_stats,
             R"doc(
Get timing stats for step() breakdown and reset counters.

Returns dict with:
    - calls: Number of step() calls since last reset
    - setup_ms: Time to set actions + signal workers (ms)
    - wait_ms: Time waiting for workers to complete (ms)
    - idle_ms: Time to mark workers idle (ms)
    - ram_ms: Time to read RAM values (ms)
    - total_ms: Total C++ time in step() (ms)
)doc")
        
        .def("get_worker_timing_stats", &VectorEmulator::get_worker_timing_stats,
             R"doc(
Get per-worker timing stats and reset counters.

Returns list of dicts, one per worker:
    - worker: Worker index
    - core: CPU core this worker is pinned to
    - calls: Number of step calls
    - wait_ms: Time spent waiting for work (ms)
    - step_ms: Time spent stepping emulator (ms)
    - step_avg_us: Average step time per call (microseconds)
)doc")
    ;
};
