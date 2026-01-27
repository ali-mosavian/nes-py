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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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
    static constexpr int STATE_IDLE = 0;
    static constexpr int STATE_PENDING = 1;
    static constexpr int STATE_DONE = 2;
    static constexpr int STATE_EXIT = 3;

public:
    VectorEmulator(const std::string& rom_path, int num_envs) 
        : num_envs_(num_envs), rom_path_(rom_path), done_count_(0) {
        
        emulators_.reserve(num_envs);
        worker_states_.reserve(num_envs);
        worker_frames_.resize(num_envs, 1);
        
        for (int i = 0; i < num_envs; i++) {
            emulators_.push_back(std::make_unique<NES::Emulator>(rom_path));
            worker_states_.push_back(std::make_unique<std::atomic<int>>(STATE_IDLE));
        }
        
        workers_.reserve(num_envs);
        for (int i = 0; i < num_envs; i++) {
            workers_.emplace_back(&VectorEmulator::worker_loop, this, i);
        }
    }
    
    ~VectorEmulator() {
        for (int i = 0; i < num_envs_; i++) {
            worker_states_[i]->store(STATE_EXIT, std::memory_order_release);
        }
        start_cv_.notify_all();
        
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
        while (worker_states_[idx]->load(std::memory_order_acquire) != STATE_IDLE) {
            std::this_thread::yield();
        }
        emulators_[idx]->reset();
    }
    
    // Step all emulators 1 frame in parallel (like NESEmulator.step())
    void step(py::array_t<uint8_t> actions) {
        step_impl(actions, 1);
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
        auto actions_buf = actions.request();
        uint8_t* actions_ptr = static_cast<uint8_t*>(actions_buf.ptr);
        
        // Set actions and signal workers
        done_count_.store(0, std::memory_order_release);
        for (int i = 0; i < num_envs_; i++) {
            *emulators_[i]->get_controller(0) = actions_ptr[i];
            worker_frames_[i] = num_frames;
            worker_states_[i]->store(STATE_PENDING, std::memory_order_release);
        }
        start_cv_.notify_all();
        
        // Wait for completion (GIL released)
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(done_mutex_);
            done_cv_.wait(lock, [this]() {
                return done_count_.load(std::memory_order_acquire) == num_envs_;
            });
        }
        
        // Mark workers idle
        for (int i = 0; i < num_envs_; i++) {
            worker_states_[i]->store(STATE_IDLE, std::memory_order_release);
        }
    }
    
    void worker_loop(int idx) {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(start_mutex_);
                start_cv_.wait(lock, [this, idx]() {
                    int state = worker_states_[idx]->load(std::memory_order_acquire);
                    return state == STATE_PENDING || state == STATE_EXIT;
                });
            }
            
            if (worker_states_[idx]->load(std::memory_order_acquire) == STATE_EXIT) {
                return;
            }
            
            // Step emulator
            for (int f = 0; f < worker_frames_[idx]; f++) {
                emulators_[idx]->step();
            }
            
            // Signal completion
            worker_states_[idx]->store(STATE_DONE, std::memory_order_release);
            if (done_count_.fetch_add(1, std::memory_order_release) + 1 == num_envs_) {
                done_cv_.notify_one();
            }
        }
    }
    
    int num_envs_;
    std::string rom_path_;
    std::vector<std::unique_ptr<NES::Emulator>> emulators_;
    std::vector<std::thread> workers_;
    std::vector<std::unique_ptr<std::atomic<int>>> worker_states_;
    std::vector<int> worker_frames_;
    
    std::mutex start_mutex_;
    std::condition_variable start_cv_;
    std::mutex done_mutex_;
    std::condition_variable done_cv_;
    std::atomic<int> done_count_;
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
    ;
};
