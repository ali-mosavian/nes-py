//  Program:      nes-py
//  File:         emulator.hpp
//  Description:  This class houses the logic and data for an NES emulator
//
//  Copyright (c) 2019 Christian Kauten. All rights reserved.
//

#ifndef EMULATOR_HPP
#define EMULATOR_HPP

#include <array>
#include <string>
#include <algorithm>
#include "common.hpp"
#include "cartridge.hpp"
#include "controller.hpp"
#include "cpu.hpp"
#include "ppu.hpp"
#include "main_bus.hpp"
#include "picture_bus.hpp"

namespace NES {

struct Core {
    /// the main data bus of the emulator
    MainBus bus;
    /// The emulator's CPU
    CPU cpu;
    /// the emulators' PPU
    PPU ppu;
    /// the picture bus from the PPU of the emulator
    PictureBus picture_bus;

    void initialize(Controller* const controllers);    
    void reset();
    void set_mapper(Mapper *mapper);
    Mapper* get_mapper() const { return bus.get_mapper(); }
    void ppu_step(NESFrameBufferT* const framebuffer);
    void step(NESFrameBufferT* const framebuffer);

    /// Copy state from another Core, preserving local mapper and callbacks.
    /// This is safe to use with snapshots from destroyed emulators because
    /// we only copy raw data, not pointers or function objects.
    inline void copy_state_from(const Core& other) {
        // Copy RAM state (preserves mapper and callbacks)
        bus.copy_ram_from(other.bus);
        // Copy CPU state (all plain data)
        cpu = other.cpu;
        // Copy PPU state (preserves vblank_callback)
        ppu.copy_state_from(other.ppu);
        // Copy picture bus RAM (preserves mapper)
        picture_bus.copy_ram_from(other.picture_bus);
    }
};

/// An NES Emulator and OpenAI Gym interface
class Emulator {
 public:
    /// The width of the NES screen in pixels
    static const int WIDTH = SCANLINE_VISIBLE_DOTS;
    /// The height of the NES screen in pixels
    static const int HEIGHT = VISIBLE_SCANLINES;

    /// Initialize a new emulator with a path to a ROM file.
    ///
    /// @param rom_path the path to the ROM for the emulator to run
    ///
    explicit Emulator(std::string rom_path);

    /// Return a 32-bit pointer to the screen buffer's first address.
    ///
    /// @return a 32-bit pointer to the screen buffer's first address
    ///
    inline NESFrameBufferT* const get_screen_buffer() { return &framebuffer; }

    /// Return a 8-bit pointer to the RAM buffer's first address.
    ///
    /// @return a 8-bit pointer to the RAM buffer's first address
    ///
    inline NES_Byte* get_memory_buffer() { return core.bus.get_memory_buffer(); }

    /// Return a pointer to a controller port
    ///
    /// @param port the port of the controller to return the pointer to
    /// @return a pointer to the byte buffer for the controller state
    ///
    inline NES_Byte* get_controller(int port) {
        return controllers[port].get_joypad_buffer();
    }

    /// Load the ROM into the NES.
    inline void reset() { core.cpu.reset(core.bus); core.ppu.reset(); }

    /// Perform a step on the emulator, i.e., a single frame.
    void step();

    /// Perform a step on the PPU, i.e., a single frame.
    void ppu_step();

    /// Create a snapshot state on the emulator.
    inline void snapshot(Core* const core) {
        *core = this->core;
    }

    /// Restore the snapshot state on the emulator.
    /// Uses copy_state_from to safely copy only data, preserving local
    /// mapper pointer and callbacks which may contain dangling references
    /// in snapshots from destroyed emulators.
    inline void restore(const Core* const core) {
        this->core.copy_state_from(*core);
    }

 private:
    /// The number of cycles in 1 frame
    static const int CYCLES_PER_FRAME = 29781;

    /// the core of the emulator
    Core core;    
    /// the virtual cartridge with ROM and mapper data
    Cartridge cartridge;
    /// the 2 controllers on the emulator
    Controller controllers[2];

    /// the rendering framebuffer of the emulator
    NESFrameBufferT framebuffer;
};

}  // namespace NES

#endif  // EMULATOR_HPP