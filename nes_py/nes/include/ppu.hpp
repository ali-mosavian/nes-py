//  Program:      nes-py
//  File:         ppu.hpp
//  Description:  This class houses the logic and data for the PPU of an NES
//
//  Copyright (c) 2019 Christian Kauten. All rights reserved.
//

#ifndef PPU_HPP
#define PPU_HPP

#include "common.hpp"
#include "picture_bus.hpp"
#include <array>

namespace NES {

/// The number of visible scan lines (i.e., the height of the screen)
const int VISIBLE_SCANLINES = 240;
/// The number of visible dots per scan line (i.e., the width of the screen)
const int SCANLINE_VISIBLE_DOTS = 256;
/// The number of cycles per scanline
const int SCANLINE_CYCLE_LENGTH = 341;
/// The last cycle of a scan line (changed from 340 to fix render glitch)
const int SCANLINE_END_CYCLE = 341;
/// The last scanline per frame
const int FRAME_END_SCANLINE = 261;

typedef NES_Pixel NESFrameBufferT[VISIBLE_SCANLINES][SCANLINE_VISIBLE_DOTS];

/// The Picture Processing Unit (PPU) for the NES
class PPU {
 private:
    /// The callback to fire when entering vertical blanking mode
    std::function<void(void)> vblank_callback;
    /// The OAM memory (sprites)
    static_vector<NES_Byte, 64 * 4> sprite_memory;
    /// OAM memory (sprites) for the next scanline
    static_vector<NES_Byte, 8> scanline_sprites;

    /// The current pipeline state of the PPU
    enum State {
        PRE_RENDER,
        RENDER,
        POST_RENDER,
        VERTICAL_BLANK
    } pipeline_state;

    /// The number of cycles left in the frame
    int cycles;
    /// the current scanline of the frame
    int scanline;
    /// whether the PPU is on an even frame
    bool is_even_frame;

    // Status

    /// whether the PPU is in vertical blanking mode
    bool is_vblank;
    /// whether sprite 0 has been hit (i.e., collision detection)
    bool is_sprite_zero_hit;

    // Registers

    /// the current data address to (read / write) (from / to)
    NES_Address data_address;
    /// a temporary address register
    NES_Address temp_address;
    /// the fine scrolling position
    NES_Byte fine_x_scroll;
    /// TODO: doc
    bool is_first_write;
    /// The address of the data buffer
    NES_Byte data_buffer;
    /// the read / write address for the OAM memory (sprites)
    NES_Byte sprite_data_address;

    // Mask

    /// whether the PPU is showing sprites
    bool is_showing_sprites;
    /// whether the PPU is showing background pixels
    bool is_showing_background;
    /// whether the PPU is hiding sprites along the edges
    bool is_hiding_edge_sprites;
    /// whether the PPU is hiding the background along the edges
    bool is_hiding_edge_background;

    // Setup flags and variables

    /// TODO: doc
    bool is_long_sprites;
    /// whether the PPU is in the interrupt handler
    bool is_interrupting;

    /// TODO: doc
    enum CharacterPage {
        LOW,
        HIGH,
    } background_page, sprite_page;

    /// The value to increment the data address by
    NES_Address data_address_increment;

 public:
    /// Initialize a new PPU.
    PPU() { }

    /// Perform a single cycle on the PPU.
    void cycle(PictureBus& bus, NESFrameBufferT* const screen);

    /// Reset the PPU.
    void reset();

    /// Set the interrupt callback for the CPU.
    inline void set_interrupt_callback(std::function<void(void)> cb) {
        vblank_callback = cb;
    }

    /// Copy state from another PPU (preserves vblank_callback)
    inline void copy_state_from(const PPU& other) {
        // Save local callback
        auto local_callback = vblank_callback;
        // Copy all data
        sprite_memory = other.sprite_memory;
        scanline_sprites = other.scanline_sprites;
        pipeline_state = other.pipeline_state;
        cycles = other.cycles;
        scanline = other.scanline;
        is_even_frame = other.is_even_frame;
        is_vblank = other.is_vblank;
        is_sprite_zero_hit = other.is_sprite_zero_hit;
        data_address = other.data_address;
        temp_address = other.temp_address;
        fine_x_scroll = other.fine_x_scroll;
        is_first_write = other.is_first_write;
        data_buffer = other.data_buffer;
        sprite_data_address = other.sprite_data_address;
        is_showing_sprites = other.is_showing_sprites;
        is_showing_background = other.is_showing_background;
        is_hiding_edge_sprites = other.is_hiding_edge_sprites;
        is_hiding_edge_background = other.is_hiding_edge_background;
        is_long_sprites = other.is_long_sprites;
        is_interrupting = other.is_interrupting;
        background_page = other.background_page;
        sprite_page = other.sprite_page;
        data_address_increment = other.data_address_increment;
        // Restore local callback (DON'T copy from other)
        vblank_callback = local_callback;
    }

    /// TODO: doc
    void do_DMA(const NES_Byte* page_ptr);

    // MARK: Callbacks mapped to CPU address space

    /// Set the control register to a new value.
    ///
    /// @param ctrl the new control register byte
    ///
    void control(NES_Byte ctrl);

    /// Set the mask register to a new value.
    ///
    /// @param mask the new mask value
    ///
    void set_mask(NES_Byte mask);

    /// Set the scroll register to a new value.
    ///
    /// @param scroll the new scroll register value
    ///
    void set_scroll(NES_Byte scroll);

    /// Return the value in the PPU status register.
    NES_Byte get_status();

    /// TODO: doc
    void set_data_address(NES_Byte address);

    /// Read data off the picture bus.
    ///
    /// @param bus the bus to read data off of
    ///
    NES_Byte get_data(PictureBus& bus);

    /// TODO: doc
    void set_data(PictureBus& bus, NES_Byte data);

    /// Set the sprite data address to a new value.
    ///
    /// @param address the new OAM data address
    ///
    inline void set_OAM_address(NES_Byte address) {
        sprite_data_address = address;
    }

    /// Read a byte from OAM memory at the sprite data address.
    ///
    /// @return the byte at the given address in OAM memory
    ///
    inline NES_Byte get_OAM_data() {
        return sprite_memory[sprite_data_address];
    }

    /// Write a byte to OAM memory at the sprite data address.
    ///
    /// @param value the byte to write to the given address
    ///
    inline void set_OAM_data(NES_Byte value) {
        sprite_memory[sprite_data_address++] = value;
    }
};

}  // namespace NES

#endif  // PPU_HPP
