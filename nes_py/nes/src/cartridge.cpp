//  Program:      nes-py
//  File:         cartridge.cpp
//  Description:  This class houses the logic and data for an NES cartridge
//
//  Copyright (c) 2019 Christian Kauten. All rights reserved.
//

#include <fstream>
#include <stdexcept>
#include "cartridge.hpp"

namespace NES {

void Cartridge::loadFromFile(std::string path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("ROM file not found or not readable: " + path);
    }
    
    // Read and validate iNES header (16 bytes)
    std::vector<NES_Byte> header(0x10);
    if (!file.read(reinterpret_cast<char*>(header.data()), 0x10)) {
        throw std::runtime_error("ROM file too small to contain iNES header: " + path);
    }
    
    // Validate magic bytes "NES\x1A"
    if (header[0] != 'N' || header[1] != 'E' || header[2] != 'S' || header[3] != 0x1A) {
        throw std::runtime_error("Invalid iNES header (missing NES magic bytes): " + path);
    }
    
    // Parse header
    NES_Byte prg_banks = header[4];
    NES_Byte chr_banks = header[5];
    
    if (prg_banks == 0) {
        throw std::runtime_error("Invalid ROM: PRG ROM size is 0: " + path);
    }
    
    name_table_mirroring = header[6] & 0xB;
    mapper_number = ((header[6] >> 4) & 0xf) | (header[7] & 0xf0);
    has_extended_ram = header[6] & 0x2;
    bool has_trainer = header[6] & 0x04;
    
    // Skip trainer if present (512 bytes)
    if (has_trainer) {
        file.seekg(512, std::ios::cur);
    }
    
    // Read PRG-ROM (16KB per bank)
    size_t prg_size = 0x4000 * prg_banks;
    prg_rom.resize(prg_size);
    if (!file.read(reinterpret_cast<char*>(prg_rom.data()), prg_size)) {
        throw std::runtime_error("ROM file truncated while reading PRG ROM: " + path);
    }
    
    // Read CHR-ROM (8KB per bank) if present
    if (chr_banks > 0) {
        size_t chr_size = 0x2000 * chr_banks;
        chr_rom.resize(chr_size);
        if (!file.read(reinterpret_cast<char*>(chr_rom.data()), chr_size)) {
            throw std::runtime_error("ROM file truncated while reading CHR ROM: " + path);
        }
    }
}

}  // namespace NES
