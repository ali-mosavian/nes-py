# Get Python-specific paths using shell commands
VENV_SITE_PACKAGES := $(shell python3 -c 'import site; print(site.getsitepackages()[0])')
PYBIND11_PATH := $(VENV_SITE_PACKAGES)/pybind11/include
PYTHON_INCLUDE := $(shell python3 -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')
PYTHON_LIBDIR := $(shell python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
PYTHON_VERSION := $(shell python3 -c 'import sysconfig; print(sysconfig.get_config_var("LDVERSION") or sysconfig.get_config_var("VERSION"))')

# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++14 -O3 -pipe -fPIC -Wno-unused-value
INCLUDES := -I$(dir $(lastword $(MAKEFILE_LIST)))include -I$(PYBIND11_PATH) -I$(PYTHON_INCLUDE)

# Platform-specific settings and common LDFLAGS
# Don't link python library directly - use undefined dynamic_lookup
LDFLAGS :=

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -undefined dynamic_lookup
    ifeq ($(shell uname -m),arm64)
        CXXFLAGS += -arch arm64
        LDFLAGS += -arch arm64
    endif
endif

# Source files
SRC_DIR := $(dir $(lastword $(MAKEFILE_LIST)))src
BUILD_DIR := $(dir $(lastword $(MAKEFILE_LIST)))build
SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Target library
TARGET := $(dir $(lastword $(MAKEFILE_LIST)))libemulator.so

# Default target
all: $(BUILD_DIR) $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build the shared library
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -shared $(LDFLAGS) -o $@ $^

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean 