PROJECT := allocatorSim
CONFIGS := Makefile.config

include $(CONFIGS)

OBJ_DIR := obj/
SRC_DIR := src/
INC_DIR := include/
CUR_DIR := $(shell pwd)

CXX ?=

CFLAGS := -std=c++17 -Wall -I$(PYTHON_INCLUDE_DIR) -I$(PYBIND11_DIR)/include
LDFLAGS ?= -L$(PYTHON_LIB_DIR)
LIBRARY ?= -lpython$(PYTHON_VERSION)

ifdef DEBUG
CFLAGS += -g -O0
else
CFLAGS += -O3
endif

SRCS := $(notdir $(wildcard $(SRC_DIR)*.cpp))
OBJS := $(addprefix $(OBJ_DIR), $(patsubst %.cpp, %.o, $(SRCS)))

.PHONY: all
all: dirs exes

dirs: $(OBJ_DIR)
exes: $(PROJECT)

$(OBJ_DIR):
	mkdir -p $@

$(PROJECT): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBRARY)

$(OBJ_DIR)%.o: $(SRC_DIR)%.cpp
	$(CXX) $(CFLAGS) -I$(INC_DIR) -o $@ -c $<

.PHONY: clean
clean:
	-rm -rf $(OBJ_DIR) $(PROJECT)