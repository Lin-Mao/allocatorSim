PROJECT := allocatorSim

OBJ_DIR := obj/
SRC_DIR := ./
INC_DIR := ./
CUR_DIR := $(shell pwd)

CXX ?=

CFLAGS := -std=c++17 -Wall
LDFLAGS ?=

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
	$(CXX) $(LDFLAGS) -o $@ $^

$(OBJ_DIR)%.o: $(SRC_DIR)%.cpp
	$(CXX) $(CFLAGS) -I$(INC_DIR) -o $@ -c $<

.PHONY: clean
clean:
	-rm -rf $(OBJ_DIR) $(PROJECT)