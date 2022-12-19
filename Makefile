PROJECT := allocatorSim

OBJ_DIR := obj/
SRC_DIR := src/
INC_DIR := include/
CUR_DIR := $(shell pwd)

CXX ?=

CFLAGS := -std=c++17
LDFLAGS ?=

SRCS := $(notdir $(wildcard $(SRC_DIR)*.cpp))
OBJS := $(addprefix $(OBJ_DIR), $(patsubst %.cpp, %.o, $(SRCS)))

all: dirs exes

dirs: $(OBJ_DIR)
exes: $(PROJECT)

$(OBJ_DIR):
	mkdir -p $@

$(PROJECT): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

$(OBJ_DIR)%.o: $(SRC_DIR)%.cpp
	$(CXX) $(CFLAGS) -I$(INC_DIR) -o $@ -c $<


clean:
	-rm -rf $(OBJ_DIR) $(PROJECT)
