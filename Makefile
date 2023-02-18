PROJECT := allocatorsim
CONFIGS := Makefile.config

include $(CONFIGS)

OBJ_DIR := obj/
SRC_DIR := src/
INC_DIR := include/
LIB_DIR := lib/
EXE_DIR := bin/
APP := $(EXE_DIR)$(PROJECT)
LIB := $(LIB_DIR)lib$(PROJECT).so
CUR_DIR := $(shell pwd)

CXX ?=

CFLAGS := -std=c++17 -Wall -I$(PYTHON_INCLUDE_DIR) -I$(PYBIND11_DIR)/include
LDFLAGS ?= -L$(PYTHON_LIB_DIR)
LIBRARY ?= -lpython$(PYTHON_VERSION) -lunwind

ifdef DEBUG
CFLAGS += -g -O0
else
CFLAGS += -O3
endif

SRCS := $(notdir $(wildcard $(SRC_DIR)*.cpp $(SRC_DIR)*/*.cpp))
OBJS := $(addprefix $(OBJ_DIR), $(patsubst %.cpp, %.o, $(SRCS)))

.PHONY: all
all: dirs app lib

dirs: $(OBJ_DIR) $(EXE_DIR) $(LIB_DIR)
app: $(APP)
lib: $(LIB)

$(EXE_DIR):
	mkdir -p $@

$(LIB_DIR):
	mkdir -p $@

$(OBJ_DIR):
	mkdir -p $@

$(APP): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBRARY)

$(LIB): $(OBJS)
	$(CXX) $(LDFLAGS) -fPIC -shared -o $@ $^ $(LIBRARY)

$(OBJ_DIR)%.o: $(SRC_DIR)%.cpp
	$(CXX) $(CFLAGS) -fPIC -I$(INC_DIR) -I$(INC_DIR)/utils -o $@ -c $<

$(OBJ_DIR)%.o: $(SRC_DIR)/*/%.cpp
	$(CXX) $(CFLAGS) -fPIC -I$(INC_DIR) -I$(INC_DIR)/utils -o $@ -c $<

.PHONY: clean
clean:
	-rm -rf $(OBJ_DIR) $(EXE_DIR) $(LIB_DIR) build/
