CC = nvcc
INC_DIR = ./include
LIB_DIR = ./lib
SRC_DIR = ./src
TARGET_DIR = ./target
OBJ_DIR = ./obj
FLAGS = -ccbin g++ -O3 -std=c++14 -Xcompiler -fopenmp -arch=compute_60 -code=sm_60 -lcuda -lineinfo -I $(INC_DIR)


OBJS_RAW = BPSDBSCAN.o Dataset.o DBSCAN.o DenseBox.o HybridDBSCAN.o OriginalDBSCAN.o Shadower.o utils.o main.o
OBJS = $(addprefix $(OBJ_DIR)/, $(OBJS_RAW))

INCS_RAW = BPSDBSCAN.hpp Dataset.hpp DBSCAN.hpp DenseBox.hpp global.hpp HybridDBSCAN.hpp OriginalDBSCAN.hpp Shadower.cuh utils.hpp
INCS = $(addprefix $(INC_DIR)/, $(INCS_RAW))

TARGET = $(TARGET_DIR)/multicoreDBSCAN

mkdir:
	mkdir -p ./target
	mkdir -p ./obj

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(FLAGS) $^ -o $@

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp $(INCS)
	$(CC) $(FLAGS) -c -o $@ $<

$(OBJ_DIR)/BPSDBSCAN.o: $(LIB_DIR)/BPSDBSCAN.cpp $(INCS)
	$(CC) $(FLAGS) -c -o $@ $<

$(OBJ_DIR)/Dataset.o: $(LIB_DIR)/Dataset.cpp $(INCS)
	$(CC) $(FLAGS) -c -o $@ $<

$(OBJ_DIR)/DBSCAN.o: $(LIB_DIR)/DBSCAN.cu $(INCS)
	$(CC) $(FLAGS) -c -o $@ $<

$(OBJ_DIR)/DenseBox.o: $(LIB_DIR)/DenseBox.cpp $(INCS)
	$(CC) $(FLAGS) -c -o $@ $<

$(OBJ_DIR)/HybridDBSCAN.o: $(LIB_DIR)/HybridDBSCAN.cu $(INCS)
	$(CC) $(FLAGS) -c -o $@ $<

$(OBJ_DIR)/OriginalDBSCAN.o: $(LIB_DIR)/OriginalDBSCAN.cpp $(INCS)
	$(CC) $(FLAGS) -c -o $@ $<

$(OBJ_DIR)/Shadower.o: $(LIB_DIR)/Shadower.cu $(INCS)
	$(CC) $(FLAGS) -c -o $@ $<

$(OBJ_DIR)/utils.o: $(LIB_DIR)/utils.cpp $(INCS)
	$(CC) $(FLAGS) -c -o $@ $<
	
.PHONY:clean
clean:
	rm -rf ./target
	rm -rf ./obj