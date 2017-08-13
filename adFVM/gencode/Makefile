CC=nvcc
CFLAGS=-std=c++11 -Xcompiler="-fPIC" -O3 -g -DGPU --ptxas-options=-v
LDFLAGS=-shared
INCDIRS=-I/usr/include/python2.7 -I/usr/lib/python2.7/site-packages/numpy/core/include -I/home/talnikar/adFVM/adFVM/cpp/include
LIBDIRS=-L/usr/lib/openmpi
LIBS=-lpython2.7 -lmpi
BUILD=build
OBJ=$(BUILD)/interface.o $(BUILD)/kernel.o $(BUILD)/code.o $(BUILD)/parallel.o $(BUILD)/mesh.o
TARGET=interface.so

all:
	python2 setup.py build_ext --inplace
python3:
	python3 setup.py build_ext --inplace

$(BUILD):
	mkdir $(BUILD)

$(BUILD)/%.o: %.cpp 
	$(CC) -x cu $(CFLAGS) $(INCDIRS) $^ -c -o $@

$(TARGET): $(BUILD) $(OBJ)
	$(CC) $(LDFLAGS) $(LIBDIRS) $(LIBS) $(OBJ) -o $(TARGET)

clean: $(BUILD) $(TARGET)

gpu: $(TARGET)

