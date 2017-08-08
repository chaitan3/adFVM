CC=nvcc
CFLAGS=-std=c++11 -Xcompiler="-fPIC" -O3 -g -DGPU
LDFLAGS=
INCDIRS=-I/usr/include/python2.7 -I/usr/lib/python2.7/site-packages/numpy/core/include
LIBDIRS=-I/usr/lib/openmpi
LIBS=-lpython2.7 -lmpi
BUILD=build
OBJ=$(BUILD)/interface.o $(BUILD)/kernel.o $(BUILD)/code.o
TARGET=interface.so

all:
	python2 setup.py build_ext --inplace
python3:
	python3 setup.py build_ext --inplace

$(BUILD):
	mkdir $(BUILD)

$(BUILD)/%.o: %.cpp 
	$(CC) -x cu $(CFLAGS) $(INCDIRS) $^ -c -o $@

gpu: $(BUILD) $(OBJ)
	$(CC) $(LDFLAGS) $(LIBDIRS) $(LIBS) $(OBJ) -o $(TARGET)
