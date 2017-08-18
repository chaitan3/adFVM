CC=nvcc
#NV_ARCH=-gencode=arch=compute_35,code=\"sm_37,compute_35\"
NV_ARCH=-gencode=arch=compute_52,code=\"sm_52,compute_52\"
CFLAGS=-std=c++11 -Xcompiler="-fPIC" -O3 -g -DGPU --ptxas-options=-v $(NVARCH) 
LDFLAGS=-shared
INCDIRS=-I/usr/include/python2.7 -I/usr/lib/python2.7/site-packages/numpy/core/include -I/home/talnikar/adFVM/adFVM/cpp/include -I/home/talnikar/.local/include/
LIBDIRS=-L/home/talnikar/.local/lib
LIBS=-lpython2.7 -lmpi -llapack
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

test: test.cpp
	$(CC) -x cu $(CFLAGS) $(INCDIRS) $(LIBDIRS) $(LIBS) $^  -o $@

clean: $(BUILD) $(TARGET)

gpu: $(TARGET)

