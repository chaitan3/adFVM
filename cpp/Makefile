all: 
	python2 setup.py build

install: 
	python2 setup.py install

clean:
	rm -rf build

#CC = g++
##CC = clang
#
#CFLAGS = -std=c++11 -Wno-write-strings -O0 -g 
##CFLAGS = -std=c++11 -Wno-write-strings -DNDEBUG -O2 -msse2  #-fno-inline
#LFLAGS = 
#INCLUDE = -I/usr/include/eigen3/ -I/usr/include/python2.7/
#LIBS = -lpython2.7 #-larmadillo
#
#TARGET = rcf.exe
#SOLVER = cppRCF.o
#OBJ = interface.o field.o interp.o op.o $(SOLVER)
#
#ERRORS = errors.out
#
#all: $(TARGET)
#
#$(TARGET): $(OBJ)
#	$(CC) $(LFLAGS) $(OBJ) -o $@ $(LIBS)
#
#%.o: %.cpp
#	$(CC) $(CFLAGS) $(INCLUDE) -c $^
#clean:
#	rm -f $(OBJ) $(TARGET)
#memtest:
#	valgrind --tool=memcheck --leak-check=full $(TARGET) > $(ERRORS) 2>&1
#profile:
#valgrind --tool=callgrind --instr-atstart=no ./$(TARGET)
