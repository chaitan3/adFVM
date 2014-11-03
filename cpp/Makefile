CC = g++
TARGET = ./test.exe
CFLAGS = -Wno-write-strings -O2 -g -std=c++11
LFLAGS = 
INCLUDE = -I/usr/include/eigen3/ -I/usr/include/python2.7/
LIBS = -lpython2.7 #-larmadillo

#OBJ = arma.o
OBJ = interface.o test.o field.o interp.o op.o

ERRORS = errors.out

all: $(TARGET)
%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDE) -c $^
$(TARGET): $(OBJ)
	$(CC) $(LFLAGS) $(OBJ) -o $@ $(LIBS)
clean:
	rm -f $(OBJ) $(TARGET)
memtest:
	valgrind --tool=memcheck --leak-check=full $(TARGET) > $(ERRORS) 2>&1
