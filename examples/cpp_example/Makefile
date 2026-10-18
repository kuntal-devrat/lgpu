CXX ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O2
INCLUDES = -I../../crates/lgpu-c/include
LIB_DIR = ../../target/release
LIBS = -L$(LIB_DIR) -llgpu_c -Wl,-rpath,$(LIB_DIR)

TARGET = cpp_example

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) main.cpp $(INCLUDES) $(LIBS) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
