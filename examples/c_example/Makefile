CC ?= gcc
CFLAGS ?= -std=c99 -Wall -Wextra -O2
INCLUDES = -I../../crates/lgpu-c/include
LIB_DIR = ../../target/release
LIBS = -L$(LIB_DIR) -llgpu_c -Wl,-rpath,$(LIB_DIR)

TARGET = c_example

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) main.c $(INCLUDES) $(LIBS) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
