ZIG ?= $(shell which zig 2>/dev/null || echo "$$HOME/.local/bin/zig")

TARGET = zig_example

all: $(TARGET)

$(TARGET): main.zig
	$(ZIG) build-exe main.zig -I../../crates/lgpu-c/include -L../../target/release -llgpu_c -lc -O ReleaseFast
	@if [ -f main ]; then mv main $(TARGET); fi

run: $(TARGET)
	LD_LIBRARY_PATH=../../target/release ./$(TARGET)

clean:
	rm -f $(TARGET) main main.o

.PHONY: all run clean
