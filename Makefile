.PHONY: all build build-rust gen-ffi test clean benchmark run-example

# Default task: rebuild everything and run tests
all: build test

# Full rebuild: Rust + FFI bindings
build: build-rust gen-ffi

# Build Rust library in release mode
build-rust:
	cd native && cargo build --release

# Generate Dart FFI bindings
gen-ffi:
	dart run ffigen

# Run all tests
test:
	dart test

# Run benchmarks
benchmark:
	@echo "Running Benchmarks for Potion Models..."
	@BENCH_MODEL=minishlab/potion-base-2M dart run benchmark/embedding_benchmark.dart
	@BENCH_MODEL=minishlab/potion-base-8M dart run benchmark/embedding_benchmark.dart
	@BENCH_MODEL=minishlab/potion-multilingual-128M dart run benchmark/embedding_benchmark.dart

# Clean build artifacts
clean:
	cd native && cargo clean
	rm -rf .dart_tool/native_assets.yaml

# Helper for quick example run
run-example:
	dart run example/main.dart
