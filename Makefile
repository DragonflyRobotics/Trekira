# Makefile

# The name of the executable
EXEC = Staxy

# Default target
all: run 

# Run the project
run:
		python3 test.py

.PHONY: all build run clean
