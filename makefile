# Variables
PYTHON=python3
CODE=model_code/code.py

# Default target
all: model

# Run the Python script
run-model:
	$(PYTHON) $(CODE)

# Install dependencies
install:
	pip install -r requirements.txt

# Clean generated files
clean:
	rm -f model/*.pkl
