# Variables
PYTHON=python3
CODE=model/train.py

# Default target
all: run-model

run-app:
	$(PYTHON) app.py

# Run the Python script
run-model:
	$(PYTHON) $(CODE)

# Install dependencies
install:
	pip install -r requirements.txt

# Clean generated files
clean:
	rm -f model/*.pkl
