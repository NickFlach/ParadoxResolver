#!/bin/bash
# Crypto_ParadoxOS CLI Wrapper

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    echo "Please install Python 3.8 or newer."
    exit 1
fi

# Check for dependencies
python3 -c "import numpy" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Installing required dependencies..."
    pip install numpy matplotlib
fi

# Execute the CLI with all arguments passed to this script
python3 paradox_cli.py "$@"