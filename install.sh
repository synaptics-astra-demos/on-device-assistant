#!/bin/bash

# Define ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print messages in color
function print_message() {
    echo -e "${1}${2}${NC}"
}

# Save the current directory
CURRENT_DIR=$(pwd)

# Check if .venv directory exists
if [ ! -d ".venv" ]; then
    print_message $GREEN "Virtual environment not found. Creating one..."
    python3 -m venv .venv || { print_message $RED "Failed to create virtual environment. Exiting."; exit 1; }
fi

# Activate the virtual environment
source .venv/bin/activate || { print_message $RED "Failed to activate virtual environment. Exiting."; exit 1; }

# Install required Python packages, using the path from $CURRENT_DIR
if [ -f "$CURRENT_DIR/requirements.txt" ]; then
    pip install -r "$CURRENT_DIR/requirements.txt" || { print_message $RED "Failed to install Python packages. Exiting."; exit 1; }
else
    print_message $RED "requirements.txt not found in $CURRENT_DIR. Exiting."
    exit 1
fi

# Prompt the user
print_message $GREEN "\nInstallation complete."

# Print completion message
print_message $GREEN "Setup complete. Run the following commands to run demo:\n"
print_message $GREEN "source .venv/bin/activate\npython assistant.py"