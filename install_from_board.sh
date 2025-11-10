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
    python3 -m venv .venv --system-site-packages || { print_message $RED "Failed to create virtual environment. Exiting."; exit 1; }
fi

# Activate the virtual environment
source .venv/bin/activate || { print_message $RED "Failed to activate virtual environment. Exiting."; exit 1; }

# Download and install python3-dev deb package
DEB_URL="https://synaptics-synap.github.io/examples-prebuilts/packages/python3-dev_3.10.13-r0_arm64.deb"
DEB_FILE="python3-dev_3.10.13-r0_arm64.deb"

print_message $YELLOW "Downloading python3-dev package..."
wget "$DEB_URL" -O "$DEB_FILE" || { print_message $RED "Failed to download $DEB_FILE. Exiting."; exit 1; }

print_message $YELLOW "Installing python3-dev package..."
dpkg -i "$DEB_FILE" || { print_message $RED "Failed to install $DEB_FILE. Exiting."; exit 1; }


# Install required Python packages, using the path from $CURRENT_DIR
if [ -f "$CURRENT_DIR/requirements.txt" ]; then
    pip install -r "$CURRENT_DIR/requirements.txt" || { print_message $RED "Failed to install Python packages. Exiting."; exit 1; }
else
    print_message $RED "requirements.txt not found in $CURRENT_DIR. Exiting."
    exit 1
fi

# Download models and pre-generate TTS
python initialize_models.py || { print_message $YELLOW "Failed to download models.";}

print_message $GREEN "\nInitialization complete."

# Add assistant as a start-up service
read -p "Do you want to install the On Device Assistant service to run on boot? (y/n) [default: n]: " user_input
user_input=${user_input:-n}

if [[ "$user_input" =~ ^[Yy]$ ]]; then
    echo "Installing the On Device Assistant service..."

    # Define paths
    SCRIPT_PATH="$CURRENT_DIR/assistant.py"
    SERVICE_PATH="/etc/systemd/system/on-device-ai-assistant.service"

    # Make sure the Python script is executable
    chmod +x "$SCRIPT_PATH"

    # Create the systemd service file
    echo "[Unit]
Description=On Device Assistant
After=network.target

[Service]
WorkingDirectory=$CURRENT_DIR
ExecStart=$CURRENT_DIR/.venv/bin/python3 $CURRENT_DIR/assistant.py
Restart=on-failure
User=root

[Install]
WantedBy=multi-user.target
" > "$SERVICE_PATH"
    chmod 644 "$SERVICE_PATH"
    systemctl daemon-reload
    systemctl enable on-device-ai-assistant.service
    systemctl start on-device-ai-assistant.service
    echo "Service has been installed. It will run on boot."
else
    echo "Service installation skipped."
fi

# Print completion message
print_message $GREEN "Setup complete. Run the following commands to start demo:\n"
print_message $BLUE "source .venv/bin/activate\npython assistant.py"