wget https://nvidia.github.io/nvidia-docker/gpgkey --no-check-certificate
sudo apt-key add gpgkey
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

read -p "This installation requires a reboot, reboot now? (yes/no): " choice

# Convert the input to lowercase for case-insensitive comparison
choice_lower=$(echo "$choice" | tr '[:upper:]' '[:lower:]')

# Check the user's choice
if [ "$choice_lower" == "yes" ]; then
    echo "Rebooting the system..."
    sleep 3  # Optional delay for user to see the message
    sudo reboot
elif [ "$choice_lower" == "no" ]; then
    echo "Exiting without rebooting."
else
    echo "Invalid choice. Please enter 'yes' or 'no'."
fi
