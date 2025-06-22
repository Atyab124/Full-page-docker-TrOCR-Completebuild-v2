sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
apt-cache policy docker-ce
sudo apt install docker-ce
sudo usermod -aG docker ${USER}
su - ${USER}
sudo usermod -aG docker username

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
