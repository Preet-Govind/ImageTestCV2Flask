#!/bin/bash
# for linux , check conda and activate conda env , install project related deps

# Get a list of conda environments
env_list=$(conda env list | grep -v '#' | awk '{print $1}')

# Check if there are any environments
if [[ -z "$env_list" ]]; then
  echo "No conda environments found."
  exit 1
fi

# Display environment list
echo "Available conda environments:"
echo "$env_list"

# Prompt user to select an environment
read -p "Enter the name of the environment to activate (or 'q' to quit): " selected_env

# Check user input
if [[ "$selected_env" == "q" ]]; then
  exit 0
fi

if [[ -z "$selected_env" || ! [[ "$env_list" =~ "$selected_env" ]] ]]; then
  echo "Invalid environment name. Please choose from the list above or enter 'q' to quit."
  exit 1
fi

# Activate the selected environment
source activate "$selected_env"

# Check if activation was successful
if [[ $? -eq 0 ]]; then
  echo "Successfully activated environment: $selected_env"
  read -p "Are you sure you want to continue? (y/N) " response
  if [[ "$response" =~ ^[Yy]$ ]]; then
  # User confirmed, continue with the script
  echo "Continuing...installing associated libraries..."
  pip install flask , pymupdf 
  # Your script logic here
else
  # User declined, exit or perform alternative actions
  echo "Exiting as requested."
  exit 0
fi

else
  echo "Error: Failed to activate environment: $selected_env"
  exit 1
fi

