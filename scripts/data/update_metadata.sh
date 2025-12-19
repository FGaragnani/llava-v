#!/bin/bash

# Check if the folder path argument is provided
if [ -z "$1" ]; then
        echo "Usage: $0 <folder_path>"
        exit 1
fi

# Use the first argument as the folder path
folder_path="$1"

# Validate if the provided path is a valid directory
if [ ! -d "$folder_path" ]; then
        echo "Error: $folder_path is not a valid directory."
        exit 1
fi

# Get the current date and time for updating file timestamps
new_date=$(date +"%Y-%m-%d %H:%M:%S")

# Log the initial time and folder
start_time=$(date +"%s")
echo "Script started at: $(date +"%Y-%m-%d %H:%M:%S")"
echo "Working on folder: $folder_path"

# Initialize file and error counters
file_count=0
symlink_count=0
error_count=0

# Find all files recursively and update their timestamps, log only errors
while IFS= read -r -d '' file; do
    if [ -L "$file" ]; then
        # Handle symlinks - use -h flag to update symlink itself, not target
        if touch -h "$file"; then
            symlink_count=$((symlink_count + 1))
        else
            echo "Error updating symlink $file"
            error_count=$((error_count + 1))
        fi
    fi
    if touch "$file"; then
        file_count=$((file_count + 1))
    else
        echo "Error updating $file"  # Only log errors
        error_count=$((error_count + 1))
    fi
done < <(find "$folder_path" -print0)

# Log the finish time, folder, and elapsed time
finish_time=$(date +"%s")
elapsed_time=$((finish_time - start_time))

echo "Script finished at: $(date +"%Y-%m-%d %H:%M:%S")"
echo "Processed $file_count files and $symlink_count symlinks in folder: $folder_path"
if (( error_count > 0 )); then
    echo "Encountered $error_count errors during processing."
fi
echo "Elapsed time: $elapsed_time seconds"
echo "All data updated"
