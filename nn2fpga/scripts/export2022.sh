#!/bin/bash 

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 DIR_PROJ TOP_NAME"
    exit 1
fi

# Define local arrays for names and paths
projects = ("mobilenet_v2" "mm2s_activations" "mm2s_weights" "s2mm_outputs")
paths=("" "/path/to/dir2" "/path/to/dir3" "/path/to/dir4")

# Check if arrays have the same length
if [ "${#names[@]}" -ne "${#paths[@]}" ]; then
    echo "Error: Number of names and paths must be the same."
    exit 1
fi

# Loop through the arrays
for ((i = 0; i < ${#names[@]}; i++)); do
    name="${names[i]}"
    path="${paths[i]}"

    # Navigate to the specified path
    cd "$path" || exit 1

    # Launch the program (replace "your_program" with the actual program)
    your_program

    # Copy the result two directories above
    cp result_directory/* ../..

    # Print a message for each iteration
    echo "Processed $name at $path"
done

echo "Script completed successfully."
