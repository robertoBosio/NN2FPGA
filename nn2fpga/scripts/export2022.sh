#!/bin/bash 

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 DIR_PROJ TOP_NAME"
    exit 1
fi

# Define local arrays for names and paths
projects=("${2}" "mm2s_activations" "mm2s_weights" "s2mm_outputs")
kernels=("${2}.xo" "mm2s_a.xo" "mm2s_w.xo" "s2mm.xo")
paths=( "${2}/${2}" "mm2s_a/mm2s_activations" "mm2s_w/mm2s_weights" "s2mm/s2mm_outputs")

for ((i = 0; i < ${#paths[@]}; i++)); do
    path="$1/${paths[i]}"
    if [ ! -d "$path" ]; then
        echo "Path does not exists: ${path}"
        exit 1
    fi
done

cd ${1}

# Loop through the arrays
for ((i = 0; i < ${#projects[@]}; i++)); do
    path="${paths[i]}"
    project="${projects[i]}"

    # Navigate to the specified path
    cd "${path}"
    # echo "cd ${path}"

    # Launch the program (replace "your_program" with the actual program)
    # echo "vitis_hls -i -f ../../../../nn2fpga/tcl/export2022.tcl ${project}"
    vitis_hls -i -f ../../../../nn2fpga/tcl/export2022.tcl ${project}

    # Copy the result two directories above
    cp ./${project}/solution/impl/export.xo ../../${kernels[i]}

    cd ../../
    # echo "cp ./${project}/solution/impl/export.xo ../../${kernels[i]}"
done

echo "Script completed successfully."
