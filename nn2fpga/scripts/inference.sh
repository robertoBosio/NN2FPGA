if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    exit
fi

if [ $1 = "KRIA" ]; then
    user="root"
    path="/home/ubuntu/"
    ip="kriahlslab0"
    device="${user}@kriahlslab0"
elif [ $1 = "ULTRA96v2" ]; then
    user="root"
    path="~/"
    # ip="192.168.1.63"
    # ip="192.168.1.84"
    ip="192.168.3.1"
    device="${user}@${ip}"
else
    echo "KRIA or ULTRA96v2"
    exit
fi

prj_name="$1_${TOP_NAME}"
res_file=results_${1}_$(date +%m%d%H%M).csv
#echo "exec_time_avg,exec_time_std,power_avg,power_std,energy_avg,energy_std" > ${res_file}

if [ -d overlay ]; then
    rm ${PRJ_ROOT}/overlay -r
fi

mkdir -p ${PRJ_ROOT}/overlay
scp ${PRJ_ROOT}/${prj_name}_example/${prj_name}_example.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh ${PRJ_ROOT}/overlay/design_1.hwh
scp ${PRJ_ROOT}/${prj_name}_example/design_1.bit ${PRJ_ROOT}/overlay/design_1.bit
cp ${NN2FPGA_ROOT}/scripts/inference.py ${PRJ_ROOT}/overlay/inference.py
cp ${NN2FPGA_ROOT}/scripts/boards.py ${PRJ_ROOT}/overlay/boards.py
cp ${NN2FPGA_ROOT}/scripts/datasets.py ${PRJ_ROOT}/overlay/datasets.py
cp ${NN2FPGA_ROOT}/scripts/coco.py ${PRJ_ROOT}/overlay/coco.py
cp ${NN2FPGA_ROOT}/scripts/cifar10.py ${PRJ_ROOT}/overlay/cifar10.py
if [ ${URAM_STORAGE} = 1 ]; then
    cp ${PRJ_ROOT}/npy/uram_${TOP_NAME}.npy ${PRJ_ROOT}/overlay/uram.npy
fi

# upload bitstream to sdcard
# scp -r overlay root@${ip}:/home/xilinx/
scp -r ${PRJ_ROOT}/overlay ${device}:${path}

# execute kernel
#cat ./host.py | ssh root@192.168.3.1 'python3 -'
ssh ${device} "cd ${path} && source /etc/profile && python3 ${path}overlay/inference.py $1 cifar10 ${URAM_STORAGE}"

# cleanup
scp ${device}:${path}overlay/results.txt ${PRJ_ROOT}/${res_file}
if [ $1 = "KRIA" ]; then
		ssh ${device} "rm -r ${path}overlay && xmutil unloadapp k26-starter-kits && xmutil loadapp k26-starter-kits"
fi
