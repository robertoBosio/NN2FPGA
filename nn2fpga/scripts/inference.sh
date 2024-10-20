if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    exit
fi

if [ $1 = "KRIA" ]; then
    user="root"
    path="/home/ubuntu/"
    ip="kriahlslab1"
    device="${user}@kriahlslab1"
elif [ $1 = "ZCU102" ]; then
    user="root"
    path="/home/ubuntu/"
    ip="192.168.166.168"
    device="${user}@192.168.166.168"
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
    rm ${PRJ_FULL_ROOT}/overlay -r
fi

mkdir -p ${PRJ_FULL_ROOT}/overlay
scp ${PRJ_FULL_ROOT}/${prj_name}_example/${prj_name}_example.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh ${PRJ_FULL_ROOT}/overlay/design_1.hwh
scp ${PRJ_FULL_ROOT}/${prj_name}_example/design_1.bit ${PRJ_FULL_ROOT}/overlay/design_1.bit
# cp /home-ssd/roberto/Documents/nn2fpga-container/NN2FPGA/work/notable_bitstream/mobilenet_ZCU102_20240313/design_1.bit ${PRJ_FULL_ROOT}/overlay/design_1.bit
# cp /home-ssd/roberto/Documents/nn2fpga-container/NN2FPGA/work/notable_bitstream/mobilenet_ZCU102_20240313/design_1.hwh ${PRJ_FULL_ROOT}/overlay/design_1.hwh
cp ${NN2FPGA_ROOT}/scripts/inference.py ${PRJ_FULL_ROOT}/overlay/inference.py
cp ${NN2FPGA_ROOT}/scripts/boards.py ${PRJ_FULL_ROOT}/overlay/boards.py
cp ${NN2FPGA_ROOT}/scripts/datasets.py ${PRJ_FULL_ROOT}/overlay/datasets.py
cp ${NN2FPGA_ROOT}/scripts/coco.py ${PRJ_FULL_ROOT}/overlay/coco.py
cp ${NN2FPGA_ROOT}/scripts/cifar10.py ${PRJ_FULL_ROOT}/overlay/cifar10.py
cp ${NN2FPGA_ROOT}/scripts/vw.py ${PRJ_FULL_ROOT}/overlay/vw.py
cp ${NN2FPGA_ROOT}/scripts/imagenet.py ${PRJ_FULL_ROOT}/overlay/imagenet.py

if [ ${DYNAMIC_INIT} = 1 ]; then
    cp ${PRJ_FULL_ROOT}/npy/uram_${TOP_NAME}.npy ${PRJ_FULL_ROOT}/overlay/uram.npy
	# cp /home-ssd/roberto/Documents/nn2fpga-container/NN2FPGA/work/notable_bitstream/mobilenet_ZCU102_20240313/parameters.npy ${PRJ_FULL_ROOT}/overlay/uram.npy
fi

# upload bitstream to sdcard
# scp -r overlay root@${ip}:/home/xilinx/
scp -r ${PRJ_FULL_ROOT}/overlay ${device}:${path}

# execute kernel
#cat ./host.py | ssh root@192.168.3.1 'python3 -'
ssh ${device} "cd ${path} && source /etc/profile && python3 ${path}overlay/inference.py $1 $2 32 100"

# cleanup
scp ${device}:${path}overlay/results.txt ${PRJ_FULL_ROOT}/${res_file}
if [ $1 = "KRIA" ]; then
		ssh ${device} "rm -r ${path}overlay && xmutil unloadapp k26-starter-kits && xmutil loadapp k26-starter-kits"
fi
