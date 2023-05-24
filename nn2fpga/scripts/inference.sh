if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    exit
fi

if [ $1 = "kria" ]; then
    user="root"
    path="/home/ubuntu/"
    ip="192.168.99.160"
    device="kria160"
		export prj_name="KRIA"
elif [ $1 = "ultra96" ]; then
    user="root"
    path="~/"
    # ip="192.168.1.63"
    # ip="192.168.1.84"
    ip="192.168.3.1"
    device="${user}@${ip}"
		export prj_name="ULTRA96v2"
else
    echo "kria or ultra96"
    exit
fi

res_file=results_${1}_$(date +%m%d%H%M).csv
#echo "exec_time_avg,exec_time_std,power_avg,power_std,energy_avg,energy_std" > ${res_file}
scp_folder="workspace/NN2FPGA/"

if [ -d overlay ]; then
    rm overlay -r
fi

mkdir overlay
./inference/copy_bitstream.sh
cp ./tmp/design_1.bit overlay/design_1.bit
cp ./tmp/design_1.hwh overlay/design_1.hwh
cp ./inference/inference.py overlay/inference.py
cp ./inference/boards.py overlay/boards.py
cp ./inference/datasets.py overlay/datasets.py

# upload bitstream to sdcard
# scp -r overlay root@${ip}:/home/xilinx/
scp -r overlay ${device}:${path}

# execute kernel
#cat ./host.py | ssh root@192.168.3.1 'python3 -'
ssh ${device} "cd ${path} && source /etc/profile && python3 ${path}overlay/inference.py $1 cifar10"

# cleanup
scp ${device}:${path}overlay/results.txt ./${res_file}
if [ $1 = "kria" ]; then
		ssh ${device} "rm -r ${path}overlay && xmutil unloadapp k26-starter-kits && xmutil loadapp k26-starter-kits"
fi
