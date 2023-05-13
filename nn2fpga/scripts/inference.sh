if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    exit
fi

if [ $1 = "kria" ]; then
    user="ubuntu"
    path="/home/ubuntu/"
    ip="192.168.99.160"
    device="kria160"
elif [ $1 = "ultra96" ]; then
    user="root"
    path="/home/xilinx/"
    ip="192.168.3.1"
    device="${user}@${ip}"
else
    echo "kria or ultra96"
    exit
fi

res_file=results_${1}_$(date +%m%d%H%M).csv
#echo "exec_time_avg,exec_time_std,power_avg,power_std,energy_avg,energy_std" > ${res_file}

#project=subiso_bd

if [ -d overlay ]; then
    rm overlay -r
fi

mkdir overlay
cp bitstream_${1}/design_1_wrapper.bit overlay/design_1.bit
cp bitstream_${1}/design_1.hwh overlay/design_1.hwh
cp ./host.py overlay/host.py

cp ./test.txt overlay/test.txt

mkdir overlay/data
cp ./dataset/benchmark/labelled/email-EnronRM.csv overlay/data/
cp ./dataset/benchmark/labelled/musae_githubRM.csv overlay/data/
cp ./dataset/benchmark/labelled/musae_facebookRM.csv overlay/data/
cp ./dataset/benchmark/labelled/twitter_combinedRM.csv overlay/data/
cp ./dataset/benchmark/queries/query0RM.csv overlay/data/
cp ./dataset/benchmark/queries/query1RM.csv overlay/data/
cp ./dataset/benchmark/queries/query2RM.csv overlay/data/
cp ./dataset/benchmark/queries/query3RM.csv overlay/data/
cp ./dataset/benchmark/queries/query4RM.csv overlay/data/
cp ./dataset/benchmark/queries/query5RM.csv overlay/data/

# upload bitstream to sdcard
# scp -r overlay root@${ip}:/home/xilinx/
scp -r overlay ${device}:${path}

# execute kernel
#cat ./host.py | ssh root@192.168.3.1 'python3 -'
ssh ${device} "source /etc/profile && python3 ${path}overlay/host.py ${path}overlay/"
#>> ${res_file}

# cleanup
scp ${device}:${path}overlay/results.txt ./${res_file}
ssh ${device} "rm -r ${path}overlay"
rm -r overlay
