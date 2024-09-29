<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" name="yolo_U280_ip" top="yolo">
    <solutions>
        <solution name="solution_nopack" status=""/>
    </solutions>
    <files>
        <file name="/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/work/project_yolo_U280_20240928/npy" sc="0" tb="1" cflags="-Wno-unknown-pragmas" csimflags="" blackbox="false"/>
        <file name="/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/tb/../py/utils" sc="0" tb="1" cflags="-Wno-unknown-pragmas" csimflags="" blackbox="false"/>
        <file name="/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/tb/coco/preprocess.py" sc="0" tb="1" cflags="-Wno-unknown-pragmas" csimflags="" blackbox="false"/>
        <file name="/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/tb/coco/network_tb.cc" sc="0" tb="1" cflags="-DCSIM -I../../cc/include -I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/nn2fpga/cc/include -I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/tb/common/cmdparser -I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/tb/common/logger -I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/work/project_yolo_U280_20240928/Vitis_Libraries/vision/L1/include/. -Wno-unknown-pragmas" csimflags="" blackbox="false"/>
        <file name="/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/tb/common/logger/logger.cpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/tb/common/cmdparser/cmdlineparser.cpp" sc="0" tb="false" cflags="-I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/models/tb/common/logger/." csimflags="" blackbox="false"/>
        <file name="cc/src/yolo.cc" sc="0" tb="false" cflags="-Icc/include -I/home-ssd/teodoro/Github/work0/NNtwoFPGA_ROBERTO/NN2FPGA/nn2fpga/cc/include" csimflags="" blackbox="false"/>
    </files>
</AutoPilot:project>

