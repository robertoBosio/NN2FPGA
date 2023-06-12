#include <ap_utils.h>
#include <unistd.h>

#include "common/xf_headers.hpp"
#include "aie/imgproc/xf_resize_config.hpp"

#include "../../test/tb/cifar10/include/cifar/cifar10_reader.hpp"
#include "../../nn2fpga/cc/include/nn2fpga/debug.h"
#include "../../work/cc/include/network_sim.h"
#include "hls_stream.h"
/* #include "MemoryWeights.hpp" */
/* #include "../src/MemoryManagement.hpp" */
char *getcwd(char *buf, size_t size);
#define READ_WIDTH 8
#define READ_BYTES 1

int main() {
    hls::stream<t_inp_1> i_data;
    /* t_o_data o_data_exp[12]; */
    hls::stream<t_o_data> o_data_sim;
    #pragma HLS interface axis port = i_data
    #pragma HLS interface axis port = o_data_sim

    const int c_par = c_inp_1 / 8;
    const int c_index =
        (c_produce_stream_ich * c_produce_stream_ih * c_produce_stream_iw) / c_par;
    const int c_labels = 1;

    char cwd[100];
    std::cout << "CURRENT WORKING DIRECTORY" << std::endl;
    std::cout << getcwd(cwd, sizeof(cwd)) << std::endl;
    auto dataset = cifar::read_dataset<std::vector, std::vector, char, char>();

    /* for (int i = 0; i < dataset.test_images.size(); i++) { */
    /* 	std::cout << dataset.test_images.at(i) << ' '; */
    /* } */
    /* const int c_batch = dataset.test_images.size(); */
    const int c_batch = 3;
    const int n_bytes = c_index * c_par;
    std::cout << "SENDING " << c_batch << " IMAGES" << std::endl;
    std::cout << "SENDING " << n_bytes << " BYTES" << std::endl;

    int s_batch = 0;
    int results[c_batch];

    int s_bytes = 0;
    cv::Mat img;
    cv::Mat result_ocv;
    cv::Mat error;
    img = cv::imread("000001.jpg",1);

    result_ocv.create(cv::Size(640, 640),CV_8UC3);
    error.create(cv::Size(640, 640),CV_8UC3);
    cv::resize(img,result_ocv,cv::Size(640,640),0,0,CV_INTER_LINEAR);

    // Iterate over elements of result_ocv per channel
    for (int i = 0; i < result_ocv.rows; i++) {
        for (int j = 0; j < result_ocv.cols; j++) {
            cv::Vec3b pixel = result_ocv.at<cv::Vec3b>(i, j);
            // Iterate over channels (typically BGR)
            // Iterate over channells on RGB
            for (int c = 2; c > -1; c--) {
                //std::cout << (int)pixel[c] << std::endl;
                t_inp_1 s_data;
                int s_par = (s_bytes % c_par);
                s_data.data(8 * (s_par + 1) - 1, 8 * s_par) = (ap_uint<8>)(pixel[c]);
                s_data.keep = -1;

                #ifdef DEBUG
                std::cout << (ap_uint<8>)(pixel[c]) << " ";
                #endif

                if (s_bytes == (n_bytes - 1))
                    s_data.last = true;
                else
                    s_data.last = false;
                if (s_par == (c_par - 1)) {
                    i_data.write(s_data);
                }

                s_bytes++;
                if (s_bytes == n_bytes) break;
            }
        }
    }
    ///////////////////////// KERNEL EXECUTION ON IMAGE ///////////////////////

    /* std::cout << "--------------------- KERNEL -----------------------" <<
        * "\n"; */
    networkSim(i_data,
            o_data_sim);

    std::cout << "" << std::endl;
    t_o_data s_o_data;
    s_o_data = o_data_sim.read();
    int32_t max_value = -1;
    max_value = (int32_t)(s_o_data.data);
    std::cout << (int32_t)(s_o_data.data) << std::endl;
    int max_index = 0;
    int s_index = 1;

    do {
        s_o_data = o_data_sim.read();
        std::cout << (int32_t)(s_o_data.data) << std::endl;
        if ((int32_t)(s_o_data.data) > max_value) {
        max_value = (int32_t)(s_o_data.data);
        /* std::cout << "INDEX " << s_index << std::endl; */
        /* std::cout << "MAX VALUE " << (int32_t)(max_value) << std::endl; */
        max_index = s_index;
        }
        s_index++;
    } while (!s_o_data.last);
    results[s_batch] = max_index;

    s_batch++;
    if (s_batch == c_batch) break;

    const int n_bytes_labels = c_batch;

    /* while(o_last == 0); */
    /* while(o_last == 1); */

    /* std::cout << "EXP: " << o_data_exp[0] << "\n"; */

    return 0;
}

