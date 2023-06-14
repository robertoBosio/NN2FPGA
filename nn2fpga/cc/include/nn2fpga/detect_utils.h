#ifndef NN2FPGA_DETECT_UTILS_H_
#define NN2FPGA_DETECT_UTILS_H_

#include "ap_int.h"
#include "hls_stream.h"

namespace nn2fpga {

// Code for YOLOv5 detect layer
template <typename t_input_struct, typename t_input,
          typename t_output_struct, typename t_output,
          typename t_detect_lut, typename t_grid_h, typename t_grid_w,
          typename t_anchor_grid, typename t_stride,
          int ICH, int OCH, int IH, int IW,
          int SPLIT>
          void detect(
            hls::stream<t_input_struct> &i_data, 
	        const t_detect_lut detect_lut[255][ICH],
	        const t_grid_h grid_h[1][IH],
	        const t_grid_w grid_w[1][IW],
	        const t_anchor_grid anchor_grid[1][ICH][2],
	        const t_stride stride,
            hls::stream<t_output_struct> &o_data
        ) {

    // Rewriting the following code in C++ HLS

    // Offset for an integer number depending on its number of bits
    const auto offset = (1 << (sizeof(t_input) * 8 - 1));

    t_output s_buffer;
    // Loop over input width IW and height IH
    for (auto k = 0; k < IH; k++) {
        for (auto l = 0; l < IW; l++) {
            for (auto h = 0; h < ICH; h++) {
                for (auto j = 0; j < OCH; j++) {
                    #pragma HLS pipeline II=1
                    // Reading input data
                    auto din_read = i_data.read();
                    unsigned int din_data = din_read.data + offset;
                    auto din_last = din_read.last;
                    auto detect_addr = 2; 
                    if (j < SPLIT)
                        detect_addr = 0;
                    if (j >= SPLIT && j < SPLIT*2)
                        detect_addr = 1;

                    ap_fixed<32, 16> dout_data = detect_lut[din_data][detect_addr];
                    // std::cout << ap_int<32>(din_data) << std::endl;

                    // If j less than SPLIT the compute the xy variable
                    if (j < SPLIT) {
                        // Compute the xy variable
                        dout_data += (j == 0) ? grid_w[l] : grid_h[k];
                        dout_data *= stride;
                    }

                    // If j greater than SPLIT and lower then SPLIT*2 compute the wh variable
                    if (j >= SPLIT && j < SPLIT*2) {
                        // Compute the wh variable
                        auto addr = j - SPLIT;
                        // Compute the xy variable
                        dout_data *= anchor_grid[0][h][addr];
                    }

                    // Write output data to the output stream composing the t_output_struct
                    s_buffer[j] = dout_data;
                    if (j == (OCH - 1)) {
                        t_output_struct dout_write;
                        dout_write.data = s_buffer;
                        dout_write.last = din_last;
                        o_data.write(dout_write);
                    }
                }
            }
        }
    }

}


} // namespace nn2fpga
#endif