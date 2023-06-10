#ifndef NN2FPGA_ACTIVATIONS_UTILS_H_
#define NN2FPGA_ACTIVATIONS_UTILS_H_

#include "ap_int.h"
#include "hls_stream.h"

namespace nn2fpga {

// Code for YOLOv5 detect layer
template <typename t_data_struct, typename t_data,
          int ICH, int OCH, int IH, int IW,
          int SPLIT>
          void detect(
            hls::stream<t_data_struct> &i_data, 
            hls::stream<t_data_struct> &o_data
        ) {

    // Rewriting the following code in C++ HLS

    // Loop over input width IW and height IH
    for (auto k = 0; k < IH; k++) {
        for (auto l = 0; l < IW; l++) {
            for (auto h = 0; h < ICH; h++) {
                for (auto j = 0; j < OCH; j++) {
                    #pragma HLS pipeline II=1
                    // Reading input data
                    auto din_read = i_data.read();
                    auto din_last = din_read.last;
                    auto detect_addr = 2; 
                    if (j < SPLIT)
                        detect_addr = 0;
                    if (j >= SPLIT && j < SPLIT*2)
                        detect_addr = 1;

                    auto dout_data = detect_lut[din_read.data][detect_addr];

                    // If j less than SPLIT the compute the xy variable
                    if (j < SPLIT) {
                        auto addr = (j == 0) ? l : k;
                        // Compute the xy variable
                        dout_data += grid[addr];
                        dout_data *= stride[i];
                    }

                    // If j greater than SPLIT and lower then SPLIT*2 compute the wh variable
                    if (j >= SPLIT && j < SPLIT*2) {
                        // Compute the wh variable
                        auto addr = j - SPLIT;
                        // Compute the xy variable
                        dout_data *= anchor_grid[h][addr];
                    }

                    // Write output data to the output stream composing the t_output_struct
                    t_data_struct dout_write;
                    dout_write.data = dout_data;
                    dout_write.last = din_last;
                    dout_write.keep = 0xff;
                    o_data.write(dout_write);
                }
            }
        }
    }

}


// Code for YOLOv5 detect layer
template <typename t_data_struct, typename t_data, typename t_box_struct,
          typename t_box, int ICH, int OCH, int IH, int IW,
          int SPLIT, int NL, ap_ufixed<8,8> CONF_TH>
          void non_max_suppression(hls::stream<t_data_struct> *i_data, hls::stream<t_data_struct> &o_data) {

    // Rewriting the following code in C++ HLS

    bool detected;
    ap_ufixed<8,8> detected_class;
    auto detected_class_index = 0;
    t_box box;
    // Loop over input width IW and height IH
    for (auto k = 0; k < IH; k++) {
        for (auto l = 0; l < IW; l++) {
            for (auto i = 0; i < NL; i++) {
                for (auto h = 0; h < ICH; h++) {
                    for (auto j = 0; j < OCH; j++) {
                        #pragma HLS pipeline II=1
                        // Adding non-max suppression
                        if (j == 0) {
                            detected = true;
                            detected_class = 0;
                        }

                        // Reading input data
                        auto din_read = din[i].read();
                        auto din_last = din_read.last;

                        if (j < 4) {
                            box[j] = din_read.data;
                        }

                        if ((j == SPLIT*2) & (dout_data < CONF_TH)) {
                            detected = false;
                        }

                        if ((j > SPLIT*2) & (detected == true) & dout_data > detected_class) {
                            detected_class = dout_data;
                            detected_class_index = j - SPLIT*2;
                        }

                        if ((j == OCH - 1) & (detected == true))  {
                            if (detected_class > CONF_TH) {
                                box[5] = detected_class;
                                t_box_struct dout_write;
                                dout_write.data = box;
                                dout_write.last = din_last;
                                dout_write.keep = 0xff;
                                o_data.write(dout_write);
                            }
                        }
                        // Write output data to the output stream composing the t_output_struct
                    }
                }
            }
        }
    }

}


} // namespace nn2fpga