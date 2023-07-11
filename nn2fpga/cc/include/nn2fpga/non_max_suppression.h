#ifndef NON_MAX_SUPPRESSION_H
#define NON_MAX_SUPPRESSION_H

#include "ap_int.h"
#include "hls_stream.h"

namespace nn2fpga {

// Code for YOLOv5 detect layer
template <typename t_data_struct, typename t_data,
          typename t_box, int OCH, int TOTAL,
          int SPLIT, int NL>
          void consume_stream(
            hls::stream<t_data_struct> &i_data,
            const ap_fixed<32,16> conf_th,
            hls::stream<t_box> &o_data
          ) {

    // Rewriting the following code in C++ HLS

    bool detected;
    ap_fixed<32,16> detected_class;
    auto detected_class_index = 0;
    t_data box;

    t_data s_buffer;
    bool din_last;
    // Loop over input width IW and height IH
    for (auto k = 0; k < TOTAL; k++) {
        for (auto j = 0; j < OCH; j++) {

            #pragma HLS pipeline style=stp
            // Adding non-max suppression
            if (j == 0) {

                detected = true;
                // Reading input data
                auto din_read = i_data.read();
                s_buffer = din_read.data;
                din_last = din_read.last;
                detected_class = 0;

            }

            if (j <= SPLIT*2) {
                box[j] = s_buffer[j];
            }

            if ((j > SPLIT*2) & s_buffer[j] > detected_class) {
                detected_class = s_buffer[j];
                detected_class_index = j - SPLIT*2;
            }

            if ((j == OCH - 1) & (detected == true))  {
                if ((detected_class)*(box[SPLIT*2]) > conf_th) {
                    box[SPLIT*2+1] = detected_class_index;
                    t_box dout_write;
                    for (auto i = 0; i < 6; i++) {
                        #pragma HLS unroll
                        dout_write.data[i] = box[i];
                    }
                    for (auto i = 6; i < 8; i++) {
                        #pragma HLS unroll
                        dout_write.data[i] = 0xffff;
                    }
                    dout_write.last = false;
                    dout_write.keep = -1;
                    o_data.write(dout_write);
                }
            }
            // Write output data to the output stream composing the t_output_struct
        }
    }

    t_box dout_write;
    for (auto i = 0; i < 8; i++) {
        #pragma HLS unroll
        dout_write.data[i] = 0xffff;
    }
    dout_write.last = true;
    dout_write.keep = -1;
    o_data.write(dout_write);

}
    
} // namespace nn2fpga

#endif // NON_MAX_SUPPRESSION_H