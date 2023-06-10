#ifndef NON_MAX_SUPPRESSION_H
#define NON_MAX_SUPPRESSION_H

#include "ap_int.h"
#include "hls_stream.h"

namespace nn2fpga {

// Code for YOLOv5 detect layer
template <typename t_data_struct, typename t_data, typename t_box_struct,
          typename t_box, int TOTAL, int OCH,
          int SPLIT, int NL>
          void non_max_suppression(
            hls::stream<t_data_struct> &i_data,
            const ap_ufixed<8,8> conf_th,
            hls::stream<t_box_struct> &o_data
          ) {

    // Rewriting the following code in C++ HLS

    bool detected;
    ap_ufixed<8,8> detected_class;
    auto detected_class_index = 0;
    t_box box;

    t_data s_buffer;
    bool din_last;
    // Loop over input width IW and height IH
    for (auto k = 0; k < TOTAL; k++) {
        for (auto j = 0; j < OCH; j++) {

            #pragma HLS pipeline II=1
            // Adding non-max suppression
            if (j == 0) {
                detected = true;
                // Reading input data
                auto din_read = i_data.read();
                s_buffer = din_read.data;
                din_last = din_read.last;
                detected_class = 0;
            }


            if (j < 4) {
                box[j] = s_buffer[j];
            }

            if ((j == SPLIT*2) & (s_buffer[j] < conf_th)) {
                detected = false;
            }

            if ((j > SPLIT*2) & (detected == true) & s_buffer[j] > detected_class) {
                detected_class = s_buffer[j];
                detected_class_index = j - SPLIT*2;
            }

            if ((j == OCH - 1) & (detected == true))  {
                if (detected_class > conf_th) {
                    box[5] = detected_class;
                    t_box_struct dout_write;
                    dout_write.data = box;
                    dout_write.last = din_last;
                    o_data.write(dout_write);
                }
            }
            // Write output data to the output stream composing the t_output_struct
        }
    }

}
    
} // namespace nn2fpga

#endif // NON_MAX_SUPPRESSION_H