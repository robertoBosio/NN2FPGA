#ifndef NN2FPGA_CONV_H_
#define NN2FPGA_CONV_H_

// Vitis HLS dependencies.
#include <ap_int.h>

template <typename din_t, typename weight_t, typename dout_t, typename acc_t,
          unsigned ICH, unsigned OCH, unsigned IW, unsigned IH, unsigned OW,
          unsigned OH, unsigned FW, unsigned FH, unsigned PAD, unsigned STRIDE>
void Conv(din_t* din, weight_t* win, dout_t* dout) {
  unsigned s_oh_pad = 0;
  unsigned s_ow_pad = 0;

  if (PAD == 1) {
    s_oh_pad = FH - 1;
    s_ow_pad = FW - 1;
  }

  for (unsigned s_och = 0; s_och < OCH; s_och++) {
    for (unsigned s_oh = 0; s_oh < OH; s_oh++) {
      for (unsigned s_ow = 0; s_ow < OW; s_ow++) {
        unsigned s_o_index = s_oh * OW * OCH + s_ow * OCH + s_och;
        dout[s_o_index] = 0;
        acc_t s_acc = 0;
        for (unsigned s_ich = 0; s_ich < ICH; s_ich++) {
          for (unsigned s_fh = 0; s_fh < FH; s_fh++) {
            for (unsigned s_fw = 0; s_fw < FW; s_fw++) {
              unsigned s_ih = s_oh + s_fh - s_oh_pad / 2;
              unsigned s_iw = s_ow + s_fw - s_ow_pad / 2;

              unsigned s_i_index = s_ih * IW * ICH + s_iw * ICH + s_ich;
              unsigned s_w_index =
                  s_ich * FH * FW * OCH + s_fh * FW * OCH + s_fw * OCH + s_och;

              if ((s_ih >= 0) & (s_iw >= 0) & (s_ih < c_ih) & (s_iw < c_iw)) {
                /* std::cout << (unsigned int)(din[s_i_index] & 0xff) << " "
                 * << (unsigned int)(win[s_w_index] & 0xff) << "\n"; */
                s_acc += din[s_i_index] * win[s_w_index];
              }
            }
          }
        }
        /* std::cout << "\n"; */
        dout[s_o_index] = (dout_t)(s_acc);
      }
    }
  }
}

#endif  // NN2FPGA_CONV_H_
