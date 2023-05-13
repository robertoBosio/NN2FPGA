#ifndef __CONV__
#define __CONV__

#include <ap_int.h>

template <typename din_t, typename weight_t, typename dout_t, typename acc_t,
          int C_ICH, int C_OCH, int C_IW, int C_IH, int C_OW, int C_OH,
          int C_FW, int C_FH, int C_PAD, int C_STR>
void Conv(din_t* din, weight_t* win, dout_t* dout) {
  int s_oh_pad = 0;
  int s_ow_pad = 0;

  if (C_PAD == 1) {
    s_oh_pad = C_FH - 1;
    s_ow_pad = C_FW - 1;
  }

  for (int s_och = 0; s_och < C_OCH; s_och++) {
    for (int s_oh = 0; s_oh < C_OH; s_oh++) {
      for (int s_ow = 0; s_ow < C_OW; s_ow++) {
        int s_o_index = s_oh * C_OW * C_OCH + s_ow * C_OCH + s_och;
        dout[s_o_index] = 0;
        acc_t s_acc = 0;
        for (int s_ich = 0; s_ich < C_ICH; s_ich++) {
          for (int s_fh = 0; s_fh < C_FH; s_fh++) {
            for (int s_fw = 0; s_fw < C_FW; s_fw++) {
              int s_ih = s_oh + s_fh - s_oh_pad / 2;
              int s_iw = s_ow + s_fw - s_ow_pad / 2;

              int s_i_index = s_ih * C_IW * C_ICH + s_iw * C_ICH + s_ich;
              int s_w_index = s_ich * C_FH * C_FW * C_OCH +
                              s_fh * C_FW * C_OCH + s_fw * C_OCH + s_och;

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

#endif  // __CONV__
