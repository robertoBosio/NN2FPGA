#ifndef __CONV__
#define __CONV__

#include "ap_int.h"

template <class t_input, class t_weight, class t_output, class t_acc, int c_ich,
          int c_och, int c_iw, int c_ih, int c_ow, int c_oh, int c_fw, int c_fh,
          int c_pad, int c_str>
void Conv(t_input *i_data, t_weight *i_weight, t_output *o_data) {
  int s_oh_pad = 0;
  int s_ow_pad = 0;

  if (c_pad == 1) {
    s_oh_pad = c_fh - 1;
    s_ow_pad = c_fw - 1;
  }

  for (int s_och = 0; s_och < c_och; s_och++) {
    for (int s_oh = 0; s_oh < c_oh; s_oh++) {
      for (int s_ow = 0; s_ow < c_ow; s_ow++) {
        int s_o_index = s_oh * c_ow * c_och + s_ow * c_och + s_och;
        o_data[s_o_index] = 0;
        t_acc s_acc = 0;
        for (int s_ich = 0; s_ich < c_ich; s_ich++) {
          for (int s_fh = 0; s_fh < c_fh; s_fh++) {
            for (int s_fw = 0; s_fw < c_fw; s_fw++) {
              int s_ih = s_oh + s_fh - s_oh_pad / 2;
              int s_iw = s_ow + s_fw - s_ow_pad / 2;

              int s_i_index = s_ih * c_iw * c_ich + s_iw * c_ich + s_ich;
              int s_w_index = s_ich * c_fh * c_fw * c_och +
                              s_fh * c_fw * c_och + s_fw * c_och + s_och;

              if ((s_ih >= 0) & (s_iw >= 0) & (s_ih < c_ih) & (s_iw < c_iw)) {
                /* std::cout << (unsigned int)(i_data[s_i_index] & 0xff) << " "
                 * << (unsigned int)(i_weight[s_w_index] & 0xff) << "\n"; */
                s_acc += i_data[s_i_index] * i_weight[s_w_index];
              }
            }
          }
        }
        /* std::cout << "\n"; */
        o_data[s_o_index] = (t_output)(s_acc);
      }
    }
  }
}

#endif
