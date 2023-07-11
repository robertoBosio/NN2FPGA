#ifndef NN2FPGA_LINE_BUFFER_UTILS_H_
#define NN2FPGA_LINE_BUFFER_UTILS_H_

#include "ap_int.h"
#include "hls_stream.h"
#include "nn2fpga/line_buffer.h"

namespace nn2fpga {

template <typename din_t, int ICH, int OCH, int IH, int IW, int OH, int OW,
          int c_fh, int c_fw, int c_str, int c_pad, int c_pos_h, int c_pos_w>
void shift_op(hls::stream<din_t> &din, hls::stream<din_t> &o_compute) {
  /* #pragma HLS inline */

  constexpr int c_pad_index_h = c_pad * (c_fh - 1) / 2;
  constexpr int c_pad_index_w = c_pad * (c_fw - 1) / 2;
  constexpr int IH_PAD = IH + c_pad_index_h * 2;
  constexpr int IW_PAD = IW + c_pad_index_w * 2;
  constexpr int c_paddingh_shift = c_pos_h;
  constexpr int c_paddingw_shift = c_pos_w;
  constexpr int c_strideh_shift = (c_str - 1);
  constexpr int c_stridew_shift = (c_str - 1);
  constexpr int c_end_paddingh_shift = (c_fh - 1 - c_pos_h);
  constexpr int c_end_paddingw_shift = (c_fw - 1 - c_pos_w);

  /* Constants for new version */
  constexpr int c_i_index = IH_PAD * IW_PAD * ICH;

  constexpr int c_starth = c_pad_index_h;
  constexpr int c_startw = c_pad_index_w;
  constexpr int c_endh = IH_PAD - c_pad_index_h;
  constexpr int c_endw = IW_PAD - c_pad_index_w;

  for (auto s_index_h = c_starth; s_index_h < c_endh; s_index_h++) {
    for (auto s_index_w = c_startw; s_index_w < c_endw; s_index_w++) {
      for (auto s_index_ich = 0; s_index_ich < ICH; s_index_ich++) {
#pragma HLS pipeline style = stp
        bool s_compute_write = true;
        auto s_index_h_str = s_index_h % c_str;
        auto s_index_w_str = s_index_w % c_str;

        s_compute_write &= (s_index_h >= c_paddingh_shift);
        s_compute_write &= (s_index_h < (IH_PAD - c_end_paddingh_shift));
        s_compute_write &= (s_index_w >= c_paddingw_shift);
        s_compute_write &= (s_index_w < (IW_PAD - c_end_paddingw_shift));
        s_compute_write &= (s_index_h_str == (c_paddingh_shift % c_str));
        s_compute_write &= (s_index_w_str == (c_paddingw_shift % c_str));

        din_t s_input = din.read();
        if (s_compute_write) o_compute.write(s_input);
      }
    }
  }
}

template <typename din_t, int ICH, int IH, int IW, int c_fw, int c_fh,
          int c_str, int c_pad>
void pad_input(hls::stream<din_t> din[c_fw * c_fh],
               hls::stream<din_t> o_data[c_fw * c_fh]) {
  /* #pragma HLS inline */

  /* This handles padding aware inputs */

  constexpr int c_pad_index_h = c_pad * (c_fh - 1) / 2;
  constexpr int c_pad_index_w = c_pad * (c_fw - 1) / 2;
  constexpr int IH_REM = IH - (IH % c_str)*(1-c_pad);
  constexpr int IW_REM = IW - (IW % c_str)*(1-c_pad);
  constexpr int IH_PAD = IH + c_pad_index_h * 2 - IH_REM*(1-c_pad);
  constexpr int IW_PAD = IW + c_pad_index_w * 2 - IW_REM*(1-c_pad);
  constexpr int FSZ = c_fh * c_fw;

  bool s_last;
  // std::cout << "pad_input" << std::endl;
  for (auto s_index_h = 0; s_index_h < IH_REM; s_index_h += c_str) {
    for (auto s_index_w = 0; s_index_w < IW_REM; s_index_w += c_str) {
      for (auto s_index_ich = 0; s_index_ich < ICH; s_index_ich++) {
#pragma HLS pipeline style = stp
        for (auto s_fh = 0; s_fh < c_fh; s_fh++) {
          for (auto s_fw = 0; s_fw < c_fw; s_fw++) {
            din_t s_write;

            auto s_index = s_fh * c_fw + s_fw;

            bool s_data_read = true;

            s_data_read &= (s_index_h >= (c_pad_index_h - s_fh));
            s_data_read &= (s_index_h < (IH + c_pad_index_h - s_fh));
            s_data_read &= (s_index_w >= (c_pad_index_w - s_fw));
            s_data_read &= (s_index_w < (IW + c_pad_index_w - s_fw));

            if (s_data_read) {
              s_write = din[FSZ - s_index - 1].read();
              if (s_index == FSZ - 1) s_last = s_write.last;
            } else {
              s_write.data = 0;
              s_write.last = s_last;
            }
            o_data[FSZ - s_index - 1].write(s_write);
          }
        }
      }
    }
  }
}

template <typename din_t, int ICH, int OCH, int IH, int IW, int OH, int OW,
          int c_fh, int c_fw, int c_str, int c_pad, int c_pos_h, int c_pos_w>
void shift_op(hls::stream<din_t> &din, hls::stream<din_t> &o_compute,
              hls::stream<din_t> &o_data) {
  /* #pragma HLS inline */

  // std::cout << "shift_op " << IH << " " << IW << std::endl;
  constexpr int c_pad_index_h = c_pad * (c_fh - 1) / 2;
  constexpr int c_pad_index_w = c_pad * (c_fw - 1) / 2;
  constexpr int IH_PAD = IH + c_pad_index_h * 2;
  constexpr int IW_PAD = IW + c_pad_index_w * 2;
  constexpr int c_paddingh_shift = c_pos_h;
  constexpr int c_paddingw_shift = c_pos_w;
  constexpr int c_strideh_shift = (c_str - 1);
  constexpr int c_stridew_shift = (c_str - 1);
  constexpr int c_end_paddingh_shift = (c_fh - 1 - c_pos_h);
  constexpr int c_end_paddingw_shift = (c_fw - 1 - c_pos_w);

  /* Constants for new version */
  constexpr int c_i_index = IH_PAD * IW_PAD * ICH;

  constexpr int c_starth = c_pad_index_h;
  constexpr int c_startw = c_pad_index_w;
  constexpr int c_endh = IH_PAD - c_pad_index_h;
  constexpr int c_endw = IW_PAD - c_pad_index_w;

  hls::stream<din_t> s_compute;
#pragma HLS stream variable = s_compute depth = 2 type = fifo

  for (auto s_index_h = c_starth; s_index_h < c_endh; s_index_h++) {
    for (auto s_index_w = c_startw; s_index_w < c_endw; s_index_w++) {
      for (auto s_index_ich = 0; s_index_ich < ICH; s_index_ich++) {
#pragma HLS pipeline style = stp
        bool s_compute_write = true;
        auto s_index_h_str = s_index_h % c_str;
        auto s_index_w_str = s_index_w % c_str;

        s_compute_write &= (s_index_h >= c_paddingh_shift);
        s_compute_write &= (s_index_h < (IH_PAD - c_end_paddingh_shift));
        s_compute_write &= (s_index_w >= c_paddingw_shift);
        s_compute_write &= (s_index_w < (IW_PAD - c_end_paddingw_shift));
        s_compute_write &= (s_index_h_str == (c_paddingh_shift % c_str));
        s_compute_write &= (s_index_w_str == (c_paddingw_shift % c_str));

        din_t s_input = din.read();
        if (s_compute_write) o_compute.write(s_input);
        o_data.write(s_input);
      }
    }
  }
}

}  // namespace nn2fpga

#endif  // NN2FPGA_LINE_BUFFER_UTILS_H_
