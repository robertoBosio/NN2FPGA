#ifndef NN2FPGA_LINE_BUFFER_H_
#define NN2FPGA_LINE_BUFFER_H_

#include <ap_int.h>
#include <hls_stream.h>

#include "nn2fpga/debug.h"

namespace nn2fpga {

template <typename din_t, typename dout_t, int ICH, int OCH, int IH, int IW, int OH, int OW,
          int c_fh, int c_fw, int c_str, int c_pad, int c_pos_h, int c_pos_w, int c_ws>
void line_buffer(hls::stream<din_t> &din, hls::stream<din_t> o_compute[c_fh*c_fw],
              hls::stream<dout_t> &o_data) {
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
  // constexpr int c_end_paddingw_shift = (c_fw + c_ws - 2 - c_pos_w);
  constexpr int c_end_paddingw_shift = (c_fw - 1 - c_pos_w);

  /* Constants for new version */
  constexpr int c_i_index = IH_PAD * IW_PAD * ICH;

  constexpr int c_starth = c_pad_index_h;
  constexpr int c_startw = c_pad_index_w + (c_fw - c_pos_w) % c_ws;
  constexpr int c_endh = IH_PAD - c_pad_index_h;
  constexpr int c_endw = IW_PAD - c_pad_index_w + (c_fw - c_pos_w) % c_ws;

  hls::stream<din_t> s_compute;
#pragma HLS stream variable = s_compute depth = 2 type = fifo

  for (auto s_index_h = c_starth; s_index_h < c_endh; s_index_h++) {
    for (auto s_index_w = c_startw; s_index_w < c_endw; s_index_w+=c_ws) {
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
        if constexpr(std::is_same<dout_t, std::nullptr_t>::value == false) o_data.write(s_input);
      }
    }
  }
}

}  // namespace nn2fpga

#endif  // NN2FPGA_LINE_BUFFER_H_
