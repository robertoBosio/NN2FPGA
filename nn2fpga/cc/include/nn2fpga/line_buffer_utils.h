#ifndef NN2FPGA_LINE_BUFFER_UTILS_H_
#define NN2FPGA_LINE_BUFFER_UTILS_H_

#include "ap_int.h"
#include "hls_stream.h"

namespace nn2fpga {

template <typename din_t, typename dout_t, int ICH, int IH, int IW, int c_fw, int c_fh,
          int c_str, int c_pad, int c_ws, int c_ops, int c_ops_out>
void pad_input(hls::stream<din_t> din[(c_fw+(c_ws-1)*c_str) * c_fh],
               hls::stream<dout_t> o_data[1]) {
  /* #pragma HLS inline */

  /* This handles padding aware inputs */

  constexpr int c_pad_index_h = c_pad * (c_fh - 1) / 2;
  constexpr int c_pad_index_w = c_pad * (c_fw - 1) / 2;
  constexpr int IH_REM = IH - (IH % c_str)*(1-c_pad);
  constexpr int IW_REM = IW - (IW % c_str)*(1-c_pad);
  constexpr int IH_PAD = IH + c_pad_index_h * 2 - IH_REM*(1-c_pad);
  constexpr int IW_PAD = IW + c_pad_index_w * 2 - IW_REM*(1-c_pad);
  constexpr int FSZ = c_fh * (c_fw+(c_ws-1)*c_str);
  constexpr int FW = (c_fw+(c_ws-1)*c_str);
  constexpr int LAST_IDX = FSZ - 1 - (c_fw+(c_ws-1)*c_str-c_pad_index_w)%c_ws;

  bool s_last;
  
  din_t s_read[FSZ];

  for (auto s_index_h = 0; s_index_h < IH_REM; s_index_h += c_str) {
    for (auto s_index_w = 0; s_index_w < IW_REM; s_index_w += c_str*c_ws) {
      for (auto s_index_ich = 0; s_index_ich < ICH; s_index_ich+=c_ops_out) {
#pragma HLS pipeline style = stp II=1
        dout_t s_write;
        auto s_ops = s_index_ich % c_ops;
        for (auto s_fh = 0; s_fh < c_fh; s_fh++) {
          for (auto s_fw = 0; s_fw < FW; s_fw++) {

            auto s_index = s_fh * FW + s_fw;

            bool s_data_read = true;

            s_data_read &= (s_index_h >= (c_pad_index_h - s_fh));
            s_data_read &= (s_index_h < (IH + c_pad_index_h - s_fh));
            s_data_read &= (s_index_w >= (c_pad_index_w - s_fw));
            s_data_read &= (s_index_w < (IW + c_pad_index_w - s_fw));

            if (s_data_read) {
              if (s_ops == 0) {
                s_read[FSZ - s_index - 1] = din[FSZ - s_index - 1].read();
                if (s_index == LAST_IDX) s_last = s_read[FSZ - s_index - 1].last;
              }
              for (auto s_i = 0; s_i < c_ops_out; s_i++) {
                s_write.data[FSZ - s_index - 1][s_i] = s_read[FSZ - s_index - 1].data[0][s_ops + s_i];
              }
              s_write.last = s_read[FSZ - s_index - 1].last;
            } else {
              // for (auto s_i = 0; s_i < c_ops; s_i++) {
              // This is padding branch, if the data of the window should not be read
              // form the input stream then we pad it with zeros
              for (auto s_i = 0; s_i < c_ops_out; s_i++) {
                s_write.data[FSZ - s_index - 1][s_i] = 0;
              }
              s_write.last = s_last;
            }
          }
        }
        o_data[0].write(s_write);
      }
    }
  }
}

template <typename din_t, typename dout_t, int ICH, int IH, int IW,
          int c_ws, int c_ws_out, int c_ops>
void change_ws(hls::stream<din_t> din[c_ws], 
              hls::stream<dout_t> o_data[c_ws_out]) {

  constexpr int c_i_index = IH * IW;
  constexpr int c_num_ich = ICH/c_ops;

  din_t s_read[c_ws];
  for (auto s_index = 0; s_index < c_i_index; s_index++) {
    for (auto s_ich = 0; s_ich < c_num_ich; s_ich++) {
      #pragma HLS pipeline style = stp
      o_data[s_index % c_ws_out].write(din[s_index % c_ws].read());
    }
  }

}

template <typename din_t, typename dcomp_t, typename dout_t, int ICH, int OCH, int IH, int IW, int OH, int OW,
          int c_fh, int c_fw, int c_str, int c_pad, int c_pos_h, int c_pos_w, int c_ws, int c_ops, int c_ops_out>
void shift_op(hls::stream<din_t> &din, hls::stream<dcomp_t> &o_compute,
              hls::stream<dout_t> &o_data) {
  /* #pragma HLS inline */

  constexpr int c_pad_index_h = c_pad * (c_fh - 1) / 2;
  constexpr int c_pad_index_w = c_pad * (c_fw - 1) / 2;
  constexpr int IH_PAD = IH + c_pad_index_h * 2;
  constexpr int IW_PAD = IW + c_pad_index_w * 2;
  // constexpr int c_paddingh_shift = c_pos_h;
  // constexpr int c_paddingw_shift = c_pos_w;
  constexpr int c_strideh_shift = (c_str - 1);
  constexpr int c_stridew_shift = (c_str - 1);
  // constexpr int c_end_paddingh_shift = (c_fh - 1 - c_pos_h);
  // constexpr int c_end_paddingw_shift = (c_fw + c_ws - 2 - c_pos_w);
  // constexpr int c_end_paddingw_shift = (c_fw - 1 - c_pos_w);

  /* Constants for new version */
  constexpr int c_i_index = IH_PAD * IW_PAD * ICH;

  constexpr int c_endh = IH_PAD - c_pad_index_h;
  constexpr int c_endw = IW_PAD - c_pad_index_w + (c_fw - c_pos_w) % (c_ws);
  constexpr int FW = (c_fw+(c_ws-1)*c_str);

  // The window output is written as soon as the first data which falls
  // in the windows position is available
  // The window position is affected by the padding which adds data to 
  // the beginning of the tensor width
  constexpr int c_paddingh_shift = c_pos_h - c_pad_index_h;
  constexpr int c_paddingw_shift = c_pos_w - c_pad_index_w;

  // No window output is written if the data is not in the windows position
  // At the end of the tensor width
  // The window position is affected by the padding which adds data to
  // the end of the tensor width
  constexpr int c_end_paddingh_shift = c_fh - 1 - c_pos_h - c_pad_index_h;
  constexpr int c_end_paddingw_shift = c_fw - 1 - c_pos_w - c_pad_index_w;

  // Given the weight stationary degree and the position in the window,
  // the first pixel received varies
  constexpr int c_starth = 0;
  constexpr int c_startw = (c_paddingw_shift % c_ws > 0) ? (c_paddingw_shift % c_ws) : -1 * (c_paddingw_shift % c_ws);
  const int c_str_adj = (c_str == 1) ? 1 : (c_ws);

  // Stride selects which pixels are sent for computations
  constexpr int c_strh = (c_paddingh_shift > 0) ? (c_paddingh_shift % c_str) : -1 * (c_paddingh_shift % c_str);
  constexpr int c_strw = (c_paddingw_shift > 0) ? (c_paddingw_shift % (c_str*c_str_adj)) : (FW - 1 + c_paddingw_shift) % (c_str*c_str_adj);

  din_t s_input;
  dcomp_t s_output;
  for (auto s_index_h = c_starth; s_index_h < IH; s_index_h++) {
    for (auto s_index_w = c_startw; s_index_w < IW; s_index_w+=c_ws) {
      for (auto s_index_ich = 0; s_index_ich < (ICH); s_index_ich+=c_ops_out) {
      // for (auto s_index_ich = 0; s_index_ich < ICH; s_index_ich+=c_ops) {
#pragma HLS pipeline style = stp II=1
        auto s_index_read = s_index_ich % c_ops;
        auto s_data_read = s_index_read == 0;

        bool s_compute_write = true;
        auto s_index_h_str = s_index_h % c_str;
        auto s_index_w_str = s_index_w % (c_str*c_str_adj);

        s_compute_write &= (s_index_h >= c_paddingh_shift);
        s_compute_write &= (s_index_h < (IH - c_end_paddingh_shift));
        s_compute_write &= (s_index_w >= c_paddingw_shift);
        s_compute_write &= (s_index_w < (IW - c_end_paddingw_shift));
        s_compute_write &= (s_index_h_str == (c_strh));
        s_compute_write &= (s_index_w_str == (c_strw));

        if (s_data_read) s_input = din.read();
        for (auto s_index_ops = 0; s_index_ops < c_ops_out; s_index_ops++) {
          s_output.data[0][s_index_ops] = s_input.data[0][s_index_read + s_index_ops];
        }
        s_output.last = s_input.last;
        if (s_compute_write) o_compute.write(s_output);
        if constexpr(std::is_same<dout_t, std::nullptr_t>::value == false) o_data.write(s_output);
      }
    }
  }
}

}  // namespace nn2fpga

#endif  // NN2FPGA_LINE_BUFFER_UTILS_H_
