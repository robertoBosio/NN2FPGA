#ifndef NN2FPGA_BLOCK_INTERFACE_H_
#define NN2FPGA_BLOCK_INTERFACE_H_

#include "ap_int.h"
#include "hls_stream.h"

namespace nn2fpga {

template <typename din_t, int ICH, int IW, int IH, int c_fw, int c_fh,
          int c_pad>
void pad_input(hls::stream<din_t> &din, hls::stream<din_t> &o_data) {
  /* #pragma HLS inline */

  /* This handles padding aware inputs */

  constexpr int c_pad_index_h = c_pad * (c_fh - 1) / 2;
  constexpr int c_pad_index_w = c_pad * (c_fw - 1) / 2;

  constexpr int IH_PAD = IH + c_pad_index_h * 2;
  constexpr int IW_PAD = IW + c_pad_index_w * 2;

  constexpr din_t s_zero_false = {0, false};
  bool s_last = false;
  din_t s_zero = s_zero_false;

  /* OPTIMIZATION */
  constexpr int c_i_index = IH_PAD * IW_PAD * ICH;

  for (uint32_t s_ih = 0; s_ih < IH_PAD; s_ih++) {
    for (uint32_t s_iw = 0; s_iw < IW_PAD; s_iw++) {
      for (uint32_t s_ich = 0; s_ich < ICH; s_ich++) {
#pragma HLS pipeline style = stp

        bool s_data_read = true;

        s_data_read &= (s_ih >= c_pad_index_h);
        s_data_read &= (s_ih < (IH_PAD - c_pad_index_h));
        s_data_read &= (s_iw >= c_pad_index_w);
        s_data_read &= (s_iw < (IW_PAD - c_pad_index_w));

        din_t s_input = s_zero;
        if (s_data_read) s_input = din.read();

        s_zero.last = s_input.last;

        o_data.write(s_input);
      }
    }
  }
}

template <typename din_t, int ICH, int IW, int IH, int c_fw, int c_fh,
          int c_pad>
void forward_stream(hls::stream<din_t> &din) {
  constexpr int c_pad_index_h = c_pad * (c_fh - 1);
  constexpr int c_pad_index_w = c_pad * (c_fw - 1);
  constexpr int c_ih_end = IH + c_pad_index_h;
  constexpr int c_iw_end = IW + c_pad_index_w;

  for (uint8_t s_ih = 0; s_ih < c_ih_end; s_ih++) {
    for (uint8_t s_iw = 0; s_iw < c_iw_end; s_iw++) {
      for (uint8_t s_ich = 0; s_ich < ICH; s_ich++) {
        din_t s_tmp = din.read();
      }
    }
  }
}

template <typename din_t, int ICH, int IW, int IH, int c_fw, int c_fh,
          int c_pad>
void forward_stream(hls::stream<din_t> &din, hls::stream<din_t> &o_forward) {
  constexpr int c_pad_index_h = c_pad * (c_fh - 1) / 2;
  constexpr int c_pad_index_w = c_pad * (c_fw - 1) / 2;
  constexpr int c_ih_start = -1 * c_pad_index_h;
  constexpr int c_iw_start = -1 * c_pad_index_w;
  constexpr int c_ih_end = IH + c_pad_index_h;
  constexpr int c_iw_end = IW + c_pad_index_w;

  for (int8_t s_ih = c_ih_start; s_ih < c_ih_end; s_ih++) {
    for (int8_t s_iw = c_iw_start; s_iw < c_iw_end; s_iw++) {
      for (uint8_t s_ich = 0; s_ich < ICH; s_ich++) {
        din_t s_tmp = din.read();
        if ((s_ih > -1) & (s_iw > -1) & (s_ih < IH) & (s_iw < IW))
          o_forward.write(s_tmp);
      }
    }
  }
}

template <int c_split>
void split_stream(hls::stream<ap_uint<1>> &din,
                  hls::stream<ap_uint<1>> o_data[c_split]) {
  ap_uint<1> s_data = din.read();
  for (uint8_t s_split = 0; s_split < c_split; s_split++) {
#pragma HLS unroll
    o_data[s_split].write(s_data);
  }
}

template <typename dout_t, int OCH, int OW, int OH, int c_split>
void split_stream(hls::stream<dout_t> &din,
                  hls::stream<dout_t> o_data[c_split]) {
  for (uint8_t s_oh = 0; s_oh < OH; s_oh++) {
    for (uint8_t s_ow = 0; s_ow < OW; s_ow++) {
      for (uint8_t s_och = 0; s_och < OCH; s_och++) {
        dout_t s_out = din.read();
        for (uint8_t s_split = 0; s_split < c_split; s_split++) {
#pragma HLS unroll
          o_data[s_split].write(s_out);
        }
      }
    }
  }
}

}  // namespace nn2fpga

#endif  // NN2FPGA_BLOCK_INTERFACE_H_
