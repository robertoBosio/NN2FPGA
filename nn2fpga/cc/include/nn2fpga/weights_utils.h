#ifndef NN2FPGA_WEIGHTS_UTILS_H_
#define NN2FPGA_WEIGHTS_UTILS_H_

#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"

namespace nn2fpga {

// For input weights
template <typename din_t, typename dout_t, int ICH, int OCH, int IW, int IH,
          int OW, int OH, int c_str>
void produce_stream(din_t *din, hls::stream<dout_t> &s_din) {
  constexpr unsigned SZ = OCH * ICH * IH * IW;

  for (int s_oh = 0; s_oh < OH; s_oh++) {
    for (int s_ow = 0; s_ow < OW; s_ow++) {
    PRODSTR:
      for (int s_index = 0; s_index < SZ; s_index++) {
        s_din.write((dout_t)(din[s_index]));
      }
    }
  }
}
//
// For input weights
template <typename din_t, typename dout_t, int ICH, int OCH, int IW, int IH,
          int OW, int OH>
void produce_stream(const din_t din[OCH * ICH * IW * IH],
                    hls::stream<dout_t> o_data[IH * IW]) {
  constexpr unsigned OSZ = OH * OW;
  constexpr unsigned c_stream_sel = IH * IW;
  constexpr unsigned c_ch = ICH * OCH;
#pragma HLS array_partition type = cyclic factor = c_stream_sel variable = din

  for (uint16_t s_index = 0; s_index < OSZ; s_index++) {
    uint16_t s_addr = 0;
    for (uint16_t s_ch = 0; s_ch < c_ch; s_ch++) {
#pragma HLS pipeline
      for (uint8_t s_stream_sel = 0; s_stream_sel < c_stream_sel;
           s_stream_sel++) {
        o_data[s_stream_sel].write((dout_t)(din[s_addr]));
        s_addr++;
      }
    }
  }
}

template <typename din_t, typename dout_t, int ICH, int OCH, int IW, int IH,
          int OW, int OH, int c_ops>
void produce_stream(const din_t din[OCH * ICH * IW * IH],
                    hls::stream<dout_t> o_data[IH * IW]) {
  constexpr unsigned OSZ = OH * OW;
  constexpr unsigned c_stream_sel = IH * IW;
  constexpr unsigned c_ch = ICH * OCH / c_ops;
#pragma HLS array_partition type = cyclic factor = c_stream_sel variable = din

  for (uint16_t s_index = 0; s_index < OSZ; s_index++) {
    uint16_t s_addr = 0;
    for (uint16_t s_ch = 0; s_ch < c_ch; s_ch++) {
      for (uint8_t s_stream_sel = 0; s_stream_sel < c_stream_sel;
           s_stream_sel++) {
#pragma HLS pipeline
        for (uint16_t s_ops = 0; s_ops < c_ops; s_ops++) {
          o_data[s_stream_sel].write((dout_t)(din[s_addr]));
          s_addr++;
        }
      }
    }
  }
}

template <typename din_t, typename dout_t, int ICH, int OCH, int OW, int OH>
void produce_stream(const din_t din[OCH * ICH], hls::stream<dout_t> &o_data) {
  constexpr unsigned OSZ = OH * OW;
  constexpr unsigned c_ch = ICH * OCH;

  for (uint16_t s_index = 0; s_index < OSZ; s_index++) {
    for (uint16_t s_ch = 0; s_ch < c_ch; s_ch++) {
#pragma HLS pipeline
      o_data.write((dout_t)(din[s_ch]));
    }
  }
}

template <typename din_t, typename dout_t, int ICH, int OCH, int OW, int OH,
          int c_ops>
void produce_stream(const din_t din[OCH * ICH], hls::stream<dout_t> &o_data) {
  constexpr unsigned OSZ = OH * OW;
  constexpr unsigned c_ch = ICH * OCH / c_ops;
  constexpr uint8_t c_log_ops = (uint8_t)(log2(c_ops));

  for (uint16_t s_index = 0; s_index < OSZ; s_index++) {
    for (uint16_t s_ch = 0; s_ch < c_ch; s_ch++) {
#pragma HLS pipeline
      dout_t s_data = 0;
      for (uint8_t s_ops = 0; s_ops < c_ops; s_ops++) {
        s_data(8 * (s_ops + 1) - 1, 8 * s_ops) =
            din[(s_ch << c_log_ops) + s_ops];
      }
      o_data.write((dout_t)(s_data));
    }
  }
}

template <typename din_t, typename dout_t, int ICH, int OCH, int OW, int OH,
          int c_fw, int c_fh, int c_ops, int c_reuse>
void produce_stream(const din_t din[c_fh * c_fw][OCH * ICH / c_ops][c_ops],
                    hls::stream<dout_t> o_data[c_fh * c_fw]) {
  constexpr unsigned FSZ = c_fh * c_fw;
  constexpr unsigned c_ch = ICH * OCH / c_ops;
  constexpr unsigned c_o_index = OH * OW * c_ch / c_reuse;

  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
#pragma HLS pipeline
    auto s_ch = s_o_index % c_ch;
    for (auto s_index = 0; s_index < FSZ; s_index++) {
      dout_t s_output;
      for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
        s_output[s_ops] = din[s_index][s_ch][s_ops];
      }
      o_data[s_index].write(s_output);
    }
  }
}

template <typename din_t, typename dout_t, int ICH, int OCH, int OW, int OH,
          int c_fw, int c_fh, int c_ops>
void produce_stream(hls::stream<din_t> din[c_fh * c_fw],
                    hls::stream<dout_t> o_data[c_fh * c_fw]) {
  constexpr unsigned FSZ = c_fh * c_fw;
  constexpr unsigned c_ch = ICH * OCH / c_ops;
  constexpr unsigned c_iter = 64 / c_ops;
  constexpr unsigned c_o_index = OH * OW * c_ch;
  din_t s_data[FSZ];
  for (uint32_t s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
#pragma HLS pipeline style = flp
    uint8_t s_iter = s_o_index % c_iter;
    if (s_iter == 0) {
      for (uint8_t s_index = 0; s_index < FSZ; s_index++)
        s_data[s_index] = din[s_index].read();
    }

    for (uint8_t s_index = 0; s_index < FSZ; s_index++) {
      dout_t s_output;
      for (uint8_t s_ops = 0; s_ops < c_ops; s_ops++) {
        uint8_t s_part = s_iter * c_ops + c_ops;
        s_output[s_ops] = s_data[s_index](8 * (s_part + 1) - 1, 8 * s_part);
      }
      o_data[s_index].write(s_output);
    }
  }
}

}  // namespace nn2fpga

#endif  // NN2FPGA_WEIGHTS_UTILS_H_
