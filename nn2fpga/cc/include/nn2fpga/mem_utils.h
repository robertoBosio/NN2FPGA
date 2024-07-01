#ifndef NN2FPGA_MEM_UTILS_H_
#define NN2FPGA_MEM_UTILS_H_

#include <ap_int.h>
#include <hls_burst_maxi.h>
#include <hls_stream.h>
#include <stdio.h>
#include <string.h>

namespace nn2fpga {

template <unsigned NCONV, unsigned RDW>
uint8_t round_robin(hls::stream<ap_uint<RDW>> dout[NCONV]) {
#pragma HLS inline

  uint8_t s_sel = 0;

  for (int8_t n = NCONV - 1; n >= 0; n--) {
    if (dout[n].size() == 0) s_sel = n;
  }

  return s_sel;
}

template <unsigned BITW, int OFFSET, int ADDR>
void fill_stream(ap_uint<BITW> *din, uint32_t &addrIn,
                 hls::stream<ap_uint<BITW>> &doutStream) {
#pragma HLS inline
  while (!doutStream.full()) {
    doutStream << din[addrIn];
    // addrIn = addrIn++;
    addrIn++;
    // addrIn = (addrIn == ADDR) ? addrIn : OFFSET;
    if (addrIn == ADDR) addrIn = OFFSET;
  }
}

template <typename din_t, typename dout_t, int C_ICH, int C_OCH, int C_OW,
          int C_OH, int C_FW, int C_FH, int C_OPS, int BITW>
void produce_stream(hls::stream<din_t> &dinStream,
                    hls::stream<dout_t> doutStream[C_FH * C_FW]) {
  /* #pragma HLS inline */
  constexpr unsigned C_INDEX = C_FH * C_FW;
  constexpr unsigned C_CH = C_ICH * C_OCH;
  constexpr unsigned C_BYTES = BITW / (8);
  constexpr unsigned C_PACK_INV = C_BYTES / (C_OPS);
  constexpr unsigned C_INV = (C_PACK_INV == 0) ? 1 : C_PACK_INV;

  constexpr unsigned C_PACK = C_OPS / (C_BYTES);
  constexpr unsigned C_READ = (C_PACK == 0) ? 1 : C_PACK;

  constexpr unsigned C_O_INDEX = C_OH * C_OW * C_CH / (C_OPS);
  constexpr unsigned C_R_INDEX = C_INDEX * C_OH * C_OW * C_CH / (C_OPS);
  constexpr unsigned C_BUFFER = C_BYTES / (C_INDEX * C_OPS) + 1;

  /* const ap_uint<C_OPS*8> c_mask = C_OPS*256-1; */

  /* Maximum input bandwidth is 64bytes */
  din_t tmp;
  for (unsigned rIdx = 0; rIdx < C_R_INDEX; rIdx++) {
#pragma HLS pipeline
    uint8_t inv = rIdx % C_INV;
    uint8_t idx = rIdx % C_INDEX;
    if (inv == 0) tmp = dinStream.read();
    dout_t dout;
    for (unsigned op = 0; op < C_OPS; op++) {
      dout[op] = tmp & 0xff;
      tmp >>= 8;
    }
    doutStream[idx] << dout;
  }
}

template <typename dout_t, int C_ICH, int C_OCH, int C_OW, int C_OH, int C_FW,
          int C_FH, int C_OPS, int BITW, int C_START>
void mem_algo(hls::stream<dout_t> dout[C_FH * C_FW], ap_int<BITW> *din) {
#pragma HLS inline
#pragma HLS dataflow
  constexpr int C_BYTES = BITW / 8;
  constexpr int c_words = 4096 / (C_BYTES);
  constexpr int c_index = C_FH * C_FW;
  constexpr int c_f_index = C_START + c_index * C_OCH * C_ICH;
  constexpr int c_w_index = C_OCH * C_ICH / C_OPS;
  constexpr int C_O_INDEX = C_OW * C_OH;

  hls::stream<dout_t> s_data_stream("data_stream");
#pragma HLS stream variable = s_data_stream depth = 1
  for (auto s_o_index = 0; s_o_index < C_O_INDEX; s_o_index++) {
    for (auto s_f_index = C_START; s_f_index < c_f_index; s_f_index += C_OPS) {
#pragma HLS pipeline style = frp
      dout_t s_data;
      for (auto s_ops = 0; s_ops < C_OPS; s_ops++)
        s_data[s_ops] = din[s_f_index + s_ops];
      s_data_stream.write(s_data);
    }
  }

  for (auto s_o_index = 0; s_o_index < C_O_INDEX; s_o_index++) {
    for (auto s_w_index = 0; s_w_index < c_w_index; s_w_index++) {
      for (auto idx = 0; idx < c_index; idx++) {
#pragma HLS pipeline style = frp
        dout[idx].write(s_data_stream.read());
      }
    }
  }
}

template <typename dout_t, int C_ICH, int C_OCH, int C_OW, int C_OH, int C_FW,
          int C_FH, int C_OPS, int BITW, int C_START>
void mem_algo(hls::stream<dout_t> &dout, ap_int<BITW> *din) {
  constexpr int C_BYTES = BITW / 8;
  constexpr int c_words = 4096 / (C_BYTES);
  constexpr int c_f_index = C_START + C_FH * C_FW * C_OCH * C_ICH;
  constexpr int C_O_INDEX = C_OW * C_OH;

  for (auto s_o_index = 0; s_o_index < C_O_INDEX; s_o_index++) {
    for (auto s_f_index = C_START; s_f_index < c_f_index; s_f_index += C_OPS) {
#pragma HLS pipeline
      dout_t s_data;
      for (auto s_ops = 0; s_ops < C_OPS; s_ops++)
        s_data[s_ops] = din[s_f_index + s_ops];
      dout.write(s_data);
    }
  }
}

template <typename din_t, typename dout_t, int C_ICH, int C_OCH, int C_OW,
          int C_OH, int C_FW, int C_FH, int C_OPS, int BITW>
void produce_stream(hls::stream<dout_t> &din,
                    hls::stream<dout_t> dout[C_FH * C_FW]) {
  /* #pragma HLS inline */
  constexpr int c_index = C_FH * C_FW;
  constexpr int C_CH = C_ICH * C_OCH;
  constexpr int C_O_INDEX = C_OH * C_OW * C_CH / (C_OPS);

  /* const ap_uint<C_OPS*8> c_mask = C_OPS*256-1; */

  /* Maximum input bandwidth is 64bytes */
  din_t s_tmp;
  for (auto s_o_index = 0; s_o_index < C_O_INDEX; s_o_index++) {
    for (auto idx = 0; idx < c_index; idx++) {
#pragma HLS pipeline
      dout[idx].write(din.read());
    }
  }
}

template <typename din_t, typename dout_t, int C_ICH, int C_OCH, int C_OW,
          int C_OH, int C_FW, int C_FH, int C_OPS, int BITW, int c_bw,
          int c_reuse, int C_START>
void mem_algo(hls::stream<dout_t> dout[c_bw], const din_t *din) {
  constexpr int C_BYTES = BITW / 8;
  constexpr int c_words = 4096 / (C_BYTES);
  constexpr int c_f_index = C_START + C_FH * C_FW * C_OCH * C_ICH;
  constexpr int C_O_INDEX = C_OW * C_OH / c_reuse;

  for (auto s_o_index = 0; s_o_index < C_O_INDEX; s_o_index++) {
    for (auto s_f_index = C_START; s_f_index < c_f_index;
         s_f_index += C_OPS * c_bw) {
#pragma HLS pipeline
      for (auto s_bw = 0; s_bw < c_bw * C_OPS; s_bw += C_OPS) {
        dout_t s_data;
        for (auto s_ops = 0; s_ops < C_OPS; s_ops++)
          s_data[s_ops] = din[s_f_index + s_bw + s_ops];
        dout[s_bw / C_OPS].write(s_data);
      }
    }
  }
}

template <typename din_t, typename dout_t, int C_ICH, int C_OCH, int C_OW,
          int C_OH, int C_FW, int C_FH, int C_OPS, int c_bw, int c_reuse,
          int BITW>
void produce_stream(hls::stream<dout_t> dinStream[c_bw],
                    hls::stream<dout_t> doutStream[C_FH * C_FW]) {
  /* #pragma HLS inline */
  constexpr int c_index = C_FH * C_FW;
  constexpr int C_CH = C_ICH * C_OCH;
  constexpr int C_O_INDEX = c_index * C_OH * C_OW * C_CH / (C_OPS * c_reuse);

  /* const ap_uint<C_OPS*8> c_mask = C_OPS*256-1; */

  /* Maximum input bandwidth is 64bytes */
  din_t s_tmp;
  for (auto s_o_index = 0; s_o_index < C_O_INDEX; s_o_index += c_bw) {
#pragma HLS pipeline
    for (auto s_bw = 0; s_bw < c_bw; s_bw++) {
      auto idx = (s_o_index + s_bw) % c_index;
      doutStream[idx].write(dinStream[s_bw].read());
    }
  }
}

template <int C_ICH, int C_OCH, int C_OW, int C_OH, int C_FW, int C_FH,
          int C_OPS, int BITW, int C_START>
void mem_algo(hls::stream<ap_uint<BITW>> &dout,
              hls::burst_maxi<ap_uint<BITW>> din) {
  constexpr int C_BYTES = BITW / 8;
  constexpr int c_words = 4096 / (C_BYTES);
  constexpr int c_f_index = C_FH * C_FW * C_OCH * C_ICH;
  constexpr int c_b_index = c_f_index / 4096;
  constexpr int c_b_rem = c_f_index % 4096;
  constexpr int C_START_w = C_START / C_BYTES;
  constexpr int c_b_rem_words = c_b_rem / C_BYTES;
  constexpr int C_O_INDEX = C_OW * C_OH;

  for (auto s_o_index = 0; s_o_index < C_O_INDEX; s_o_index++) {
    for (auto s_b_index = 0; s_b_index < c_b_index; s_b_index++) {
      uint32_t s_read = C_START_w + s_b_index * c_words;
      din.read_request(s_read, c_words);
      for (auto s_words = 0; s_words < c_words; s_words++) {
#pragma HLS pipeline
        ap_uint<BITW> s_data = din.read();
        dout.write(s_data);
      }
    }
    if (c_b_rem != 0) {
      uint32_t c_rem_start = C_START_w + c_b_index * c_words;
      din.read_request(c_rem_start, c_b_rem_words);
      for (auto s_words = 0; s_words < c_b_rem_words; s_words++) {
#pragma HLS pipeline
        ap_uint<BITW> s_data = din.read();
        dout.write(s_data);
      }
    }
  }
}

}  // namespace nn2fpga

#endif // NN2FPGA_MEM_UTILS_H_
