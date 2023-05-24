#ifndef NN2FPGA_LINE_BUFFER_H_
#define NN2FPGA_LINE_BUFFER_H_

#include <ap_int.h>
#include <hls_stream.h>

#include "nn2fpga/debug.h"

namespace nn2fpga {

template <typename T, int DEPTH>
class LineStream {
 public:
#ifndef __SYNTHESIS__
  int n_elems;
#endif  // __SYNTHESIS__
  hls::stream<T, DEPTH> s_stream;

  LineStream() {
#ifndef __SYNTHESIS__
    n_elems = 0;
#endif  // __SYNTHESIS__
#pragma HLS STREAM variable = s_stream depth = DEPTH type = fifo
  }

  void write(T din) {
    s_stream << din;
#ifndef __SYNTHESIS__
    if (!(n_elems < DEPTH)) std::cout << "WRITING FULL BUFFER" << std::endl;
    n_elems++;
#endif  // __SYNTHESIS__
  }

  T read() {
#ifndef __SYNTHESIS__
    if (n_elems > 0) n_elems--;
#endif  // __SYNTHESIS__
    return s_stream.read();
  }

  bool full() {
#ifndef __SYNTHESIS__
    return !(n_elems < DEPTH);
#else
    return s_stream.full();
#endif  // __SYNTHESIS__
  }

  bool empty() { return s_stream.empty(); }
};

template <typename T, int C_FH, int C_FW, int C_ICH, int C_IW>
class LineBuffer {
 public:
  static constexpr int C_INDEX = C_FH * C_FW;
  static constexpr ap_uint<8> C_ADDR[C_FH * C_FW] = {
      0xff, 0x10, 0x21, 0x42, 0x54, 0x65, 0x86, 0x98, 0xf9};
  LineStream<T, C_ICH> s_stream_c[C_FH][C_FW - 1];
  LineStream<T, C_ICH * C_IW> s_stream_r[C_FH - 1];

  LineBuffer() {}

  bool full(ap_uint<2> i_fh, ap_uint<2> i_fw) {
    if (i_fh == 3) return 0;

    if (i_fw(1, 1) == 1)
      return s_stream_r[i_fh].full();
    else
      return s_stream_c[i_fh][i_fw].full();
  }

  /* Fill the buffer for init */
  void fill(hls::stream<T>& din) {
    for (uint8_t s_index = C_INDEX - 1; s_index > 0; s_index--) {
      ap_uint<4> s_addr = C_ADDR[s_index](3, 0);
      do {
        SetLineBuffer(s_addr(3, 2), s_addr(1, 0), din.read());
      } while (!full(s_addr(3, 2), s_addr(1, 0)));
    }
  }

  /* Fill and retrieve */
  T pop_first() { return s_stream_c[C_FH - 1][C_FW - 2].read(); }

  void push_first(T din) { s_stream_c[0][0].write(din); }

  /* Fill and retrieve */
  T get(ap_uint<2> i_fh, ap_uint<2> i_fw) {
#pragma HLS inline
    if (i_fh == 3) return 0;

    if (i_fw(1, 1) == 1)
      return s_stream_r[i_fh].read();
    else
      return s_stream_c[i_fh][i_fw].read();
  }

  bool empty(ap_uint<2> i_fh, ap_uint<2> i_fw) {
    if (i_fh == 3) return 0;

    if (i_fw(1, 1) == 1)
      return s_stream_r[i_fh].empty();
    else
      return s_stream_c[i_fh][i_fw].empty();
  }

  void set(ap_uint<2> i_fh, ap_uint<2> i_fw, T din) {
#pragma HLS inline
    if (i_fh == 3) return;

    if (i_fw(1, 1) == 1)
      s_stream_r[i_fh] << din;
    else
      s_stream_c[i_fh][i_fw] << din;
  }

  T shift(uint8_t idx) {
#pragma HLS inline
    ap_uint<8> s_addr = C_ADDR[idx - 1];
    T s_stream = get(s_addr(3, 2), s_addr(1, 0));
    set(s_addr(7, 6), s_addr(5, 4), s_stream);

    return s_stream;
  }

  /* Empty the buffer */
  void empty(hls::stream<T>& dout) {
    for (uint8_t s_index = C_INDEX - 1; s_index > 0; s_index--) {
      ap_uint<4> s_addr = C_ADDR[s_index](3, 0);
      do {
        dout << get(s_addr(3, 2), s_addr(1, 0));
      } while (!empty(s_addr(3, 2), s_addr(1, 0)));
    }
  }

  void empty() {
    for (uint8_t s_index = C_INDEX - 1; s_index > 0; s_index--) {
      ap_uint<4> s_addr = C_ADDR[s_index](3, 0);
      do {
        get(s_addr(3, 2), s_addr(1, 0));
      } while (!empty(s_addr(3, 2), s_addr(1, 0)));
    }
  }

#ifndef __SYNTHESIS__
  void print_num_data() {
    std::cout << "-----------------------------" << std::endl;
    for (ap_int<4> s_fh = C_FH - 1; s_fh > -1; s_fh--) {
      for (ap_int<4> s_fw = C_FW - 2; s_fw > -1; s_fw--) {
        std::cout << s_stream_c[s_fh][s_fw].n_elems << std::endl;
      }

      if (s_fh > 0) {
        std::cout << s_stream_r[s_fh - 1].n_elems << std::endl;
      }
    }
  }
#endif  // __SYNTHESIS__
};

}  // namespace nn2fpga

#endif  // NN2FPGA_LINE_BUFFER_H_
