#ifndef __LINEBUFFER__
#define __LINEBUFFER__

#include "Debug.hpp"
#include "ap_int.h"
#include "hls_stream.h"

template <class t_stream, int c_depth>
class LineStream {
 public:
#ifndef __SYNTHESIS__
  int n_elems;
#endif
  hls::stream<t_stream, c_depth> s_stream;

  LineStream() {
#ifndef __SYNTHESIS__
    n_elems = 0;
#endif
#pragma HLS STREAM variable = s_stream depth = c_depth type = fifo
  }

  void write(t_stream i_write) {
    s_stream.write(i_write);
#ifndef __SYNTHESIS__
    if (!(n_elems < c_depth)) std::cout << "WRITING FULL BUFFER" << std::endl;
    n_elems++;
#endif
  }

  t_stream read() {
#ifndef __SYNTHESIS__
    if (n_elems > 0) n_elems--;
#endif
    return s_stream.read();
  }

  bool full() {
#ifndef __SYNTHESIS__
    return !(n_elems < c_depth);
#else
    return s_stream.full();
#endif
  }

  bool empty() { return s_stream.empty(); }
};

template <class t_stream, int c_fh, int c_fw, int c_ich, int c_iw>
class LineBuffer {
 public:
  const int c_index = c_fh * c_fw;
  LineStream<t_stream, c_ich> s_stream_c[c_fh][c_fw - 1];
  LineStream<t_stream, c_ich * c_iw> s_stream_r[c_fh - 1];
  const ap_uint<8> c_addr[c_fh * c_fw] = {0xff, 0x10, 0x21, 0x42, 0x54,
                                          0x65, 0x86, 0x98, 0xf9};

  LineBuffer() {}

  bool FullLineBuffer(ap_uint<2> i_fh, ap_uint<2> i_fw) {
    if (i_fh == 3) return 0;

    if (i_fw(1, 1) == 1)
      return s_stream_r[i_fh].full();
    else
      return s_stream_c[i_fh][i_fw].full();
  }

  /* Fill the buffer for init */
  void FillLineBuffer(hls::stream<t_stream> &i_data) {
    for (uint8_t s_index = c_index - 1; s_index > 0; s_index--) {
      ap_uint<4> s_addr = c_addr[s_index](3, 0);
      do {
        SetLineBuffer(s_addr(3, 2), s_addr(1, 0), i_data.read());
      } while (!FullLineBuffer(s_addr(3, 2), s_addr(1, 0)));
    }
  }

  /* Fill and retrieve */
  t_stream PopFirst() { return s_stream_c[c_fh - 1][c_fw - 2].read(); }

  void PushFirst(t_stream i_write) { s_stream_c[0][0].write(i_write); }

  /* Fill and retrieve */
  t_stream GetLineBuffer(ap_uint<2> i_fh, ap_uint<2> i_fw) {
#pragma HLS inline
    if (i_fh == 3) return 0;

    if (i_fw(1, 1) == 1)
      return s_stream_r[i_fh].read();
    else
      return s_stream_c[i_fh][i_fw].read();
  }

  bool EmptyLineBuffer(ap_uint<2> i_fh, ap_uint<2> i_fw) {
    if (i_fh == 3) return 0;

    if (i_fw(1, 1) == 1)
      return s_stream_r[i_fh].empty();
    else
      return s_stream_c[i_fh][i_fw].empty();
  }

  void SetLineBuffer(ap_uint<2> i_fh, ap_uint<2> i_fw, t_stream i_write) {
#pragma HLS inline
    if (i_fh == 3) return;

    if (i_fw(1, 1) == 1)
      s_stream_r[i_fh].write(i_write);
    else
      s_stream_c[i_fh][i_fw].write(i_write);
  }

  t_stream ShiftLineBuffer(uint8_t i_index) {
#pragma HLS inline
    ap_uint<8> s_addr = c_addr[i_index - 1];
    t_stream s_stream = GetLineBuffer(s_addr(3, 2), s_addr(1, 0));
    SetLineBuffer(s_addr(7, 6), s_addr(5, 4), s_stream);

    return s_stream;
  }

  /* Empty the buffer */
  void EmptyLineBuffer(hls::stream<t_stream> &o_data) {
    for (uint8_t s_index = c_index - 1; s_index > 0; s_index--) {
      ap_uint<4> s_addr = c_addr[s_index](3, 0);
      do {
        o_data.write(GetLineBuffer(s_addr(3, 2), s_addr(1, 0)));
      } while (!EmptyLineBuffer(s_addr(3, 2), s_addr(1, 0)));
    }
  }

  void EmptyLineBuffer() {
    for (uint8_t s_index = c_index - 1; s_index > 0; s_index--) {
      ap_uint<4> s_addr = c_addr[s_index](3, 0);
      do {
        GetLineBuffer(s_addr(3, 2), s_addr(1, 0));
      } while (!EmptyLineBuffer(s_addr(3, 2), s_addr(1, 0)));
    }
  }

#ifndef __SYNTHESIS__
  void PrintNumData() {
    std::cout << "-----------------------------" << std::endl;
    for (ap_int<4> s_fh = c_fh - 1; s_fh > -1; s_fh--) {
      for (ap_int<4> s_fw = c_fw - 2; s_fw > -1; s_fw--) {
        std::cout << s_stream_c[s_fh][s_fw].n_elems << std::endl;
      }

      if (s_fh > 0) {
        std::cout << s_stream_r[s_fh - 1].n_elems << std::endl;
      }
    }
  }
#endif
};

#endif
