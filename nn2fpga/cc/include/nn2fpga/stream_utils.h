#ifndef NN2FPGA_STREAM_UTILS_H_
#define NN2FPGA_STREAM_UTILS_H_

#include "nn2fpga/quantisation.h"

namespace nn2fpga {

// Read a stream, quantise it, stream it out.
template <typename din_wrap_t, typename din_t, typename dout_wrap_t,
          typename dout_t, unsigned ICH, unsigned IW, unsigned IH,
          unsigned c_ws_out, unsigned BITS, unsigned OPS>
void produce_stream(hls::stream<din_wrap_t>& dinStream,
                    hls::stream<dout_wrap_t> doutStream[c_ws_out]) {
  constexpr auto PAR = BITS / 8;
  constexpr auto ISZ = (ICH * IH * IW);

  din_wrap_t dinWrap;
	ap_uint<BITS> din_par;
PRODSTR:
  for (auto i = 0; i < ISZ; i++) {
#pragma HLS pipeline style = stp
    auto par = i % PAR;
    auto ops = i % OPS;
    auto ws_out = (i / ICH) % c_ws_out;

    ap_ufixed<8,0, AP_RND, AP_SAT> din;
    if (par == 0) {
      dinWrap = dinStream.read();
      din_par = dinWrap.data;
    }

    dout_wrap_t doutWrap;
    din.range(7,0) = din_par & 0xff;
    doutWrap.data[0][ops] = dout_t(din);

    if (par < (PAR - 1)) {
      doutWrap.last = false;
    } else {
      doutWrap.last = dinWrap.last;
    }

    doutStream[ws_out] << doutWrap;
    din_par >>= 8;
  }
}

// Translate the stream to an array.
template <typename din_t, typename dout_t, int OCH, int OW, int OH, int WS>
void consume_stream(hls::stream<din_t> dinStream[WS], dout_t dout[OCH * OW * OH]) {
  constexpr unsigned OSZ = OCH * OH * OW;

  for (auto i = 0; i < OSZ; i++) {
    dout[i] = dout_t(dinStream[0].read());
  }
}

template <typename din_wrap_t, typename dout_wrap_t, int OCH, int OW, int OH, int WS>
void consume_stream(hls::stream<din_wrap_t> dinStream[WS],
                    hls::stream<dout_wrap_t>& doutStream) {
  constexpr unsigned OSZ = OCH * OH * OW;

  for (auto i = 0; i < OSZ; i++) {
    auto wrap = dinStream[0].read();
    dout_wrap_t dout;
    dout.data = wrap.data[0][0];
    dout.last = wrap.last & (i == (OSZ - 1));
    dout.keep = -1;
    doutStream << dout;
  }
}

}  // namespace nn2fpga

#endif  // NN2FPGA_STREAM_UTILS_H_
