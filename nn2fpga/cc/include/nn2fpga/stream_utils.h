#ifndef NN2FPGA_STREAM_UTILS_H_
#define NN2FPGA_STREAM_UTILS_H_

#include "nn2fpga/debug.h"
#include "nn2fpga/quantisation.h"

namespace nn2fpga {

// Read a stream, quantise it, stream it out.
template <typename din_wrap_t, typename din_t, typename dout_wrap_t,
          typename dout_t, typename d_format_t, unsigned ICH, unsigned IW, unsigned IH,
          unsigned c_ow_ops_out, unsigned BITS, unsigned OPS, unsigned PREPROC>
void produce_stream(hls::stream<din_wrap_t>& dinStream,
                    hls::stream<dout_wrap_t> doutStream[c_ow_ops_out]) {
  constexpr auto PAR = BITS / 8;
  constexpr auto ISZ = (ICH * IH * IW);
  const ap_fixed<16,8> c_mean[3] = {0.485, 0.456, 0.406};
  const ap_fixed<16,8> c_std[3] = {0.229, 0.224, 0.225};

  #ifndef __SYNTHESIS__
      std::cout << "produce_stream act " << ICH << " " << c_ow_ops_out << std::endl;
  #endif
  din_wrap_t dinWrap;
	ap_uint<BITS> din_par;
PRODSTR:
  for (auto i = 0; i < ISZ; i++) {
#pragma HLS pipeline style = stp
    auto par = i % PAR;
    auto ops = i % OPS;
    auto ich = i % ICH;
    auto ow_ops_out = (i / ICH) % c_ow_ops_out;

    ap_ufixed<8,0> din;
    if (par == 0) {
      dinWrap = dinStream.read();
      din_par = dinWrap.data;
    }

    dout_wrap_t doutWrap;
    din.range(7,0) = din_par & 0xff;
    if constexpr(PREPROC == 1)
      doutWrap.data[0][ops] = dout_t((din-c_mean[ich])/c_std[ich]);
    else
      doutWrap.data[0][ops] = (dout_t(din));
    #ifndef __SYNTHESIS__
      #ifdef DEBUG
        std::cout << din << " ";
        std::cout << doutWrap.data[0][ops] << " ";
      #endif
    #endif

    if (par < (PAR - 1)) {
      doutWrap.last = false;
    } else {
      doutWrap.last = dinWrap.last;
    }

    if (ops == (OPS - 1)) {
      doutStream[ow_ops_out] << doutWrap;
      #ifndef __SYNTHESIS__
        #ifdef DEBUG
          std::cout << std::endl;
        #endif
      #endif
    }
    din_par >>= 8;
  }
  #ifndef __SYNTHESIS__
      std::cout << "end produce_stream act " << std::endl;
  #endif
}

// Translate the stream to an array.
template <typename din_t, typename dout_t, int OCH, int OW, int OH, int ow_ops>
void consume_stream(hls::stream<din_t> dinStream[ow_ops], dout_t dout[OCH * OW * OH]) {
  constexpr unsigned OSZ = OCH * OH * OW;

  for (auto i = 0; i < OSZ; i++) {
    dout[i] = dout_t(dinStream[0].read());
  }
}

template <typename din_wrap_t, typename dout_wrap_t, int OCH, int OW, int OH, int ow_ops, int OPS>
void consume_stream(hls::stream<din_wrap_t> dinStream[ow_ops],
                    hls::stream<dout_wrap_t>& doutStream) {
  constexpr unsigned OSZ = OCH * OH * OW / OPS;

  // If synthesis pragma in not defined then print consume_stream
  // function name
  #ifndef __SYNTHESIS__
      std::cout << "consume_stream " << OCH << std::endl;
  #endif
  din_wrap_t wrap;
  for (auto i = 0; i < OSZ; i++) {
    for (auto s_ops = 0; s_ops < OPS; s_ops++) {
      if (s_ops == 0)
        wrap = dinStream[0].read();
      dout_wrap_t dout;
      dout.data = wrap.data[0][s_ops];
      #ifndef __SYNTHESIS__
        #ifdef DEBUG
        // if (c_depth == 1)
          std::cout << dout.data << std::endl;
        #endif
      #endif
      dout.last = wrap.last & (i == (OSZ - 1));
      dout.keep = -1;
      dout.strb = 1;
      doutStream << dout;
    }
  }
  #ifndef __SYNTHESIS__
      std::cout << "end consume_stream " << std::endl;
  #endif
}

}  // namespace nn2fpga

#endif  // NN2FPGA_STREAM_UTILS_H_
