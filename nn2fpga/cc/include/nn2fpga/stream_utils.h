#ifndef NN2FPGA_STREAM_UTILS_H_
#define NN2FPGA_STREAM_UTILS_H_

namespace nn2fpga {

// Read a stream, quantise it, stream it out.
template <typename din_wrap_t, typename din_t, typename dout_wrap_t,
          typename dout_t, unsigned ICH, unsigned IW, unsigned IH,
          unsigned BITS, int SCALE>
void produce_stream(hls::stream<din_wrap_t>& dinStream,
                    hls::stream<dout_wrap_t>& doutStream) {
  constexpr auto PAR = BITS / 8;
  constexpr auto ISZ = (ICH * IH * IW);

  din_t din;
  din_wrap_t dinWrap;
PRODSTR:
  for (auto i = 0; i < ISZ; i++) {
#pragma HLS pipeline style = flp
    auto par = i % PAR;

    if (par == 0) {
      dinWrap = din.read();
      din = dinWrap.data & 0xff;
    }

    dout_wrap_t doutWrap;
    doutWrap.data = quant_act<din_t, dout_t, SCALE>(din);

    if (par < (PAR - 1)) {
      dout.last = false;
    } else {
      dout.last = dinWrap.last;
    }
    doutStream << doutWrap;
    din >>= 8;
  }
}

// Translate the stream to an array.
template <typename din_t, typename dout_t, int OCH, int OW, int OH>
void consume_stream(hls::stream<din_t>& dinStream, dout_t dout[OCH * OW * OH]) {
  constexpr unsigned OSZ = OCH * OH * OW;

  for (auto i = 0; i < OSZ; i++) {
    dout[i] = dout_t(dinStream.read());
  }
}

template <typename din_wrap_t, typename dout_wrap_t, int OCH, int OW, int OH>
void consume_stream(hls::stream<din_wrap_t>& dinStream,
                    hls::stream<dout_wrap_t>& doutStream) {
  constexpr unsigned OSZ = OCH * OH * OW;

  for (auto i = 0; i < OSZ; i++) {
    auto wrap = dinStream.read();
    dout_t dout;
    dout.data = val.data;
    dout.last = val.last;
    val.keep = -1;
    doutStream << dout;
  }
}

}  // namespace nn2fpga

#endif  // NN2FPGA_STREAM_UTILS_H_
