#ifndef NN2FPGA_ADD_STREAMS_H_
#define NN2FPGA_ADD_STREAMS_H_

// Vitis HLS dependencies.
#include <ap_int.h>
#include <hls_stream.h>

namespace nn2fpga {

template <typename din_t, typename dout_t, unsigned ICH, unsigned IH,
          unsigned IW>
void add_streams(hls::stream<din_t>& dinStream1, hls::stream<din_t>& dinStream2,
                 hls::stream<dout_t>& doutStream) {
  if (dinStream1.empty() || dinStream2.empty()) return;

  constexpr int ISZ = ICH * IH * IW;

  for (int i = 0; i < ISZ; i++) {
    auto din1 = dinStream1.read(), din2 = dinStream2.read();
    dout_t dout = din1 + din2;
    doutStream << dout;
  }
}

}  // namespace nn2fpga

#endif  // NN2FPGA_ADD_STREAMS_H_
