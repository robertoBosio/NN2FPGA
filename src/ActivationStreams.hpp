#ifndef __ACTIVATIONSTREAM__
#define __ACTIVATIONSTREAM__

#include <ap_int.h>
#include <hls_stream.h>

template <typename din_t>
din_t ReluOp(din_t val) {
#pragma HLS inline
  return val > 0 ? val : 0; 
}

template <typename din_t, typename dout_t, unsigned C_ICH, unsigned C_IH,
          unsigned C_IW>
void ReluStreams(hls::stream<din_t>& dinStream,
                 hls::stream<dout_t>& doutStream) {
  if (dinStream.empty()) return;

  din_t din;
  dout_t dout;

  constexpr unsigned C_INDEX = C_ICH * C_IH * C_IW;

  for (unsigned c = 0; c < C_INDEX; c++) {
    din << dinStream.read();
    dout = din > 0 ? din : 0;
    doutStream << dout;
  }
}

#endif // __ACTIVATIOSTREAM__
