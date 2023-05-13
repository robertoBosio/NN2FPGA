#ifndef __ADDSTREAM__
#define __ADDSTREAM__

#include <ap_int.h>
#include <hls_stream.h>

template <typename din_t, typename dout_t, int C_ICH, int C_IH,
          int C_IW>
void AddStreams(hls::stream<din_t>& dinStream1, hls::stream<din_t>& dinStream2,
                hls::stream<dout_t>& doutStream) {
  if (dinStream1.empty() || dinStream2.empty()) return;

  constexpr int C_INDEX = C_ICH * C_IH * C_IW;

  for (int c = 0; c < C_INDEX; c++) {
    auto din1 = dinStream1.read(), din2 = dinStream2.read();
    dout_t dout = din1 + din2;
    doutStream << dout;
  }
}

#endif  // __ADDSTREAM__
