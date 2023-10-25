#include "hls_stream.h"

namespace nn2fpga {
template<typename T_mem, typename T_stream>
void
mm2s(const T_mem* mem, const unsigned int size, hls::stream<T_stream>& stream)
{
#pragma HLS INLINE
MM2S_LOOP:
  for (auto it = 0; it < size; ++it) {
#pragma HLS pipeline II = 1
    T_stream s_data;
    auto data = mem[it];
    s_data.data = data;
    s_data.last = (it == (size - 1));
    stream.write(s_data);
  }
}
} // namespace nn2fpga
