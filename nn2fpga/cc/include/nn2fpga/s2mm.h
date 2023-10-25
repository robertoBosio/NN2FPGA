#include "hls_stream.h"

namespace nn2fpga {
template<typename T_mem, typename T_stream>
void
s2mm(T_mem* mem, const unsigned int size, hls::stream<T_stream>& stream)
{
#pragma HLS INLINE
S2MM_LOOP:
  for (int i = 0; i < size; i++) {
#pragma HLS pipeline II = 1
    T_stream v = stream.read();
    mem[i] = v.data;
  }
}

} // namespace nn2fpga
