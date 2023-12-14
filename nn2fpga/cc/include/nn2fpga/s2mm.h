#include "hls_stream.h"

namespace nn2fpga {
template<typename T_mem, typename T_stream>
void
s2mm(T_mem* mem, const unsigned int size, hls::stream<T_stream>& stream)
{
#pragma HLS INLINE
  auto p = 0;
  T_stream v;
S2MM_LOOP:
  do {
#pragma HLS pipeline II = 1
    v = stream.read();
    mem[p] = v.data;
    p++;
#ifndef __SYNTHESIS__
    if (p > size)
      std::cout << "ERROR: s2mm: p > size\n";
#endif
  } while (!v.last);
}

} // namespace nn2fpga
