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
#ifndef __SYNTHESIS__
  if (stream.size() > 0)
    std::cout << "ERROR: s2mm: stream.size() > 0\n";
  assert(stream.size() == 0);
#endif
}

} // namespace nn2fpga
