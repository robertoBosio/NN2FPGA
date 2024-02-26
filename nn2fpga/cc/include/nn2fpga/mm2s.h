#include "hls_stream.h"

namespace nn2fpga {
template<typename T_mem, typename T_stream>
void
mm2s(const T_mem* mem, const unsigned int size, hls::stream<T_stream>& stream)
{

  std::cout << "INFO: Call to mm2s" << std::endl;
  std::cout << "\t\tsize: " << size << std::endl;

#pragma HLS INLINE
MM2S_LOOP:
  for (auto it = 0; it < size; ++it) {
#pragma HLS pipeline II = 1
    T_stream s_data;
    auto data = mem[it];
    s_data.data = data;
    s_data.last = (it == (size - 1));
    s_data.strb = ~0;
    s_data.keep = ~0;
    stream.write(s_data);
  }
  std::cout << "INFO: Finished mm2s" << std::endl;
}
} // namespace nn2fpga
