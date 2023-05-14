#ifndef NN2FPGA_STREAM_TO_PT_H_
#define NN2FPGA_STREAM_TO_PT_H_

namespace nn2fpga {

// For output activations
template <typename din_t, typename dout_t, int OCH, int OW, int OH>
void consume_stream(hls::stream<din_t> &din, dout_t o_data[OCH * OW * OH]) {
  din_t s_read;
  constexpr unsigned ISZ = OCH * OH * OW;

  for (int s_index = 0; s_index < ISZ; s_index++) {
    s_read = din.read();
    o_data[s_index] = (dout_t)(s_read);
  }
}

template <typename din_t, typename dout_t, int OCH, int OW, int OH>
void consume_stream(hls::stream<din_t> &din, hls::stream<dout_t> &o_data) {
  constexpr unsigned ISZ = OCH * OH * OW;

  for (int s_index = 0; s_index < ISZ; s_index++) {
    din_t s_read = din.read();
    dout_t tmp;
    tmp.data = s_read.data;
    tmp.last = s_read.last;
    tmp.keep = -1;
    o_data.write(tmp);
  }
}

template <typename din_t, typename dout_t, int OCH, int OW, int OH>
void consume_stream(hls::stream<din_t> &din, hls::stream<ap_uint<1>> &i_last,
                    hls::stream<dout_t> &o_data) {
  constexpr unsigned OSZ = OCH * OH * OW;

  for (int s_index = 0; s_index < OSZ - 1; s_index++) {
    din_t s_read = din.read();
    dout_t tmp;
    tmp.data = s_read;
    tmp.last = false;
    tmp.keep = -1;
    o_data.write(tmp);
  }

  dout_t tmp;
  din_t s_read = din.read();
  tmp.data = s_read;
  tmp.last = i_last.read();
  tmp.keep = -1;
  o_data.write(tmp);
}

template <typename din_t, typename dout_t, int OCH, int OW, int OH, int c_bits>
void consume_stream(hls::stream<din_t> &din, hls::stream<ap_uint<1>> &i_last,
                    hls::stream<dout_t> &o_data) {
  constexpr unsigned c_par = c_bits / 8;
  constexpr unsigned OSZ = OCH * OH * OW / c_par;
  constexpr unsigned c_out_pad = (OCH * OH * OW) % c_par;

  for (int s_index = 0; s_index < OSZ; s_index++) {
    dout_t tmp;
    for (int s_par = 0; s_par < c_par; s_par++) {
#pragma HLS pipeline
      din_t s_read = din.read();
      tmp.data((s_par + 1) * 8 - 1, s_par * 8) = s_read;
    }
    tmp.keep = -1;
    tmp.last = false;
    o_data.write(tmp);
  }

  dout_t tmp;
  tmp.data = 0;
  for (int s_out_pad = 0; s_out_pad < c_out_pad; s_out_pad++) {
#pragma HLS pipeline
    din_t s_read = din.read();
    tmp.data((s_out_pad + 1) * 8 - 1, s_out_pad * 8) = s_read;
  }
  /* The input last stream doesn't count the number of activations streamed but
   * the */
  /* number of batches analyzed */
  tmp.keep = -1;
  tmp.last = i_last.read();
  o_data.write(tmp);
}



} // namespace nn2fpga
 
#endif // NN2FPGA_STREAM_TO_PT_H_
