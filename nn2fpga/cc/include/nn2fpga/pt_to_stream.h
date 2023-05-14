#ifndef NN2FGPA_PT_TO_STREAM_H_
#define NN2FGPA_PT_TO_STREAM_H_

#include "ap_int.h"
#include "hls_stream.h"
#include "nn2fpga/block_interface.h"
#include "nn2fpga/line_buffer_utils.h"
#include "nn2fpga/quantisation.h"

namespace nn2fpga {

template <typename din_t, typename dout_t, unsigned ICH, unsigned IW,
          unsigned IH>
void produce_stream(din_t din[ICH * IH * IW], hls::stream<ap_uint<1>> &i_last,
                    hls::stream<dout_t> &doutStream) {
  constexpr unsigned ISZ = ICH * IH * IW;

  /* while(1) { */
PRODSTR:
  for (unsigned i = 0; i < ISZ; i++) {
    doutStream << dout_t(din[i]);
  }
  ap_uint<1> s_last = i_last.read();
  /* if (s_last) */
  /* 	break; */
  /* } */
}

template <typename din_t, typename part_t, typename t_output_struct,
          typename dout_t, unsigned ICH, unsigned IW, unsigned IH,
          unsigned BITS, int SCALE>
void produce_stream(hls::stream<din_t> &din,
                    hls::stream<t_output_struct> &o_data) {
  constexpr int PAR = BITS / 8;
  constexpr unsigned ISZ = (ICH * IH * IW);

  din_t tmp_r;
  ap_uint<BITS> tmp_r_par;
PRODSTR:
  for (unsigned i = 0; i < ISZ; i++) {
#pragma HLS pipeline style = flp
    uint8_t s_par = i % PAR;

    if (s_par == 0) {
      tmp_r = din.read();
      tmp_r_par = tmp_r.data;
    }

    t_output_struct tmp_w;
    part_t tmp_p = (part_t)(tmp_r_par & 0xff);
    tmp_w.data = quant_act<part_t, SCALE, dout_t>(tmp_p);

    if (s_par < (PAR - 1))
      tmp_w.last = false;
    else
      tmp_w.last = tmp_r.last;
    o_data.write(tmp_w);
    tmp_r_par = tmp_r_par >> 8;
  }
}

template <typename din_t, typename dout_t, int ICH, int IW, int IH, int BITS>
void produce_stream(hls::stream<din_t> &din, hls::stream<ap_uint<1>> &o_last,
                    hls::stream<dout_t> &o_data) {
  constexpr unsigned c_par = BITS / 8;
  constexpr unsigned ISZ = (ICH * IH * IW) / c_par;

  din_t tmp_r;
PRODSTR:
  for (int i = 0; i < ISZ; i++) {
    tmp_r = din.read();
    ap_uint<64> tmp_r_par = tmp_r.data;

    for (uint8_t s_par = 0; s_par < c_par; s_par++) {
#pragma HLS pipeline off
      /* dout_t tmp_w = (dout_t)(tmp_r.data(8*(s_par+1)-1,8*s_par)); */
      dout_t tmp_w = (dout_t)(tmp_r_par & 0xff);
      o_data.write(tmp_w);
      tmp_r_par = tmp_r_par >> 8;
    }
  }

  o_last.write(tmp_r.last);
}

}  // namespace nn2fpga

#endif  // NN2FGPA_PT_TO_STREAM_H_
