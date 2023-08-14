#ifndef NN2FPGA_UTILS_H_
#define NN2FPGA_UTILS_H_

#include <etc/autopilot_ssdm_op.h>

#include "ap_int.h"
#include "hls_math.h"
#include "nn2fpga/debug.h"
// #include "nn2fpga/line_buffer.h"

namespace nn2fpga {

//////////////////////////// QUANT FUNCTIONS //////////////////////////////////

//////////////////////////// FROM POINTER TO STREAM ///////////////////////////
// For input activations
/* template < */
/* 	typename din_t, */
/* 	typename dout_t, */
/* 	int ICH, */
/* 	int IW, */
/* 	int IH */
/* > void produce_stream( */
/* 	din_t *din, */
/* 	hls::stream<dout_t> &s_din */
/* ) { */

/* 	const int c_index = ICH*IH*IW; */

/* 	PRODSTR: for (int s_index = 0; s_index < c_index; s_index++) { */
/* 		s_din.write((dout_t)(din[s_index])); */
/* 	} */

/* } */

///////////////////////////// FROM STREAM TO POINTER //////////////////////////

///////////////////////////// FOR INPUT WEIGHTS //////////////////////////
///////////////////////////// FROM STREAM TO STREAM ///////////////////////////

template <typename din_t, typename dout_t, int ICH, int OCH, int IW, int IH,
          int OW, int OH, int c_pad>
void pad_stream(hls::stream<din_t> &din, hls::stream<dout_t> &o_data) {
  if (c_pad == 0) {
    for (uint8_t s_ih = 0; s_ih < IH; s_ih++) {
      for (uint8_t s_iw = 0; s_iw < IW; s_iw++) {
        for (uint8_t s_ich = 0; s_ich < ICH; s_ich++) {
          din_t s_input = din.read();
          dout_t s_output;
          s_output.data = s_input.data;
          s_output.last = s_input.last;
          o_data.write(s_output);
        }
      }
    }
  }

  if (c_pad == 1) {
    for (uint8_t s_ih = 0; s_ih < IH; s_ih++) {
      for (uint8_t s_iw = 0; s_iw < IW; s_iw++) {
        for (uint8_t s_ich = 0; s_ich < ICH; s_ich++) {
          din_t s_input = din.read();
          dout_t s_output;
          s_output.data = s_input.data;
          s_output.last = s_input.last;
          o_data.write(s_output);
        }
      }
    }
  }
}

//////////////////////////// BLOCK INTERFACES /////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
/* Rearranging activations for weights reusage generation */

//////////////////////////////////////////////////////////////////////////////
/* Line Buffers generation */
/* template < */
/* 	typename din_t, */
/* 	int ICH, */
/* 	int OCH, */
/* 	int IH, */
/* 	int IW, */
/* 	int OH, */
/* 	int OW, */
/* 	int c_fh, */
/* 	int c_fw, */
/* 	int c_str, */
/* 	int c_pad */
/* > void shift_op( */
/* 	hls::stream<din_t> &din, */
/* 	hls::stream<din_t> o_compute[c_fh*c_fw] */
/* ) { */
/* /1* #pragma HLS inline *1/ */

/* 	const auto c_starth = (c_fh-1)*(1-c_pad); */
/* 	const auto c_startw = (c_fw-1)*(1-c_pad); */
/* 	const auto c_pad_index_h = c_pad * (c_fh - 1) / 2; */
/* 	const auto c_pad_index_w = c_pad * (c_fw - 1) / 2; */
/* 	const auto IH_PAD = IH + c_pad_index_h*2; */
/* 	const auto IW_PAD = IW + c_pad_index_w*2; */
/* 	const auto c_strideh_shift = (c_str-1); */
/* 	const auto c_stridew_shift = (c_str-1); */

/* 	/1* Constants for new version *1/ */
/* 	const auto c_i_index = IH_PAD*IW_PAD*ICH; */
/* 	const auto c_index = c_fh*c_fw; */

/*   const auto c_size = (c_fh-1*IW+fw-1)*ICH; */
/* 	din_t s_data[c_size]; */
/*   const auto s_start = -1*c_size; */
/*   auto s_address = s_start; */

/* 	for (auto s_index_h = 0; s_index_h < IH; s_index_h++) { */
/* 		for (auto s_index_w = 0; s_index_w < IW; s_index_w++) { */
/* 			for (auto s_index_ich = 0; s_index_ich < ICH;
 * s_index_ich++) { */
/* #pragma HLS pipeline style=frp */
/* 				for (auto s_fh=0; s_fh<c_fh; s_fh++) { */
/*           auto s_addr_h = s_address+s_fh*IW*ICH */
/* 					for (auto s_fw=0; s_fw<c_fw; s_fw++) {
 */
/* 						auto s_addr_w =
 * (s_addr_h+s_fw*ICH) % c_size; */
/* 						din_t s_input; */
/*             auto s_index = s_fh*c_fw+s_fw; */
/*             s_input = s_data[s_index] */
/* 						if (s_addr_w > 0) */
/* 							o_compute[s_index] =
 * din.read();
 */

/* 					} */
/* 				} */
/*         s_address++; */
/* 			} */
/* 		} */
/* 	} */

/* } */

}  // namespace nn2fpga

#endif  // NN2FPGA_UTILS_H_
