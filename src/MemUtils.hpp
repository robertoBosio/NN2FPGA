#ifndef __MEM_UTILS__
#define __MEM_UTILS__

#include "hls_stream.h"
#include "ap_int.h"
#include "MemoryManagement.hpp"
#include <stdio.h>
#include <string.h>
#include "hls_burst_maxi.h"

template <
	int c_num_conv,
	int c_read_width
> uint8_t RoundRobin(
	hls::stream<ap_uint<c_read_width>> o_streams[c_num_conv]
) {
	#pragma HLS inline

	uint8_t s_sel = 0;

	for (int8_t s_num_conv = c_num_conv-1; s_num_conv > -1; s_num_conv--) {
		if (o_streams[s_num_conv].size()==0)
			s_sel = s_num_conv;
	}

	return s_sel;
}

template <
	int c_bits,
	int c_offset,
	int c_address
> void FillStream(
	ap_uint<c_bits> *i_data,
	uint32_t &i_address,
	hls::stream<ap_uint<c_bits>> &o_data
) {

#pragma HLS inline
	while(!o_data.full()) {
		o_data.write(i_data[i_address]);
		i_address = i_address++;
		i_address = (i_address == c_address) ? i_address : c_offset;
	}

}

template <
	class t_input,
	class t_output,
	int c_ich,
	int c_och,
	int c_ow,
	int c_oh,
	int c_fw,
	int c_fh,
	int c_ops,
	int c_bits
> void ProduceStream(
	hls::stream<t_input> &i_data,
	hls::stream<t_output> o_data[c_fh*c_fw]
) {

/* #pragma HLS inline */
	const int c_index = c_fh*c_fw;
	const int c_ch = c_ich*c_och;
	const int c_bytes = c_bits/(8);
	const int c_pack_inv  = c_bytes/(c_ops);
	const int c_inv = (c_pack_inv == 0) ? 1 : c_pack_inv;

	const int c_pack = c_ops/(c_bytes);
	const int c_read = (c_pack == 0) ? 1 : c_pack;

	const int c_o_index = c_oh*c_ow*c_ch/(c_ops);
	const int c_r_index = c_index*c_oh*c_ow*c_ch/(c_ops);
	const int c_buffer = c_bytes/(c_index*c_ops) + 1;
	
	/* const ap_uint<c_ops*8> c_mask = c_ops*256-1; */

	/* Maximum input bandwidth is 64bytes */
	t_input s_tmp;
	for (auto s_r_index = 0; s_r_index < c_r_index; s_r_index++) { 
#pragma HLS pipeline
		uint8_t s_inv   = s_r_index%c_inv;
		uint8_t s_index = s_r_index%c_index;
		if (s_inv == 0)
			s_tmp = i_data.read();
		t_output s_data;
		for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
			s_data[s_ops] = s_tmp & 0xff;
			s_tmp >>=8;
		}
		o_data[s_index].write(s_data);
	}

}

template <
	int data_width,
	int num_streams
> void MemAlgo(
	hls::stream<ap_uint<data_width>> o_streams[num_streams],
	hls::burst_maxi<ap_uint<READ_WIDTH>> i_data
) {

	const int c_bytes = data_width/8;
	const int c_words = 4096/(c_bytes);
	static uint32_t s_read_address[num_streams]; 

#ifndef __SYNTHESIS__
	int iteration = 0;
#endif

	for (auto s_sel = 0; s_sel < num_streams; s_sel++)
		s_read_address[s_sel] = c_address_start[s_sel];

	do {
		for (auto s_sel = 0; s_sel < num_streams; s_sel++) {
			if (!o_streams[s_sel].full()) {
				uint32_t c_read = (c_address_end[s_sel] - s_read_address[s_sel])/c_bytes;
				c_read = (c_read < c_words) ? c_read : c_words;
				i_data.read_request(s_read_address[s_sel]/c_bytes, c_read);
				bool s_full = false;
				for (auto s_words = 0; s_words < c_read; s_words++) {
#pragma HLS pipeline
					ap_uint<data_width> s_data = i_data.read();
					s_full = s_full | o_streams[s_sel].full();
					if (!s_full) {
						o_streams[s_sel].write(s_data);
					}
				}
				if (c_read < c_words) {
					s_read_address[s_sel] = c_address_start[s_sel];
				} else {
					s_read_address[s_sel] += c_read*c_bytes;
				}
			}
		}
#ifndef __SYNTHESIS__
		iteration++;
	} while(iteration < 10000);
#endif
#ifdef __SYNTHESIS__
	} while(1);
#endif

}

template <
	int c_ich,
	int c_och,
	int c_ow,
	int c_oh,
	int c_fw,
	int c_fh,
	int c_ops,
	int c_bits,
	int c_start
> void MemAlgo(
	hls::stream<ap_uint<c_bits>> &o_data,
	hls::burst_maxi<ap_uint<c_bits>> i_data
) {

	const int c_bytes = c_bits/8;
	const int c_words = 4096/(c_bytes);
	const int c_f_index = c_fh*c_fw*c_och*c_ich;
	const int c_b_index = c_f_index / 4096;
	const int c_b_rem = c_f_index % 4096;
	const int c_start_w = c_start/c_bytes;
	const int c_b_rem_words = c_b_rem/c_bytes;
	const int c_o_index = c_ow*c_oh;

	for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
		for (auto s_b_index = 0; s_b_index < c_b_index; s_b_index++) {
			uint32_t s_read = c_start_w + s_b_index*c_words;
			i_data.read_request(s_read, c_words);
			for (auto s_words = 0; s_words < c_words; s_words++) {
#pragma HLS pipeline
				ap_uint<c_bits> s_data = i_data.read();
				o_data.write(s_data);
			}
		}
		if (c_b_rem != 0) {
			uint32_t c_rem_start = c_start_w + c_b_index*c_words;
			i_data.read_request(c_rem_start, c_b_rem_words);
			for (auto s_words = 0; s_words < c_b_rem_words; s_words++) {
#pragma HLS pipeline
				ap_uint<c_bits> s_data = i_data.read();
				o_data.write(s_data);
			}
		}
	}

}


#endif

