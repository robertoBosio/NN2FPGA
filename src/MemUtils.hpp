#ifndef __UTILS__
#define __UTILS__

#include "hls_stream.h"
#include "ap_int.h"

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
#pragma HLS pipeline
		o_data.write(i_data[i_address]);
		i_address = i_address++;
		i_address = (i_address == c_address) ? i_address : c_offset;
	}

}


template <
	class t_output,
	int c_fw,
	int c_fh,
	int c_ops,
	int c_bits
> void ProduceStream(
	hls::stream<ap_uint<c_bits>> &i_data,
	hls::stream<t_output> o_data[c_fh*c_fw]
) {

	const int c_index = c_fh*c_fw;
	const int c_par   = c_ops*8;
	const int c_iter  = c_bits/(c_ops*8);

	uint8_t s_index = 0;

	do {
		ap_uint<c_bits> s_read = i_data.read();
		
		for (uint8_t s_iter = 0; s_iter < c_iter; s_iter++) {
#pragma HLS pipeline
			ap_uint<c_par> s_o_read = s_read((s_iter+1)*c_par-1, c_par*s_iter);
			o_data[s_index].write(s_o_read);

			s_index++;
			if (s_index == c_index)
				s_index = 0;
		}

	} while(1);

}
#endif

