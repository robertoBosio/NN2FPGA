#ifndef __POOLSTREAM__
#define __POOLSTREAM__

#include "ap_int.h"
#include "hls_stream.h"

template <
	class t_input,
	class t_acc,
	int c_ich,
	int c_och,
	int c_ih,
	int c_iw,
	int c_oh,
	int c_ow,
	int c_fh,
	int c_fw,
	int c_str,
	int c_pad,
	int c_bypass
> void AveragePoolOp(
	hls::stream<t_input> &i_data,
	hls::stream<t_acc> &o_acc
) {

#pragma HLS PIPELINE off
	/* const int c_starth = (c_fh-1)*(1-c_pad); */
	/* const int c_startw = (c_fh-1)*(1-c_pad); */
	/* const int c_bypass_w = c_fw - 1; */
	/* const int c_ih_pad = (c_ih + (c_fh-1)*(c_pad)); */
	/* const int c_iw_pad = (c_iw + (c_fw-1)*(c_pad)); */
	/* const int c_paddingh_shift = c_bypass*c_iw_pad*c_ich; */
	/* const int c_paddingw_shift = c_bypass_w*c_ich; */
	/* const int c_strideh_shift = (c_str-1)*c_iw_pad*c_ich; */
	/* const int c_stridew_shift = (c_str-1)*c_ich; */

	const int c_pad_index_h = c_pad * (c_fh - 1);
	const int c_pad_index_w = c_pad * (c_fw - 1);
	const int c_ih_pad = c_ih + c_pad_index_h;
	const int c_iw_pad = c_iw + c_pad_index_w;

	const int c_index_i = c_iw_pad * c_ih_pad * c_ich;
	const int c_index_o = c_ow * c_oh * c_och;

	for (uint32_t s_index = 0; s_index < c_index_i; s_index++) {

		t_input s_data = i_data.read();

	}

	for (uint32_t s_index = 0; s_index < c_index_o; s_index++) {

			o_acc.write(0);

	}

	EmptyStream<t_input>(i_data);

#ifndef __SYNTHESIS__
	std::cout << "AVERAGEOP: " << c_ih << " " << c_iw << " " << c_ich << " " << c_str << " " << c_pad << " " << std::endl;
#endif

}

template <
	class t_acc,
	class t_output,
	int c_och,
	int c_oh,
	int c_ow,
	int c_fh,
	int c_fw
> void WriteOutput(
	hls::stream<t_acc> &i_data,
	hls::stream<t_output> &o_data
) {

	for (uint8_t s_oh = 0; s_oh < c_oh; s_oh++) {
		for (uint8_t s_ow = 0; s_ow < c_ow; s_ow++) {
			for (uint8_t s_och = 0; s_och < c_och; s_och++) {
#pragma HLS loop_flatten
#pragma HLS PIPELINE off
				t_acc s_acc = i_data.read();
				o_data.write((t_output)(s_acc));
			}
		}
	}
}

template <
	class t_input,
	class t_acc,
	class t_output,
	int c_ich,
	int c_och,
	int c_ih,
	int c_iw,
	int c_oh,
	int c_ow,
	int c_fh,
	int c_fw,
	int c_str,
	int c_pad
> void AveragePoolKernel8x8(
	hls::stream<t_input> &i_data,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;

#pragma HLS inline

	hls::stream<t_acc> s_acc("s_acc");
	#pragma HLS STREAM variable=s_acc depth=2

	AveragePoolOp<
		t_input,
		t_acc,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_oh,
		c_ow,
		c_fh,
		c_fw,
		c_str,
		c_pad,
		0
	> (
		i_data,
		s_acc
	);

	WriteOutput<
		t_acc,
		t_output,
		c_och,
		c_oh,
		c_ow,
		c_fh,
		c_fw
	> (
		s_acc,
		o_data
	);

}

template <
	class t_input,
	class t_output,
	class t_acc,
	int c_ich,
	int c_och,
	int c_iw,
	int c_ih,
	int c_ow,
	int c_oh,
	int c_fw,
	int c_fh,
	int c_str,
	int c_pad
> void AveragePoolStreams(
	hls::stream<t_input> &i_data,
	hls::stream<t_output> &o_data
) {

#pragma HLS inline

#ifndef __SYNTHESIS__

	while(i_data.empty());

#endif

	AveragePoolKernel8x8 <
		t_input,
		t_acc,
		t_output,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_oh,
		c_ow,
		c_fh,
		c_fw,
		c_str,
		c_pad
	> (
		i_data,
		o_data
	);

}

#endif

