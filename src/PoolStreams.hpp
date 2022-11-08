#ifndef __POOLSTREAM__
#define __POOLSTREAM__

#include "hls_math.h"
#include "ap_int.h"
#include "hls_stream.h"

/* For global case to save line buffer logic */
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
	int c_pad,
	int c_bypass,
	int c_pool
> void PoolOp(
	hls::stream<t_input> &i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;
	const int c_o_index = c_oh*c_ow;
	const uint8_t c_average_scale = (uint8_t)(log2(c_fh*c_fw));

	/* while(1) { */
#ifndef __SYNTHESIS__
		while(i_data.empty());
#endif

		t_acc s_acc_buff[c_och] = {0};
		for (uint16_t s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
			for (uint8_t s_index = 0; s_index < c_index; s_index++) {
				for (uint8_t s_och = 0; s_och < c_och; s_och++) {

					t_input s_input = i_data.read();
					if (c_pool == 0) // Average Pool
						s_acc_buff[s_och] += s_input;
					if (c_pool == 1) { // Max Poool
						if (s_input > s_acc_buff[s_och])
							s_acc_buff[s_och] = s_input;
					}
				}

			}
		}

		for (uint8_t s_och = 0; s_och < c_och; s_och++) {
			if (c_pool == 0) // Average Pool
				s_acc_buff[s_och] = s_acc_buff[s_och] << c_average_scale;
			o_data.write((t_output)(s_acc_buff[s_och])); 
		}
#ifndef __SYNTHESIS__

		std::cout << "Waiting for last signal" << std::endl;

#endif

		ap_uint<1> s_last = i_last.read();
		o_last.write(s_last);
		/* if (s_last) */
		/* 	break; */

	/* } */

#ifndef __SYNTHESIS__
	EmptyStream<t_input>(i_data);
	std::cout << "POOLOP: " << c_ih << " " << c_iw << " " << c_ich << " " << c_str << " " << c_pad << " " << std::endl;
#endif

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
	int c_pad,
	int c_bypass,
	int c_pool
> void PoolOp(
	hls::stream<t_input> i_data[c_fh*c_fw],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;
	const int c_o_index = c_oh*c_ow;
	const uint8_t c_average_scale = (uint8_t)(log2(c_fh*c_fw));

	/* while(1) { */
#ifndef __SYNTHESIS__
		for (uint8_t s_index = 0; s_index < c_index; s_index++)
			while(i_data[s_index].empty());
#endif

		for (uint16_t s_o_index = 0; s_o_index < c_o_index; s_o_index++) {

			for (uint8_t s_och = 0; s_och < c_och; s_och++) {
				t_acc s_acc_buff = 0;

				for (uint8_t s_index = 0; s_index < c_index; s_index++) {
					t_input s_input = i_data[s_index].read();
					if (c_pool == 0) // Average Pool
						s_acc_buff += s_input;
					if (c_pool == 1) { // Max Poool
						if (s_input > s_acc_buff)
							s_acc_buff = s_input;
					}
				}

				if (c_pool == 0) // Average Pool
					s_acc_buff = s_acc_buff << c_average_scale;
				o_data.write((t_output)(s_acc_buff)); 
			}

		}

#ifndef __SYNTHESIS__

		std::cout << "Waiting for last signal" << std::endl;

#endif

		ap_uint<1> s_last = i_last.read();
		o_last.write(s_last);
		/* if (s_last) */
		/* 	break; */

	/* } */

#ifndef __SYNTHESIS__
	for (uint8_t s_index = 0; s_index < c_index; s_index++)
		EmptyStream<t_input>(i_data[s_index]);
	std::cout << "POOLOP: " << c_ih << " " << c_iw << " " << c_ich << " " << c_str << " " << c_pad << " " << std::endl;
#endif

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
	int c_pad,
	int c_pool
> void PoolKernel(
	hls::stream<t_input> &i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data
) {

#pragma HLS inline

	const int c_index = c_fh*c_fw;

	if (c_fh != c_ih) {

		hls::stream<t_acc> s_acc("s_acc");
		#pragma HLS STREAM variable=s_acc depth=2

		hls::stream<t_input> s_data[c_index-1];
		// TODO: Generate a script for different array FIFO sizing, c_ich*c_iw should
		// be the worst case
		#pragma HLS STREAM variable=s_data depth=c_ich*c_iw

		hls::stream<t_input> s_compute[c_fh*c_fw];
		#pragma HLS STREAM variable=s_compute depth=10

		hls::stream<ap_uint<1>> s_last[c_fh*c_fw+1];
		#pragma HLS STREAM variable=s_last depth=10

		SplitStream<
			c_fh*c_fw + 1
		> (
			i_last,
			s_last
		);

	/* Generic case implements the line buffer */
		ShiftOp<
			t_input,
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
			(c_fh-1),
			(c_fw-1)
		> (
			i_data,
			s_last[0],
			s_compute[0],
			s_data[0]
		);

		for (uint8_t s_index = 1; s_index < c_index-1; s_index++) {
			#pragma HLS unroll
			ShiftOp<
				t_input,
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
				s_data[s_index-1],
				s_last[s_index],
				s_compute[s_index],
				s_data[s_index],
				c_fh - s_index/c_fh - 1,
				c_fw - s_index%c_fw - 1
			);
		}

		ShiftOp<
			t_input,
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
			0,
			0
		> (
			s_data[c_index-2],
			s_last[c_index-1],
			s_compute[c_index-1]
		);

		PoolOp<
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
			c_pad,
			0,
			c_pool
		> (
			s_compute,
			s_last[c_index],
			o_last,
			o_data
		);

	} else {

		/* Global case does not need line buffer */
		PoolOp<
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
			c_pad,
			0,
			c_pool
		> (
			i_data,
			i_last,
			o_last,
			o_data
		);

	}

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
	int c_pad,
	int c_pool // 0 average, 1 max
> void PoolStreams(
	hls::stream<t_input> &i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data
) {

#pragma HLS inline

#ifndef __SYNTHESIS__

	while(i_data.empty());

#endif

	PoolKernel <
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
		c_pad,
		c_pool
	> (
		i_data,
		i_last,
		o_last,
		o_data
	);

}

#endif

