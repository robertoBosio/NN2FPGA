#ifndef __PACKEDCONV__
#define __PACKEDCONV__

#include "ap_int.h"
#include "hls_stream.h"
#include "Utils.hpp"

template <
	class t_input,
	class t_weight,
	class t_acc,
	int c_ich,
	int c_och,
	int c_ih,
	int c_iw,
	int c_fh,
	int c_fw,
	int c_str,
	int c_pad
> void ConvKernel(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> &i_weights,
	hls::stream<t_input> &o_data,
	hls::stream<t_acc> &o_acc,
	const int c_posh,
	const int c_posw
) {

	const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
	const int c_ih_end = c_ih + c_pad_index_h - (c_fh - 1) + c_posh;
	const int c_ih_start = 1 - c_fh + c_posh;
	const int c_iw_end = c_iw + c_pad_index_w - (c_fw - 1) + c_posw;
	const int c_iw_start = 1 - c_fw + c_posw;
	

	for (int8_t s_ih = c_ih_start; s_ih < c_ih_end; s_ih++) { 
#pragma HLS PIPELINE off
		bool s_str_condh = (s_ih % c_str) == (c_posh % c_str);
		for (int8_t s_iw = c_iw_start; s_iw < c_iw_end; s_iw++) { 
#pragma HLS PIPELINE off
			bool s_str_condw = (s_iw % c_str) == (c_posw % c_str);
			bool s_str_cond = s_str_condh & s_str_condw;
			t_acc s_acc[c_och] = {0};

			for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
#pragma HLS PIPELINE off

				t_input s_input = i_data.read();

				for (uint8_t s_och = 0; s_och < c_och; s_och++) {

#pragma HLS PIPELINE off

					s_acc[s_och] += s_input * i_weights.read();

				}

				if ((s_ih > -1) & (s_iw > -1) & (s_ih < c_ih) & (s_iw < c_iw) & s_str_cond)
					o_data.write(s_input);
				else 
					o_data.write(0);

			}

			for (uint8_t s_och = 0; s_och < c_och; s_och++)
				o_acc.write(s_acc[s_och]);

		}
	}

}

template <
	class t_input,
	class t_weight,
	class t_acc,
	int c_ich,
	int c_och,
	int c_ih,
	int c_iw,
	int c_fh,
	int c_fw,
	int c_str,
	int c_pad
> void ConvKernel(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_input> &o_data,
	hls::stream<t_acc> o_acc[c_fh*c_fw]
) {

#pragma HLS inline

	const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
	const int c_ih_end = c_ih + c_pad_index_h - (c_fh - 1);
	const int c_ih_start = 1 - c_fh;
	const int c_iw_end = c_iw + c_pad_index_w - (c_fw - 1);
	const int c_iw_start = 1 - c_fw;
	
	t_input s_buffer[c_fh*c_fw] = {0};

	for (int8_t s_ih = c_ih_start; s_ih < c_ih_end; s_ih++) { 
		for (int8_t s_iw = c_iw_start; s_iw < c_iw_end; s_iw++) { 
			t_acc s_acc[c_fh*c_fw][c_och] = {0};

			for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {

				t_input s_input = i_data.read();

				for (uint8_t s_index = c_fh*c_fw-1; s_index > 0; s_index--) {
					s_buffer[s_index] = s_buffer[s_index - 1];
				}

				s_buffer[0] = s_input;

				for (uint8_t s_och = 0; s_och < c_och; s_och++) {

					for (uint8_t s_index = 0; s_index < c_fh*c_fw; s_index++) {
#pragma HLS PIPELINE
						s_acc[s_index][s_och] += s_buffer[s_index] * i_weights[s_index].read();

					}
				}

				o_data.write(s_buffer[0]);

			}

			for (uint8_t s_och = 0; s_och < c_och; s_och++) {
#pragma HLS PIPELINE
				for (uint8_t s_fh = 0; s_fh < c_fh; s_fh++) {
					bool s_str_condh = ((s_ih % c_str) == (s_fh % c_str)) & (s_ih > -1) & (s_ih < c_ih);
					for (uint8_t s_fw = 0; s_fw < c_fw; s_fw++) {
						bool s_str_condw = ((s_iw % c_str) == (s_fw % c_str)) & (s_iw > -1) & (s_iw < c_iw);;
						if (s_str_condh & s_str_condw)
							o_acc[s_fh*c_fw+s_fw].write(s_acc[s_fh*c_fw+s_fw][s_och]);
					}
				}
			}

		}
	}

}

template <
	class t_input,
	class t_weight,
	class t_acc,
	int c_ich,
	int c_och,
	int c_ih,
	int c_iw,
	int c_fh,
	int c_fw,
	int c_pad
> void CascadedConvKernel(
	hls::stream<t_input> i_data[c_fh*c_fw+1],
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_acc> o_acc[c_fh*c_fw+1]
) {

	const int c_conv_index = c_fh*c_fw;

	for (uint8_t s_conv_index = 0; s_conv_index < c_conv_index; s_conv_index++) {
		#pragma HLS UNROLL
		ConvKernel<
			t_input,
			t_weight,
			t_acc,
			c_ich,
			c_och,
			c_ih,
			c_iw,
			c_fh,
			c_fw,
			c_pad
		> (
			i_data[s_conv_index],
			i_weights[s_conv_index],
			i_data[s_conv_index+1],
			o_acc[s_conv_index]
		);
	}

}

template <
	class t_input,
	int c_ich,
	int c_iw,
	int c_ih,
	int c_fw,
	int c_fh,
	int c_pad
> void PadInput(
	hls::stream<t_input> &i_data,
	hls::stream<t_input> &o_data
) {

	/* This handles padding aware inputs */

	const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_pad * (c_fw - 1) / 2;

#pragma HLS PIPELINE off

	for (uint8_t s_ih = 0; s_ih < c_ih; s_ih++) {

		/* Top padding */
		for (uint8_t s_pad = 0; s_pad < c_pad_index_h; s_pad++){
			for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					o_data.write(0);
				}
			}
		}

		for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {

			/* Right padding */
			for (uint8_t s_pad = 0; s_pad < c_pad_index_w; s_pad++){
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					o_data.write(0);
				}
			}

			for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
				o_data.write(i_data.read());
			}

			/* Left padding */
			for (uint8_t s_pad = 0; s_pad < c_pad_index_w; s_pad++){
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					o_data.write(0);
				}
			}

		}

		/* Bottom padding */
		for (uint8_t s_pad = 0; s_pad < c_pad_index_h; s_pad++){
			for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					o_data.write(0);
				}
			}
		}
	}

}

template <
	class t_acc,
	class t_output,
	int c_ich,
	int c_och,
	int c_iw,
	int c_ih,
	int c_ow,
	int c_oh,
	int c_fw,
	int c_fh,
	int c_pad
> void AccumulateKernel(
	hls::stream<t_acc> i_data[c_fh*c_fw],
	hls::stream<t_output> &o_data
) {

#pragma HLS inline
	const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
	const int c_ih_end = c_ih + c_pad_index_h - (c_fh - 1);
	const int c_ih_start = 1 - c_fh;
	const int c_iw_end = c_iw + c_pad_index_w - (c_fw - 1);
	const int c_iw_start = 1 - c_fw;
	

	t_acc s_acc_buffer[c_fh*2][c_ow][c_och] = {0};

	for (int8_t s_ih = c_ih_start; s_ih < c_ih_end; s_ih++) { 
#pragma HLS PIPELINE off
		for (int8_t s_iw = c_iw_start; s_iw < c_iw_end; s_iw++) { 
#pragma HLS PIPELINE off
			for (uint8_t s_fh = 0; s_fh < c_fh; s_fh++) {
#pragma HLS PIPELINE off
				int8_t s_oh = (s_ih + s_fh) % (2 * c_fh); 
				for (uint8_t s_fw = 0; s_fw < c_fw; s_fw++) {
#pragma HLS PIPELINE off
					int8_t s_ow = s_iw + s_fw; 
					for (uint8_t s_och = 0; s_och < c_och; s_och++) {
#pragma HLS PIPELINE off
						t_acc s_acc = i_data[s_fh*c_fw + s_fw].read();
#pragma HLS bind_op variable=s_acc_buffer op=add impl=dsp
						s_acc_buffer[s_oh][s_ow][s_och] += s_acc;	
					}
				}
			}

			for (uint8_t s_och = 0; s_och < c_och; s_och++) {
#pragma HLS PIPELINE off
				if ((s_iw > -1) & (s_ih > -1)) {
					int8_t s_oh = s_ih % (2 * c_fh); 
					int8_t s_ow = s_iw; 
					o_data.write(s_acc_buffer[s_ow][s_oh][s_och]);
					s_acc_buffer[s_ow][s_oh][s_och] = 0;
				}
			}
		}
	}

}

template <
	class t_acc,
	class t_output,
	int c_ich,
	int c_och,
	int c_iw,
	int c_ih,
	int c_ow,
	int c_oh,
	int c_fw,
	int c_fh,
	int c_pad
> void AccumulateKernel(
	hls::stream<t_acc> i_data[c_fh*c_fw],
	hls::stream<t_input> &i_bias,
	hls::stream<t_output> &o_data
) {

	const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
	const int c_ih_end = c_ih + c_pad_index_h - (c_fh - 1);
	const int c_ih_start = 1 - c_fh;
	const int c_iw_end = c_iw + c_pad_index_w - (c_fw - 1);
	const int c_iw_start = 1 - c_fw;
	

	t_acc s_acc_buffer[c_fh*2][c_ow][c_och] = {0};

	for (int8_t s_ih = c_ih_start; s_ih < c_ih_end; s_ih++) { 
		for (int8_t s_iw = c_iw_start; s_iw < c_iw_end; s_iw++) { 
			int8_t s_oh = s_ih % (2 * c_fh); 
			int8_t s_ow = s_iw; 
			for (uint8_t s_och = 0; s_och < c_och; s_och++) {
				t_input s_bias = i_bias.read();
				if ((s_iw > -1) & (s_ih > -1)) {
					s_acc_buffer[s_ow][s_oh][s_och] = s_bias;
				}
			}
			for (uint8_t s_fh = 0; s_fh < c_fh; s_fh++) {
				for (uint8_t s_fw = 0; s_fw < c_fw; s_fw++) {
					int8_t s_oh = (s_ih + c_fh - s_fh - 1) % (2 * c_fh); 
					int8_t s_ow = s_iw + c_fw - s_fw - 1; 
					for (uint8_t s_och = 0; s_och < c_och; s_och++) {
						t_acc s_acc = i_data[s_fh*c_fw + s_fw].read();
						if ((s_oh > -1) & (s_oh > -1)) {
							s_acc_buffer[s_oh][s_ow][s_och] += s_acc;	
						}
					}
				}
			}

			for (uint8_t s_och = 0; s_och < c_och; s_och++) {
				if ((s_iw > -1) & (s_ih > -1)) {
					o_data.write(s_acc_buffer[s_ow][s_oh][s_och]);
				}
			}
		}
	}

}


template <
	class t_input,
	int c_ich,
	int c_iw,
	int c_ih,
	int c_fw,
	int c_fh,
	int c_pad
> void ForwardStream(
	hls::stream<t_input> &i_data
) {

	const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
	const int c_ih_end = c_ih + c_pad_index_h;
	const int c_ih_start = -1 * c_pad_index_h;
	const int c_iw_end = c_iw + c_pad_index_w;
	const int c_iw_start = -1 * c_pad_index_w;
	
	for (int8_t s_ih = c_ih_start; s_ih < c_ih_end; s_ih++) { 
		for (int8_t s_iw = c_iw_start; s_iw < c_iw_end; s_iw++) { 
			for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) { 
				t_input s_tmp = i_data.read();
			}
		}
	}

}

template <
	class t_input,
	int c_ich,
	int c_iw,
	int c_ih,
	int c_fw,
	int c_fh,
	int c_pad
> void ForwardStream(
	hls::stream<t_input> &i_data,
	hls::stream<t_input> &o_forward
) {

	const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
	const int c_ih_end = c_ih + c_pad_index_h;
	const int c_ih_start = -1 * c_pad_index_h;
	const int c_iw_end = c_iw + c_pad_index_w;
	const int c_iw_start = -1 * c_pad_index_w;
	
	for (int8_t s_ih = c_ih_start; s_ih < c_ih_end; s_ih++) { 
		for (int8_t s_iw = c_iw_start; s_iw < c_iw_end; s_iw++) { 
			for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) { 
				t_input s_tmp = i_data.read();
				if ((s_ih > -1) & (s_iw > -1))
					o_forward.write(s_tmp);
			}
		}
	}

}

/* Config padded version of the packed convolution with Accumulation buffers */
/* What changes is the association between the filters kernel indices and the */ 
/* input features map, this translates in a different initialization of the loops */
template <
	class t_input,
	class t_weight,
	class t_output,
	class t_bias,
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
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_input> &i_bias,
	hls::stream<t_output> &o_data
) {


	const int c_conv_index = c_fh*c_fw;
	hls::stream<t_input, 1> s_data_stream[2];
	#pragma HLS STREAM variable=s_data_stream
	hls::stream<t_input, 1> s_bias_stream;
	#pragma HLS STREAM variable=s_bias_stream
	hls::stream<t_acc, c_conv_index> s_acc_stream[c_conv_index];
	#pragma HLS STREAM variable=s_acc_stream

	#pragma HLS DATAFLOW
	/* #pragma HLS inline */

	/* for (uint8_t s_conv_index = 0; s_conv_index < c_conv_index; s_conv_index++) { */
	/* 	if (i_weights[s_conv_index].empty()) */
	/* 		return; */
	/* } */

	/* if (i_data.empty()) */
	/* 	return; */

	/* This handles padding aware inputs */
	/* Higher indices weights can be applied only to central activations */ 
	/* 1 2 3      9 8 7 */
	/* 4 5 6 ---> 6 5 4 */
	/* 7 8 9      3 2 1 */
	/* 
	 * D11   D12     D13   D14
	 * 5 -> 4 6 -> 5 -> 4 6
	 * 2 8 -> 1 3 7 9 -> 2 8
	*/

	PadInput<
		t_input,
		c_ich,
		c_ih,
		c_iw,
		c_fh,
		c_fw,
		c_pad
	> (
		i_data,
		s_data_stream[0]
	);

	ConvKernel<
		t_input,
		t_weight,
		t_acc,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_fh,
		c_fw,
		c_str,
		c_pad
	> (
		s_data_stream[0],
		i_weights,
		s_data_stream[1],
		s_acc_stream
	);

	ForwardStream<
		t_input,
		c_ich,
		c_iw,
		c_ih,
		c_fw,
		c_fh,
		c_pad
	> (
		s_data_stream[1]
	);

	PadInput<
		t_input,
		c_ich,
		c_ih,
		c_iw,
		c_fh,
		c_fw,
		c_pad
	> (
		s_bias_stream
	);

	AccumulateKernel<
		t_acc,
		t_output,
		c_ich,
		c_och,
		c_iw,
		c_ih,
		c_ow,
		c_oh,
		c_fw,
		c_fh,
		c_pad
	> (
		s_acc_stream,
		s_bias_stream,
		o_data
	);

}

template <
	class t_input,
	class t_weight,
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
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_output> &o_data,
	hls::stream<t_input> &o_forward
) {


	const int c_conv_index = c_fh*c_fw;
	hls::stream<t_input, 1> s_data_stream[2];
	#pragma HLS STREAM variable=s_data_stream
	hls::stream<t_acc, c_conv_index> s_acc_stream[c_conv_index];
	#pragma HLS STREAM variable=s_acc_stream

	#pragma HLS DATAFLOW
/* #pragma HLS inline */

	/* for (uint8_t s_conv_index = 0; s_conv_index < c_conv_index; s_conv_index++) { */
	/* 	if (i_weights[s_conv_index].empty()) */
	/* 		return; */
	/* } */

	/* if (i_data.empty()) */
	/* 	return; */

	/* This handles padding aware inputs */
	/* Higher indices weights can be applied only to central activations */ 
	/* 1 2 3      9 8 7 */
	/* 4 5 6 ---> 6 5 4 */
	/* 7 8 9      3 2 1 */
	/* 
	 * D11   D12     D13   D14
	 * 5 -> 4 6 -> 5 -> 4 6
	 * 2 8 -> 1 3 7 9 -> 2 8
	*/

	PadInput<
		t_input,
		c_ich,
		c_ih,
		c_iw,
		c_fh,
		c_fw,
		c_pad
	> (
		i_data,
		s_data_stream[0]
	);

	ConvKernel<
		t_input,
		t_weight,
		t_acc,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_fh,
		c_fw,
		c_str,
		c_pad
	> (
		s_data_stream[0],
		i_weights,
		s_data_stream[1],
		s_acc_stream
	);

	ForwardStream<
		t_input,
		c_ich,
		c_iw,
		c_ih,
		c_fw,
		c_fh,
		c_pad
	> (
		s_data_stream[1],
		o_forward
	);

	AccumulateKernel<
		t_acc,
		t_output,
		c_ich,
		c_och,
		c_iw,
		c_ih,
		c_ow,
		c_oh,
		c_fw,
		c_fh,
		c_pad
	> (
		s_acc_stream,
		o_data
	);

}

/* Config padded version of the packed convolution with Accumulation buffers */
/* What changes is the association between the filters kernel indices and the */ 
/* input features map, this translates in a different initialization of the loops */
template <
	class t_input,
	class t_weight,
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
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_output> &o_data
) {


	const int c_conv_index = c_fh*c_fw;

	/* for (uint8_t s_conv_index = 0; s_conv_index < c_conv_index; s_conv_index++) { */
	/* 	if (i_weights[s_conv_index].empty()) */
	/* 		return; */
	/* } */

	/* if (i_data.empty()) */
	/* 	return; */

	#pragma HLS DATAFLOW
/* #pragma HLS inline */

	/* This handles padding aware inputs */
	/* Higher indices weights can be applied only to central activations */ 
	/* 1 2 3      9 8 7 */
	/* 4 5 6 ---> 6 5 4 */
	/* 7 8 9      3 2 1 */
	/* 
	 * D11   D12     D13   D14
	 * 5 -> 4 6 -> 5 -> 4 6
	 * 2 8 -> 1 3 7 9 -> 2 8
	*/

	hls::stream<t_input> s_data_stream[2];
	#pragma HLS STREAM variable=s_data_stream type=fifo
	hls::stream<t_acc, c_conv_index> s_acc_stream[c_conv_index];
	#pragma HLS STREAM variable=s_acc_stream

	PadInput<
		t_input,
		c_ich,
		c_ih,
		c_iw,
		c_fh,
		c_fw,
		c_pad
	> (
		i_data,
		s_data_stream[0]
	);

	ConvKernel<
		t_input,
		t_weight,
		t_acc,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_fh,
		c_fw,
		c_str,
		c_pad
	> (
		s_data_stream[0],
		i_weights,
		s_data_stream[1],
		s_acc_stream
	);

	ForwardStream<
		t_input,
		c_ich,
		c_iw,
		c_ih,
		c_fw,
		c_fh,
		c_pad
	> (
		s_data_stream[1]
	);

	AccumulateKernel<
		t_acc,
		t_output,
		c_ich,
		c_och,
		c_iw,
		c_ih,
		c_ow,
		c_oh,
		c_fw,
		c_fh,
		c_pad
	> (
		s_acc_stream,
		o_data
	);

}

#endif
