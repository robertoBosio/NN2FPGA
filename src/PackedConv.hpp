#ifndef __PACKEDCONV__
#define __PACKEDCONV__

#include "ap_int.h"
#include "hls_stream.h"
#include "Utils.hpp"

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
	hls::stream<t_weight> &i_weights,
	hls::stream<t_output> &o_data,
	hls::stream<t_input> &o_forward
) {

	if (i_weights.empty())
		return;

	if (i_data.empty())
		return;

	const int c_ih_loop_start = (1-c_pad)*(c_fh);
	const int c_ih_loop_end = c_ih_loop_start + c_ih;

	const int c_iw_loop_start = (1-c_pad)*(c_fw);
	const int c_iw_loop_end = c_iw_loop_start + c_iw;

	/* Buffer dimension changes depending on the selected padding */
	const int c_iw_padded = c_iw + 2*c_pad*(c_iw % c_fw);
	t_input s_data_buffer[c_ich][c_fh][c_iw_padded] = {0};

	if (c_pad == 0) {
		/* Filling the buffers for padding case */
		for (uint8_t s_fh = 0; s_fh < (c_fh - 1); s_fh++) {
			for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					t_input s_input = i_data.read();
					s_data_buffer[s_ich][s_fh][s_iw] = s_input;
					o_forward.write(s_input);
				}
			}
		}
	}

	/* Loop indices controls where to save input data */
	for (uint8_t s_ih = c_ih_loop_start; s_ih < c_ih_loop_end; s_ih+=c_str) {

		if (c_pad == 0) {
			/* Filling the buffers for padding case along w axes */
			for (uint8_t s_iw = 0; s_iw < (c_fw - 1); s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					t_input s_input = i_data.read();
					s_data_buffer[s_ich][s_ih % c_fh][s_iw] = s_input;
					o_forward.write(s_input);
				}
			}
		}

		for (uint8_t s_iw = c_iw_loop_start; s_iw < c_iw_loop_end; s_iw+=c_str) {

			/* Filling the buffer in case of stride along w dimension */
			for (uint8_t s_str = 0; s_str < c_str; s_str++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					/* In case of padding, the h and w value starts from c_f*-1 */
					t_input s_input = i_data.read();
					s_data_buffer[s_ich][s_ih % c_fh][s_iw + s_str] = s_input;
					o_forward.write(s_input);
				}
			}

			t_acc s_acc = 0;

			for (uint8_t s_och = 0; s_och < c_och; s_och++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					for (uint8_t s_fh = 0; s_fh < c_fh; s_fh++) {
						uint8_t s_ih_read = s_ih + s_fh - c_ih_loop_start;
						for (uint8_t s_fw =0; s_fw < c_fw; s_fw++) {
							uint8_t s_iw_read = s_iw + s_fw - c_iw_loop_start;
							t_input s_data = s_data_buffer[s_ich][s_ih_read % c_fh][s_iw_read];
							t_weight s_weight = i_weights.read();
							s_acc += s_data * s_weight;
						}
					}
				}
				o_data.write((t_output)(s_acc));
			}
		}

		/* TODO: report all changes to other kernels */
		/* This loop fills the new line of the buffer if stride is present */
		for (uint8_t s_str = 1; s_str < (c_str); s_str++) {
			/* Filling the buffers for padding case along w axes */
			for (uint8_t s_iw = c_iw_loop_start; s_iw < c_iw_loop_end; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					t_input s_input = i_data.read();
					s_data_buffer[s_ich][(s_ih + s_str) % c_fh][s_iw] = s_input;
					o_forward.write(s_input);
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
> void ConvKernel(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> &i_weights,
	hls::stream<t_input> &o_data,
	hls::stream<t_acc> &o_acc
) {

	const int c_pad_index_h = c_ih + c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_iw + c_pad * (c_fw - 1) / 2;
	const int c_acc_index = c_och;
	const int c_index = c_pad_index_h*c_pad_index_w;
	
	for (uint8_t s_index = 0; s_index < c_index; s_index++) {

		t_input s_input = i_data.read();

		t_acc s_acc[c_och] = {0};

		for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {

			for (uint8_t s_och = 0; s_och < c_och; s_och++) {

				s_acc[s_och] += i_data.read() * i_weights.read();

			}

		}

		for (uint8_t s_och = 0; s_och < c_och; s_och++)
			o_acc.write(s_acc[s_och]);


		o_data.write(s_input);

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

	const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
	const int c_ih_end = c_ih + c_pad_index_h - (c_fh - 1);
	const int c_ih_start = 1 - c_fh;
	const int c_iw_end = c_iw + c_pad_index_w - (c_fw - 1);
	const int c_iw_start = 1 - c_fw;
	

	t_acc s_acc_buffer[c_fh*2][c_ow][c_och] = {0};

	for (int8_t s_ih = c_ih_start; s_ih < c_ih_end; s_ih++) { 
		for (int8_t s_iw = c_iw_start; s_iw < c_iw_end; s_iw++) { 
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
	class t_acc,
	int c_ich,
	int c_och,
	int c_iw,
	int c_ih,
	int c_ow,
	int c_oh,
	int c_fw,
	int c_fh,
	int c_pad
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_output> &o_data,
	hls::stream<t_input> &o_forward
) {


	const int c_conv_index = c_fh*c_fw;
	hls::stream<t_input, 1> s_data_stream[c_conv_index + 1];
	#pragma HLS STREAM variable=s_data_stream
	hls::stream<t_acc, 1> s_acc_stream[c_conv_index + 1];
	#pragma HLS STREAM variable=s_acc_stream

	#pragma HLS DATAFLOW

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

	CascadedConvKernel<
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
		s_data_stream,
		i_weights,
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
		s_data_stream[c_fh*c_fw],
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

	hls::stream<t_input, 1> s_data_stream[c_fh*c_fw + 1];
	#pragma HLS STREAM variable=s_data_stream type=fifo
	hls::stream<t_acc, 1> s_acc_stream[c_fh*c_fw + 1];
	#pragma HLS STREAM variable=s_acc_stream type=fifo

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

	CascadedConvKernel<
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
		s_data_stream,
		i_weights,
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
		s_data_stream[c_fh*c_fw]
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
/* Version with BIAS allows to consume input stream */
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
	hls::stream<t_weight> &i_weights,
	hls::stream<t_output> &o_data
) {

	if (i_weights.empty())
		return;

	if (i_data.empty())
		return;

	t_acc s_acc_buffer[c_fh * 2][c_ow][c_och] = {0};

	const int c_ih_padded = c_ih + 2*(1 - c_pad)*(c_ih % c_fh);
	const int c_ih_start  = (1 - c_pad)*(c_ih % c_fh);
	const int c_ih_loop_start = (c_pad)*(c_ih % c_fh) + c_ih_start;
	const int c_ih_loop_end = (c_pad)*(c_ih % c_fh) + c_ih;

	const int c_iw_padded = c_iw + 2*c_pad*(c_iw % c_fw);
	const int c_iw_start  = (1 - c_pad)*(c_iw % c_fw);
	const int c_iw_loop_start = (c_pad)*(c_iw % c_fw) + c_iw_start;
	const int c_iw_loop_end = (c_pad)*(c_iw % c_fw) + c_iw;

	t_input s_data_buffer[c_ich][c_fh][c_iw_padded] = {0};

	if (c_pad == 0) {
		/* Filling the buffers for padding case */
		for (uint8_t s_fh = 0; s_fh < (c_fh - 1); s_fh++) {
			for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					t_input s_input = i_data.read();
					s_data_buffer[s_ich][s_fh][s_iw] = s_input;
				}
			}
		}
	}

	/* Loop indices controls where to save input data */
	for (uint8_t s_ih = c_ih_loop_start; s_ih < c_ih_loop_end; s_ih+=c_str) {

		if (c_pad == 0) {
			/* Filling the buffers for padding case along w axes */
			for (uint8_t s_iw = 0; s_iw < (c_fw - 1); s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					t_input s_input = i_data.read();
					s_data_buffer[s_ich][s_ih % c_fh][s_iw] = s_input;
				}
			}
		}

		for (uint8_t s_iw = c_iw_loop_start; s_iw < c_iw_loop_end; s_iw+=c_str) {

			/* Filling the buffer in case of stride along w dimension */
			for (uint8_t s_str = 0; s_str < c_str; s_str++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					/* In case of padding, the h and w value starts from c_f*-1 */
					t_input s_input = i_data.read();
					s_data_buffer[s_ich][s_ih % c_fh][s_iw + s_str] = s_input;
				}
			}

			t_acc s_acc = 0;

			for (uint8_t s_och = 0; s_och < c_och; s_och++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					for (uint8_t s_fh = 0; s_fh < c_fh; s_fh++) {
						uint8_t s_ih_read = s_ih + s_fh - c_ih_loop_start;
						for (uint8_t s_fw =0; s_fw < c_fw; s_fw++) {
							uint8_t s_iw_read = s_iw + s_fw - c_iw_loop_start;
							t_input s_data = s_data_buffer[s_ich][s_ih_read % c_fh][s_iw_read];
							t_weight s_weight = i_weights.read();
							s_acc += s_data * s_weight;
						}
					}
				}
				o_data.write((t_output)(s_acc));
			}
		}

		/* TODO: report all changes to other kernels */
		/* This loop fills the new line of the buffer if stride is present */
		for (uint8_t s_str = 1; s_str < (c_str); s_str++) {
			/* Filling the buffers for padding case along w axes */
			for (uint8_t s_iw = c_iw_loop_start; s_iw < c_iw_loop_end; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					t_input s_input = i_data.read();
					s_data_buffer[s_ich][(s_ih + s_str) % c_fh][s_iw] = s_input;
				}
			}
		}

	}

}

/* Config padded version of the packed convolution with Accumulation buffers */
/* What changes is the association between the filters kernel indices and the */ 
/* input features map, this translates in a different initialization of the loops */
/* Version with BIAS allows to consume input stream */
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
	hls::stream<t_weight> &i_weights,
	hls::stream<t_bias> &i_bias,
	hls::stream<t_output> &o_data
) {

	if (i_weights.empty())
		return;

	if (i_data.empty())
		return;

	t_acc s_acc_buffer[c_fh * 2][c_ow][c_och] = {0};

	const int c_ih_padded = c_ih + 2*(1 - c_pad)*(c_ih % c_fh);
	const int c_ih_start  = (1 - c_pad)*(c_ih % c_fh);
	const int c_ih_loop_start = (c_pad)*(c_ih % c_fh) + c_ih_start;
	const int c_ih_loop_end = (c_pad)*(c_ih % c_fh) + c_ih;

	const int c_iw_padded = c_iw + 2*c_pad*(c_iw % c_fw);
	const int c_iw_start  = (1 - c_pad)*(c_iw % c_fw);
	const int c_iw_loop_start = (c_pad)*(c_iw % c_fw) + c_iw_start;
	const int c_iw_loop_end = (c_pad)*(c_iw % c_fw) + c_iw;

	t_input s_data_buffer[c_ich][c_fh][c_iw_padded] = {0};

	if (c_pad == 0) {
		/* Filling the buffers for padding case */
		for (uint8_t s_fh = 0; s_fh < (c_fh - 1); s_fh++) {
			for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					t_input s_input = i_data.read();
					s_data_buffer[s_ich][s_fh][s_iw] = s_input;
				}
			}
		}
	}

	/* Loop indices controls where to save input data */
	for (uint8_t s_ih = c_ih_loop_start; s_ih < c_ih_loop_end; s_ih+=c_str) {

		if (c_pad == 0) {
			/* Filling the buffers for padding case along w axes */
			for (uint8_t s_iw = 0; s_iw < (c_fw - 1); s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					t_input s_input = i_data.read();
					s_data_buffer[s_ich][s_ih % c_fh][s_iw] = s_input;
				}
			}
		}

		for (uint8_t s_iw = c_iw_loop_start; s_iw < c_iw_loop_end; s_iw+=c_str) {

			/* Filling the buffer in case of stride along w dimension */
			for (uint8_t s_str = 0; s_str < c_str; s_str++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					/* In case of padding, the h and w value starts from c_f*-1 */
					t_input s_input = i_data.read();
					s_data_buffer[s_ich][s_ih % c_fh][s_iw + s_str] = s_input;
				}
			}

			t_acc s_acc = i_bias.read();

			for (uint8_t s_och = 0; s_och < c_och; s_och++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					for (uint8_t s_fh = 0; s_fh < c_fh; s_fh++) {
						uint8_t s_ih_read = s_ih + s_fh - c_ih_loop_start;
						for (uint8_t s_fw =0; s_fw < c_fw; s_fw++) {
							uint8_t s_iw_read = s_iw + s_fw - c_iw_loop_start;
							t_input s_data = s_data_buffer[s_ich][s_ih_read % c_fh][s_iw_read];
							t_weight s_weight = i_weights.read();
							s_acc += s_data * s_weight;
						}
					}
				}
				o_data.write((t_output)(s_acc));
			}
		}

		/* TODO: report all changes to other kernels */
		/* This loop fills the new line of the buffer if stride is present */
		for (uint8_t s_str = 1; s_str < (c_str); s_str++) {
			/* Filling the buffers for padding case along w axes */
			for (uint8_t s_iw = c_iw_loop_start; s_iw < c_iw_loop_end; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					t_input s_input = i_data.read();
					s_data_buffer[s_ich][(s_ih + s_str) % c_fh][s_iw] = s_input;
				}
			}
		}

	}

}

#endif
