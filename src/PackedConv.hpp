#ifndef __PACKEDCONV__
#define __PACKEDCONV__

#include "ap_int.h"
#include "hls_stream.h"
#include "Utils.hpp"
#include "ActivationStreams.hpp"

template <
	class t_input,
	class t_weight,
	class t_acc,
	int c_ich,
	int c_och,
	int c_ih,
	int c_iw,
	int c_oh,
	int c_ow,
	int c_fh,
	int c_fw,
	int c_relu,
	int c_str,
	int c_bypass
> void ConvOp(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> &i_weights,
	hls::stream<t_input> &i_bias,
	hls::stream<t_input> &o_data,
	hls::stream<t_acc> &o_acc
) {

#pragma HLS PIPELINE off
	const int c_bypass_w = c_fw - 1;

	for (uint8_t s_ih = 0; s_ih < c_bypass; s_ih+=c_str) {
		for (uint8_t s_strh = 0; s_strh < c_str; s_strh++) {
			for (uint8_t s_iw = 0; s_iw < c_iw; s_iw+=c_str) {
				for (uint8_t s_strw = 0; s_strw < c_str; s_strw++) {
					for (uint8_t s_och = 0; s_och < c_och; s_och++)
						t_input s_bias = i_bias.read();
					for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
						t_input s_input = i_data.read();
						o_data.write(s_input);
					}
				}
			}
		}
	}

	for (uint8_t s_ih = 0; s_ih < c_ih; s_ih+=c_str) {
		for (uint8_t s_strh = 0; s_strh < c_str; s_strh++) {

			/* Start shifting for padding */
			for (uint8_t s_iw = 0; s_iw < c_bypass_w; s_iw+=c_str) {
				for (uint8_t s_strw = 0; s_strw < c_str; s_strw++) {
					for (uint8_t s_och = 0; s_och < c_och; s_och++) {
						t_input s_bias = i_bias.read();
					}
					for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
						t_input s_input = i_data.read();
						o_data.write(s_input);
					}
				}
			}

			for (uint8_t s_iw = 0; s_iw < c_iw; s_iw+=c_str) {
				for (uint8_t s_strw = 0; s_strw < c_str; s_strw++) {

					uint8_t s_bypass_w = 0;

					t_acc s_acc_buff[c_och];

					for (uint8_t s_och = 0; s_och < c_och; s_och++)
						s_acc_buff[s_och] = i_bias.read();

					for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
						t_input s_input = i_data.read();
						for (uint8_t s_och = 0; s_och < c_och; s_och++) {
							t_weight s_weights = i_weights.read();
							s_acc_buff[s_och] += s_input * s_weights;
						}
						o_data.write(s_input);
					}

					for (uint8_t s_och = 0; s_och < c_och; s_och++) {
						if ((s_strh == 0) & (s_strw == 0)) {
							o_acc.write(s_acc_buff[s_och]); 
						}
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
	int c_oh,
	int c_ow,
	int c_fh,
	int c_fw,
	int c_relu,
	int c_str,
	int c_bypass
> void ConvOp(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> &i_weights,
	hls::stream<t_input> &o_data,
	hls::stream<t_acc> &o_acc
) {

#pragma HLS PIPELINE off
	const int c_bypass_w = c_fw - 1;

	for (uint8_t s_ih = 0; s_ih < c_bypass; s_ih+=c_str) {
		for (uint8_t s_strh = 0; s_strh < c_str; s_strh++) {
			for (uint8_t s_iw = 0; s_iw < c_iw; s_iw+=c_str) {
				for (uint8_t s_strw = 0; s_strw < c_str; s_strw++) {
					for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
						t_input s_input = i_data.read();
						o_data.write(s_input);
					}
				}
			}
		}
	}


	for (uint8_t s_ih = c_bypass; s_ih < c_ih; s_ih+=c_str) {
		for (uint8_t s_strh = 0; s_strh < c_str; s_strh++) {

			/* Start shifting for padding */
			for (uint8_t s_iw = 0; s_iw < c_bypass_w; s_iw+=c_str) {
				for (uint8_t s_strw = 0; s_strw < c_str; s_strw++) {
					for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
						t_input s_input = i_data.read();
						o_data.write(s_input);
					}
				}
			}

			for (uint8_t s_iw = c_bypass_w; s_iw < c_iw; s_iw+=c_str) {
				for (uint8_t s_strw = 0; s_strw < c_str; s_strw++) {
					t_acc s_acc_buff[c_och] = {0};

					for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
						t_input s_input = i_data.read();
						for (uint8_t s_och = 0; s_och < c_och; s_och++) {
							t_weight s_weights = i_weights.read();
							s_acc_buff[s_och] += s_input * s_weights;
						}
						o_data.write(s_input);
					}

					for (uint8_t s_och = 0; s_och < c_och; s_och++) {
						if ((s_strh == 0) & (s_strw == 0)) {
							o_acc.write(s_acc_buff[s_och]); 
						}
					}

				}
			}
		}
	}

}

template <
	class t_acc,
	class t_output,
	int c_och,
	int c_oh,
	int c_ow,
	int c_relu
> void WriteOutput(
	hls::stream<t_acc> &i_data,
	hls::stream<t_output> &o_data
) {

	for (uint8_t s_oh = 0; s_oh < c_oh; s_oh++) {
		for (uint8_t s_ow = 0; s_ow < c_ow; s_ow++) {
			for (uint8_t s_och = 0; s_och < c_och; s_och++) {
#pragma HLS loop_flatten
#pragma HLS PIPELINE off
				t_acc s_o_acc = i_data.read();
				if (c_relu == 1)
					s_o_acc = ReluOp<t_acc>(s_o_acc);
				o_data.write((t_output)(s_o_acc));
			}
		}
	}
}

template <
	class t_input,
	class t_weight,
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
	int c_relu,
	int c_str,
	int c_pad
> void ConvKernel1x1(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_input> &o_forward,
	hls::stream<t_input> &i_bias,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;

#pragma HLS inline

	hls::stream<t_acc> s_acc;
	#pragma HLS STREAM variable=s_acc depth=2

	ConvOp<
		t_input,
		t_weight,
		t_acc,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_oh,
		c_ow,
		c_fh,
		c_fw,
		c_relu,
		c_str,
		0
	> (
		i_data,
		i_weights[0],
		i_bias,
		o_forward,
		s_acc
	);

	WriteOutput<
		t_acc,
		t_output,
		c_och,
		c_oh,
		c_ow,
		c_relu
	> (
		s_acc,
		o_data
	);

}

template <
	class t_acc,
	class t_output,
	int c_och,
	int c_oh,
	int c_ow,
	int c_fh,
	int c_fw,
	int c_relu
> void WriteOutput(
	hls::stream<t_acc> i_data[c_fh*c_fw],
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;

	for (uint8_t s_oh = 0; s_oh < c_oh; s_oh++) {
		for (uint8_t s_ow = 0; s_ow < c_ow; s_ow++) {
			for (uint8_t s_och = 0; s_och < c_och; s_och++) {
#pragma HLS loop_flatten
				t_acc s_o_acc = 0;
				for (uint8_t s_index = 0; s_index < c_index; s_index++) {
#pragma HLS PIPELINE off
					s_o_acc += i_data[s_index].read();
				}
				if (c_relu == 1)
					s_o_acc = ReluOp<t_acc>(s_o_acc);
				o_data.write((t_output)(s_o_acc));
			}
		}
	}

}

template <
	class t_input,
	class t_weight,
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
	int c_relu,
	int c_str,
	int c_pad
> void ConvKernel3x3(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_input> &i_bias,
	hls::stream<t_input> &o_forward,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;
	hls::stream<t_input> s_data[c_index-1];
	#pragma HLS STREAM variable=s_data[0] depth=c_ich
	#pragma HLS STREAM variable=s_data[1] depth=c_ich
	#pragma HLS STREAM variable=s_data[2] depth=c_ich*(c_iw-2)
	#pragma HLS STREAM variable=s_data[3] depth=c_ich
	#pragma HLS STREAM variable=s_data[4] depth=c_ich
	#pragma HLS STREAM variable=s_data[5] depth=c_ich*(c_iw-2)
	#pragma HLS STREAM variable=s_data[6] depth=c_ich
	#pragma HLS STREAM variable=s_data[7] depth=c_ich
	hls::stream<t_acc> s_acc[c_index];
	#pragma HLS STREAM variable=s_acc[0] depth=9
	#pragma HLS STREAM variable=s_acc[1] depth=9
	#pragma HLS STREAM variable=s_acc[2] depth=9
	#pragma HLS STREAM variable=s_acc[3] depth=9
	#pragma HLS STREAM variable=s_acc[4] depth=9
	#pragma HLS STREAM variable=s_acc[5] depth=9
	#pragma HLS STREAM variable=s_acc[6] depth=9
	#pragma HLS STREAM variable=s_acc[7] depth=9
	#pragma HLS STREAM variable=s_acc[8] depth=9

#pragma HLS inline

	ConvOp<
		t_input,
		t_weight,
		t_acc,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_oh,
		c_ow,
		c_fh,
		c_fw,
		c_relu,
		c_str,
		(c_fh-1)
	> (
		i_data,
		i_weights[0],
		i_bias,
		s_data[0],
		s_acc[0]
	);

	ConvOp<
		t_input,
		t_weight,
		t_acc,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_oh,
		c_ow,
		c_fh,
		c_fw,
		c_relu,
		c_str,
		(c_fh-1)
	> (
		s_data[0],
		i_weights[1],
		s_data[1],
		s_acc[1]
	);

	ConvOp<
		t_input,
		t_weight,
		t_acc,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_oh,
		c_ow,
		c_fh,
		c_fw,
		c_relu,
		c_str,
		(c_fh-1)
	> (
		s_data[1],
		i_weights[2],
		s_data[2],
		s_acc[2]
	);

	ConvOp<
		t_input,
		t_weight,
		t_acc,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_oh,
		c_ow,
		c_fh,
		c_fw,
		c_relu,
		c_str,
		(c_fh-2)
	> (
		s_data[2],
		i_weights[3],
		s_data[3],
		s_acc[3]
	);

	ConvOp<
		t_input,
		t_weight,
		t_acc,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_oh,
		c_ow,
		c_fh,
		c_fw,
		c_relu,
		c_str,
		(c_fh-2)
	> (
		s_data[3],
		i_weights[4],
		s_data[4],
		s_acc[4]
	);

	ConvOp<
		t_input,
		t_weight,
		t_acc,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_oh,
		c_ow,
		c_fh,
		c_fw,
		c_relu,
		c_str,
		(c_fh-2)
	> (
		s_data[4],
		i_weights[5],
		s_data[5],
		s_acc[5]
	);

	ConvOp<
		t_input,
		t_weight,
		t_acc,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_oh,
		c_ow,
		c_fh,
		c_fw,
		c_relu,
		c_str,
		(c_fh-3)
	> (
		s_data[5],
		i_weights[6],
		s_data[6],
		s_acc[6]
	);

	ConvOp<
		t_input,
		t_weight,
		t_acc,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_oh,
		c_ow,
		c_fh,
		c_fw,
		c_relu,
		c_str,
		(c_fh-3)
	> (
		s_data[6],
		i_weights[7],
		s_data[7],
		s_acc[7]
	);

	ConvOp<
		t_input,
		t_weight,
		t_acc,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_oh,
		c_ow,
		c_fh,
		c_fw,
		c_relu,
		c_str,
		0
	> (
		s_data[7],
		i_weights[8],
		o_forward,
		s_acc[8]
	);


	WriteOutput<
		t_acc,
		t_output,
		c_och,
		c_oh,
		c_ow,
		c_fh,
		c_fw,
		c_relu
	> (
		s_acc,
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
	int c_relu,
	int c_str,
	int c_pad
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_input> &i_bias,
	hls::stream<t_output> &o_data
) {

	const int c_conv_index = c_fh*c_fw;
	hls::stream<t_input, 2> s_data_stream[2];
	#pragma HLS STREAM variable=s_data_stream
	hls::stream<t_input, 2> s_bias_stream;
	#pragma HLS STREAM variable=s_bias_stream

#pragma HLS inline

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

	PadInput<
		t_input,
		c_ich,
		c_ih,
		c_iw,
		c_fh,
		c_fw,
		c_pad
	> (
		i_bias,
		s_bias_stream
	);

	ConvKernel3x3 <
		t_input,
		t_weight,
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
		c_relu,
		c_str,
		c_pad
	> (
		s_data_stream[0],
		i_weights,
		s_bias_stream,
		s_data_stream[1],
		o_data
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

}

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
	int c_relu,
	int c_split,
	int c_str,
	int c_pad
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_input> &i_bias,
	hls::stream<t_output> o_data[c_split]
) {


	const int c_conv_index = c_fh*c_fw;
	hls::stream<t_input, 2> s_data_stream[2];
	#pragma HLS STREAM variable=s_data_stream
	hls::stream<t_input, 2> s_bias_stream;
	#pragma HLS STREAM variable=s_bias_stream
	hls::stream<t_output, 2> s_out_stream;
	#pragma HLS STREAM variable=s_out_stream

#pragma HLS inline

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

	PadInput<
		t_input,
		c_ich,
		c_ih,
		c_iw,
		c_fh,
		c_fw,
		c_pad
	> (
		i_bias,
		s_bias_stream
	);

	ConvKernel3x3 <
		t_input,
		t_weight,
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
		c_relu,
		c_str,
		c_pad
	> (
		s_data_stream[0],
		i_weights,
		s_bias_stream,
		s_data_stream[1],
		s_out_stream
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

	SplitStream<
		t_output,
		c_och,
		c_ow,
		c_oh,
		c_split
	> (
		s_out_stream,
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
	int c_relu,
	int c_str,
	int c_pad
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[1],
	hls::stream<t_output> &o_data,
	hls::stream<t_input> &o_forward
) {


	hls::stream<t_input, 2> s_data_stream;
	#pragma HLS STREAM variable=s_data_stream
	hls::stream<t_input, 2> s_bias_stream;
	#pragma HLS STREAM variable=s_bias_stream

#pragma HLS inline

	PadInput<
		t_input,
		c_ich,
		c_ih,
		c_iw
	> (
		s_bias_stream
	);

	ConvKernel1x1 <
		t_input,
		t_weight,
		t_acc,
		t_output,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_oh,
		c_ow,
		1,
		1,
		c_relu,
		c_str,
		c_pad
	> (
		i_data,
		i_weights,
		s_bias_stream,
		s_data_stream,
		o_data
	);

	ForwardStream<
		t_input,
		c_ich,
		c_iw,
		c_ih,
		1,
		1,
		c_pad
	> (
		s_data_stream,
		o_forward
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
	int c_relu,
	int c_str,
	int c_pad
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_output> &o_data,
	hls::stream<t_input> &o_forward
) {


	const int c_conv_index = c_fh*c_fw;
	hls::stream<t_input, 2> s_data_stream[2];
	#pragma HLS STREAM variable=s_data_stream
	hls::stream<t_input, 2> s_bias_stream;
	#pragma HLS STREAM variable=s_bias_stream

#pragma HLS inline

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

	ConvKernel3x3 <
		t_input,
		t_weight,
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
		c_relu,
		c_str,
		c_pad
	> (
		s_data_stream[0],
		i_weights,
		s_bias_stream,
		s_data_stream[1],
		o_data
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
	int c_relu,
	int c_str,
	int c_pad
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[1],
	hls::stream<t_output> &o_data
) {

	hls::stream<t_input, 2> s_data_stream;
	#pragma HLS STREAM variable=s_data_stream
	hls::stream<t_input, 2> s_bias_stream;
	#pragma HLS STREAM variable=s_bias_stream

#pragma HLS inline

	PadInput<
		t_input,
		c_ich,
		c_ih,
		c_iw
	> (
		s_bias_stream
	);

	ConvKernel1x1 <
		t_input,
		t_weight,
		t_acc,
		t_output,
		c_ich,
		c_och,
		c_ih,
		c_iw,
		c_oh,
		c_ow,
		1,
		1,
		c_relu,
		c_str,
		c_pad
	> (
		i_data,
		i_weights,
		s_bias_stream,
		s_data_stream,
		o_data
	);

	ForwardStream<
		t_input,
		c_ich,
		c_iw,
		c_ih,
		1,
		1,
		c_pad
	> (
		s_data_stream
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
	int c_relu,
	int c_str,
	int c_pad
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_output> &o_data
) {

	const int c_conv_index = c_fh*c_fw;
	hls::stream<t_input, 2> s_data_stream[2];
	#pragma HLS STREAM variable=s_data_stream
	hls::stream<t_input, 2> s_bias_stream;
	#pragma HLS STREAM variable=s_bias_stream

#pragma HLS inline

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

	ConvKernel3x3 <
		t_input,
		t_weight,
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
		c_relu,
		c_str,
		c_pad
	> (
		s_data_stream[0],
		i_weights,
		s_bias_stream,
		s_data_stream[1],
		o_data
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
	int c_relu,
	int c_split,
	int c_str,
	int c_pad
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_output> o_data[c_split]
) {



	const int c_conv_index = c_fh*c_fw;
	hls::stream<t_input, 2> s_data_stream[2];
	#pragma HLS STREAM variable=s_data_stream
	hls::stream<t_input, 2> s_bias_stream;
	#pragma HLS STREAM variable=s_bias_stream
	hls::stream<t_output, 2> s_out_stream;
	#pragma HLS STREAM variable=s_out_stream

#pragma HLS inline

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

	ConvKernel3x3 <
		t_input,
		t_weight,
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
		c_relu,
		c_str,
		c_pad
	> (
		s_data_stream[0],
		i_weights,
		s_bias_stream,
		s_data_stream[1],
		s_out_stream	
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

	SplitStream<
		t_output,
		c_och,
		c_ow,
		c_oh,
		c_split
	> (
		s_out_stream,
		o_data
	);

}

#endif
