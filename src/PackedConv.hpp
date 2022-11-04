#ifndef __PACKEDCONV__
#define __PACKEDCONV__

#include "ap_int.h"
#include "hls_stream.h"
#include "Utils.hpp"
#include "ActivationStreams.hpp"
#include "Debug.hpp"

template <
	class t_input,
	class t_weight,
	class t_acc,
	int c_och,
	int c_index,
	int c_ops
> void ConvComp(
	t_input i_input[c_index],
	hls::stream<t_weight> i_weights[c_index],
	t_acc o_acc_buff[c_och]
) {
#pragma HLS inline

	/* Assuming that the number of computations is a multiplier of the number */
	/* of operations */
	const int c_num_comp = c_och;
	const int c_pipe_iter = c_num_comp/c_ops;

	uint8_t s_index = 0;
	uint16_t s_och = 0;

	for (uint16_t s_pipe_iter = 0; s_pipe_iter < c_pipe_iter; s_pipe_iter++) {
		for (uint8_t s_ops = 0; s_ops < c_ops*c_index; s_ops++) {

#pragma HLS pipeline
			o_acc_buff[s_och] += i_input[s_index] * i_weights[s_index].read();

			s_index++;

			if (s_index == c_index) {
				s_index = 0;
				s_och++;
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
	int c_pad,
	int c_ops,
	int c_bypass
> void ConvOp(
	hls::stream<t_input> i_data[c_fh*c_fw],
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_input> &i_bias,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;
	const int c_o_index = c_oh*c_ow;

	/* while(1) { */

#ifndef __SYNTHESIS__
		for (uint8_t s_index = 0; s_index < c_index; s_index++)
			while(i_data[s_index].empty());
		for (uint8_t s_index = 0; s_index < c_index; s_index++)
			while(i_weights[s_index].empty());
		while(i_bias.empty());
#endif

		for (uint16_t s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
			t_acc s_acc_buff[c_och];
			for (uint8_t s_och = 0; s_och < c_och; s_och++)
				s_acc_buff[s_och] = i_bias.read();

			for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
				t_input s_input[c_index];
				for (uint8_t s_index = 0; s_index < c_index; s_index++) {
					s_input[s_index] = i_data[s_index].read();
				}

				ConvComp <
					t_input,
					t_weight,
					t_acc,
					c_och,
					c_index,
					c_ops
				> (
					s_input,
					i_weights,
					s_acc_buff
				);

			}

			for (uint8_t s_och = 0; s_och < c_och; s_och++) {
				t_acc s_acc = s_acc_buff[s_och];
				if (c_relu == 1)
					s_acc = ReluOp<t_acc>(s_acc);
				o_data.write((t_output)(s_acc)); 
			}
		}

#ifndef __SYNTHESIS__

		std::cout << "Waiting for last signal" << std::endl;

#endif

		ap_uint<1> s_last = i_last.read();
		o_last.write(s_last);
		/* if (s_last) */
		/* 	break; */

#ifndef __SYNTHESIS__

		std::cout << "Starting new image" << std::endl;

#endif

	/* } */

#ifndef __SYNTHESIS__
	for (uint8_t s_index = 0; s_index < c_index; s_index++)
		EmptyStream<t_input>(i_data[s_index]);
	for (uint8_t s_index = 0; s_index < c_index; s_index++)
		EmptyStream<t_weight>(i_weights[s_index]);
	std::cout << "BIAS INFO" << std::endl;
	EmptyStream<t_input>(i_bias);
	std::cout << "CONVOP: " << c_ih << " " << c_iw << " " << c_ich << " " << c_str << " " << c_pad << " " << std::endl;
#endif

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
	int c_pad,
	int c_ops,
	int c_bypass
> void ConvOp(
	hls::stream<t_input> i_data[c_fh*c_fw],
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;
	const int c_o_index = c_oh*c_ow;

	/* while(1) { */
#ifndef __SYNTHESIS__
		for (uint8_t s_index = 0; s_index < c_index; s_index++)
			while(i_data[s_index].empty());
		for (uint8_t s_index = 0; s_index < c_index; s_index++)
			while(i_weights[s_index].empty());
#endif

		for (uint16_t s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
			t_acc s_acc_buff[c_och] = {0};

			for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
				t_input s_input[c_index];
				for (uint8_t s_index = 0; s_index < c_index; s_index++) {
					s_input[s_index] = i_data[s_index].read();
				}

				ConvComp <
					t_input,
					t_weight,
					t_acc,
					c_och,
					c_index,
					c_ops
				> (
					s_input,
					i_weights,
					s_acc_buff
				);

			}

			for (uint8_t s_och = 0; s_och < c_och; s_och++) {
				t_acc s_acc = s_acc_buff[s_och];
				if (c_relu == 1)
					s_acc = ReluOp<t_acc>(s_acc);
				o_data.write((t_output)(s_acc)); 
			}
		}

#ifndef __SYNTHESIS__

		std::cout << "Waiting for last signal" << std::endl;

#endif

		ap_uint<1> s_last = i_last.read();
		o_last.write(s_last);
		/* if (s_last) */
		/* 	break; */

#ifndef __SYNTHESIS__

		std::cout << "Starting new image" << std::endl;

#endif

	/* } */

#ifndef __SYNTHESIS__
	for (uint8_t s_index = 0; s_index < c_index; s_index++)
		EmptyStream<t_input>(i_data[s_index]);
	for (uint8_t s_index = 0; s_index < c_index; s_index++)
		EmptyStream<t_weight>(i_weights[s_index]);
	std::cout << "CONVOP: " << c_ih << " " << c_iw << " " << c_ich << " " << c_str << " " << c_pad << " " << std::endl;
#endif

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
	int c_pad,
	int c_ops
> void ConvKernel1x1(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_input> &o_forward,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;

#pragma HLS inline

	hls::stream<t_input> s_compute[c_fh*c_fw];
	#pragma HLS STREAM variable=s_compute depth=10

	hls::stream<ap_uint<1>> s_last[2];
	#pragma HLS STREAM variable=s_last depth=10

	SplitStream<
		2
	> (
		i_last,
		s_last
	);

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
		i_data,
		s_compute[0],
		s_last[0],
		o_forward
	);

	ConvOp<
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
		c_pad,
		c_ops,
		0
	> (
		s_compute,
		i_weights,
		s_last[1],
		o_last,
		o_data
	);

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
	int c_pad,
	int c_ops
> void ConvKernel1x1(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;

#pragma HLS inline

	hls::stream<t_input> s_compute[c_fh*c_fw];
	#pragma HLS STREAM variable=s_compute depth=10

	hls::stream<ap_uint<1>> s_last[2];
	#pragma HLS STREAM variable=s_last depth=10

	SplitStream<
		2
	> (
		i_last,
		s_last
	);

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
		i_data,
		s_last[0],
		s_compute[0]
	);

	ConvOp<
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
		c_pad,
		c_ops,
		0
	> (
		s_compute,
		i_weights,
		s_last[1],
		o_last,
		o_data
	);

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
	int c_pad,
	int c_ops
> void ConvKernel3x3(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;
	hls::stream<t_input> s_data[c_index-1];
	#pragma HLS STREAM variable=s_data[0] depth=c_ich
	#pragma HLS STREAM variable=s_data[1] depth=c_ich
	#pragma HLS STREAM variable=s_data[2] depth=c_ich*(c_iw)
	#pragma HLS STREAM variable=s_data[3] depth=c_ich
	#pragma HLS STREAM variable=s_data[4] depth=c_ich
	#pragma HLS STREAM variable=s_data[5] depth=c_ich*(c_iw)
	#pragma HLS STREAM variable=s_data[6] depth=c_ich
	#pragma HLS STREAM variable=s_data[7] depth=c_ich

	hls::stream<t_input> s_compute[c_index];
	#pragma HLS STREAM variable=s_compute[0] depth=10
	#pragma HLS STREAM variable=s_compute[1] depth=10
	#pragma HLS STREAM variable=s_compute[2] depth=10
	#pragma HLS STREAM variable=s_compute[3] depth=10
	#pragma HLS STREAM variable=s_compute[4] depth=10
	#pragma HLS STREAM variable=s_compute[5] depth=10
	#pragma HLS STREAM variable=s_compute[6] depth=10
	#pragma HLS STREAM variable=s_compute[7] depth=10
	#pragma HLS STREAM variable=s_compute[8] depth=10

#pragma HLS inline

	hls::stream<ap_uint<1>> s_last[10];
	#pragma HLS STREAM variable=s_last depth=11

	SplitStream<
		10
	> (
		i_last,
		s_last
	);

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
		(c_fw-2)
	> (
		s_data[0],
		s_last[1],
		s_compute[1],
		s_data[1]
	);

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
		(c_fw-3)
	> (
		s_data[1],
		s_last[2],
		s_compute[2],
		s_data[2]
	);

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
		(c_fh-2),
		(c_fw-1)
	> (
		s_data[2],
		s_last[3],
		s_compute[3],
		s_data[3]
	);

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
		(c_fh-2),
		(c_fw-2)
	> (
		s_data[3],
		s_last[4],
		s_compute[4],
		s_data[4]
	);

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
		(c_fh-2),
		(c_fw-3)
	> (
		s_data[4],
		s_last[5],
		s_compute[5],
		s_data[5]
	);

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
		(c_fh-3),
		(c_fw-1)
	> (
		s_data[5],
		s_last[6],
		s_compute[6],
		s_data[6]
	);

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
		(c_fh-3),
		(c_fw-2)
	> (
		s_data[6],
		s_last[7],
		s_compute[7],
		s_data[7]
	);

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
		(c_fh-3),
		(c_fw-3)
	> (
		s_data[7],
		s_last[8],
		s_compute[8]
	);

	ConvOp<
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
		c_pad,
		c_ops,
		0
	> (
		s_compute,
		i_weights,
		s_last[9],
		o_last,
		o_data
	);

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
	int c_pad,
	int c_ops
> void ConvKernel3x3(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_input> &o_forward,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;
	hls::stream<t_input> s_data[c_index-1];
	#pragma HLS STREAM variable=s_data[0] depth=c_ich
	#pragma HLS STREAM variable=s_data[1] depth=c_ich
	#pragma HLS STREAM variable=s_data[2] depth=c_ich*(c_iw)
	#pragma HLS STREAM variable=s_data[3] depth=c_ich
	#pragma HLS STREAM variable=s_data[4] depth=c_ich
	#pragma HLS STREAM variable=s_data[5] depth=c_ich*(c_iw)
	#pragma HLS STREAM variable=s_data[6] depth=c_ich
	#pragma HLS STREAM variable=s_data[7] depth=c_ich

	hls::stream<t_input> s_compute[c_index];
	#pragma HLS STREAM variable=s_compute[0] depth=10
	#pragma HLS STREAM variable=s_compute[1] depth=10
	#pragma HLS STREAM variable=s_compute[2] depth=10
	#pragma HLS STREAM variable=s_compute[3] depth=10
	#pragma HLS STREAM variable=s_compute[4] depth=10
	#pragma HLS STREAM variable=s_compute[5] depth=10
	#pragma HLS STREAM variable=s_compute[6] depth=10
	#pragma HLS STREAM variable=s_compute[7] depth=10
	#pragma HLS STREAM variable=s_compute[8] depth=10

#pragma HLS inline

	hls::stream<ap_uint<1>> s_last[10];
	#pragma HLS STREAM variable=s_last depth=11

	SplitStream<
		10
	> (
		i_last,
		s_last
	);

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
		(c_fw-2)
	> (
		s_data[0],
		s_last[1],
		s_compute[1],
		s_data[1]
	);

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
		(c_fw-3)
	> (
		s_data[1],
		s_last[2],
		s_compute[2],
		s_data[2]
	);

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
		(c_fh-2),
		(c_fw-1)
	> (
		s_data[2],
		s_last[3],
		s_compute[3],
		s_data[3]
	);

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
		(c_fh-2),
		(c_fw-2)
	> (
		s_data[3],
		s_last[4],
		s_compute[4],
		s_data[4]
	);

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
		(c_fh-2),
		(c_fw-3)
	> (
		s_data[4],
		s_last[5],
		s_compute[5],
		s_data[5]
	);

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
		(c_fh-3),
		(c_fw-1)
	> (
		s_data[5],
		s_last[6],
		s_compute[6],
		s_data[6]
	);

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
		(c_fh-3),
		(c_fw-2)
	> (
		s_data[6],
		s_last[7],
		s_compute[7],
		s_data[7]
	);

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
		(c_fh-3),
		(c_fw-3)
	> (
		s_data[7],
		s_last[8],
		s_compute[8],
		o_forward
	);

	ConvOp<
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
		c_pad,
		c_ops,
		0
	> (
		s_compute,
		i_weights,
		s_last[9],
		o_last,
		o_data
	);

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
	int c_pad,
	int c_ops,
	int c_bias
> void ConvKernel3x3(
	hls::stream<t_input> &i_data,
	hls::stream<t_input> &i_bias,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;
	hls::stream<t_input> s_data[c_index-1];
	#pragma HLS STREAM variable=s_data[0] depth=c_ich
	#pragma HLS STREAM variable=s_data[1] depth=c_ich
	#pragma HLS STREAM variable=s_data[2] depth=c_ich*(c_iw)
	#pragma HLS STREAM variable=s_data[3] depth=c_ich
	#pragma HLS STREAM variable=s_data[4] depth=c_ich
	#pragma HLS STREAM variable=s_data[5] depth=c_ich*(c_iw)
	#pragma HLS STREAM variable=s_data[6] depth=c_ich
	#pragma HLS STREAM variable=s_data[7] depth=c_ich
	hls::stream<t_input> s_compute[c_index];
	#pragma HLS STREAM variable=s_compute[0] depth=10
	#pragma HLS STREAM variable=s_compute[1] depth=10
	#pragma HLS STREAM variable=s_compute[2] depth=10
	#pragma HLS STREAM variable=s_compute[3] depth=10
	#pragma HLS STREAM variable=s_compute[4] depth=10
	#pragma HLS STREAM variable=s_compute[5] depth=10
	#pragma HLS STREAM variable=s_compute[6] depth=10
	#pragma HLS STREAM variable=s_compute[7] depth=10
	#pragma HLS STREAM variable=s_compute[8] depth=10

#pragma HLS inline

	hls::stream<ap_uint<1>> s_last[10];
	#pragma HLS STREAM variable=s_last depth=11

	SplitStream<
		10
	> (
		i_last,
		s_last
	);

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
		(c_fw-2)
	> (
		s_data[0],
		s_last[1],
		s_compute[1],
		s_data[1]
	);

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
		(c_fw-3)
	> (
		s_data[1],
		s_last[2],
		s_compute[2],
		s_data[2]
	);

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
		(c_fh-2),
		(c_fw-1)
	> (
		s_data[2],
		s_last[3],
		s_compute[3],
		s_data[3]
	);

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
		(c_fh-2),
		(c_fw-2)
	> (
		s_data[3],
		s_last[4],
		s_compute[4],
		s_data[4]
	);

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
		(c_fh-2),
		(c_fw-3)
	> (
		s_data[4],
		s_last[5],
		s_compute[5],
		s_data[5]
	);

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
		(c_fh-3),
		(c_fw-1)
	> (
		s_data[5],
		s_last[6],
		s_compute[6],
		s_data[6]
	);

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
		(c_fh-3),
		(c_fw-2)
	> (
		s_data[6],
		s_last[7],
		s_compute[7],
		s_data[7]
	);

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
		(c_fh-3),
		(c_fw-3)
	> (
		s_data[7],
		s_last[8],
		s_compute[8]
	);

	ConvOp<
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
		c_pad,
		c_ops,
		0
	> (
		s_compute,
		i_weights,
		i_bias,
		s_last[9],
		o_last,
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
	int c_pad,
	int c_ops
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_input> &i_bias,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data
) {

	const int c_conv_index = c_fh*c_fw;
	hls::stream<t_input> s_data_in_stream("data_in");
	#pragma HLS STREAM variable=s_data_in_stream depth=3
	hls::stream<t_input> s_data_out_stream("data_out");
	#pragma HLS STREAM variable=s_data_out_stream depth=3

#pragma HLS inline

#ifndef __SYNTHESIS__

	while(i_data.empty());
	while(i_bias.empty());

	for (int s_index = 0; s_index < c_fh*c_fw; s_index++) {
		if (i_weights[s_index].empty()) {
			return;
		}
	}

#endif

	hls::stream<ap_uint<1>> s_last;
	#pragma HLS STREAM variable=s_last depth=10

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
		i_last,
		s_last,
		s_data_in_stream
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
		c_pad,
		c_ops,
		1
	> (
		s_data_in_stream,
		i_bias,
		i_weights,
		s_last,
		o_last,
		o_data
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
	int c_pad,
	int c_ops
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_input> &i_bias,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> o_data[c_split]
) {


	const int c_conv_index = c_fh*c_fw;
	hls::stream<t_input> s_data_in_stream("data_in");
	#pragma HLS STREAM variable=s_data_in_stream depth=3
	hls::stream<t_output> s_out_stream;
	#pragma HLS STREAM variable=s_out_stream depth=3

#pragma HLS inline

#ifndef __SYNTHESIS__

	while(i_data.empty());
	while(i_bias.empty());

	for (int s_index = 0; s_index < c_fh*c_fw; s_index++) {
		if (i_weights[s_index].empty()) {
			return;
		}
	}

#endif

	hls::stream<ap_uint<1>> s_last[2];
	#pragma HLS STREAM variable=s_last depth=10

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
		i_last,
		s_last[0],
		s_data_in_stream
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
		c_pad,
		c_ops,
		1
	> (
		s_data_in_stream,
		i_bias,
		i_weights,
		s_last[0],
		s_last[1],
		s_out_stream
	);

	SplitStream<
		t_output,
		c_och,
		c_ow,
		c_oh,
		c_split
	> (
		s_out_stream,
		s_last[1],
		o_last,
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
	int c_pad,
	int c_ops
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[1],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data,
	hls::stream<t_input> &o_forward
) {


	hls::stream<t_input> s_data_stream("data_out_1x1");
	#pragma HLS STREAM variable=s_data_stream depth=3

#pragma HLS inline

#ifndef __SYNTHESIS__

	while(i_data.empty());

	for (int s_index = 0; s_index < 1; s_index++) {
		if (i_weights[s_index].empty()) {
			return;
		}
	}

#endif

	hls::stream<ap_uint<1>> s_last;
	#pragma HLS STREAM variable=s_last depth=10

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
		c_pad,
		c_ops
	> (
		i_data,
		i_weights,
		s_data_stream,
		i_last,
		s_last,
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
		s_last,
		o_last,
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
	int c_pad,
	int c_ops
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data,
	hls::stream<t_input> &o_forward
) {


	const int c_conv_index = c_fh*c_fw;
	hls::stream<t_input> s_data_in_stream("data_in");
	#pragma HLS STREAM variable=s_data_in_stream depth=3
	hls::stream<t_output> s_data_out_stream;
	#pragma HLS STREAM variable=s_data_out_stream depth=3

#pragma HLS inline

#ifndef __SYNTHESIS__

	while(i_data.empty());

	for (int s_index = 0; s_index < c_fh*c_fw; s_index++) {
		if (i_weights[s_index].empty()) {
			return;
		}
	}

#endif

	hls::stream<ap_uint<1>> s_last[2];
	#pragma HLS STREAM variable=s_last depth=10

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
		i_last,
		s_last[0],
		s_data_in_stream
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
		c_pad,
		c_ops
	> (
		s_data_in_stream,
		i_weights,
		s_last[0],
		s_last[1],
		s_data_out_stream,
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
		s_data_out_stream,
		s_last[1],
		o_last,
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
	int c_pad,
	int c_ops
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[1],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data
) {

#pragma HLS inline

#ifndef __SYNTHESIS__

	while(i_data.empty());

	for (int s_index = 0; s_index < 1; s_index++) {
		if (i_weights[s_index].empty()) {
			return;
		}
	}

#endif

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
		c_pad,
		c_ops
	> (
		i_data,
		i_weights,
		i_last,
		o_last,
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
	int c_relu,
	int c_str,
	int c_pad,
	int c_ops
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data
) {

	const int c_conv_index = c_fh*c_fw;
	hls::stream<t_input> s_data_in_stream("data_in");
	#pragma HLS STREAM variable=s_data_in_stream depth=3

#pragma HLS inline

#ifndef __SYNTHESIS__

	while(i_data.empty());

	for (int s_index = 0; s_index < c_fh*c_fw; s_index++) {
		if (i_weights[s_index].empty()) {
			return;
		}
	}

#endif

	hls::stream<ap_uint<1>> s_last;
	#pragma HLS STREAM variable=s_last depth=10

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
		i_last,
		s_last,
		s_data_in_stream
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
		c_pad,
		c_ops
	> (
		s_data_in_stream,
		i_weights,
		s_last,
		o_last,
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
	int c_relu,
	int c_split,
	int c_str,
	int c_pad,
	int c_ops
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> o_data[c_split]
) {



	const int c_conv_index = c_fh*c_fw;
	hls::stream<t_input> s_data_in_stream("data_in");
	#pragma HLS STREAM variable=s_data_in_stream depth=3
	hls::stream<t_output> s_out_stream;
	#pragma HLS STREAM variable=s_out_stream depth=3

#pragma HLS inline

#ifndef __SYNTHESIS__

	while(i_data.empty());

	for (int s_index = 0; s_index < c_fh*c_fw; s_index++) {
		if (i_weights[s_index].empty()) {
			return;
		}
	}

#endif

	hls::stream<ap_uint<1>> s_last[2];
	#pragma HLS STREAM variable=s_last depth=10

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
		i_last,
		s_last[0],
		s_data_in_stream
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
		c_pad,
		c_ops
	> (
		s_data_in_stream,
		i_weights,
		s_last[0],
		s_last[1],
		s_out_stream	
	);

	SplitStream<
		t_output,
		c_och,
		c_ow,
		c_oh,
		c_split
	> (
		s_out_stream,
		s_last[1],
		o_last,
		o_data
	);

}

#endif
