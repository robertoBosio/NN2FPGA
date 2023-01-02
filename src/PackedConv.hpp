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
	int c_ich,
	int c_och,
	int c_index,
	int c_str,
	int c_ops
> void ConvComp(
	hls::stream<t_input> i_input[c_index],
	hls::stream<t_weight> i_weights[c_index],
	hls::stream<t_acc> o_acc_stream[c_ops]
) {
	/* Assuming that the number of computations is a multiplier of the number */
	/* of operations */
	const int c_num_comp = c_ich*c_och;
	const int c_num_och = c_och/c_ops;
	const int c_pipe_iter = c_num_comp/c_ops;

	t_acc s_acc_buff[c_och];
#pragma HLS array_partition variable=s_acc_buff type=complete
	t_input s_input[c_index];
#pragma HLS array_partition variable=s_input type=complete

	for (uint16_t s_pipe_iter = 0; s_pipe_iter < c_pipe_iter; s_pipe_iter++) {

#pragma HLS pipeline

		if (s_pipe_iter == 0) {
			for (uint8_t s_och = 0; s_och < c_och; s_och++)
				s_acc_buff[s_och] = 0;
		}


		uint8_t s_num_och = s_pipe_iter % c_num_och;
		uint8_t s_ich = s_pipe_iter / c_num_och;

		if (s_num_och == 0) {
			for (uint8_t s_index = 0; s_index < c_index; s_index++) {
				s_input[s_index] = i_input[s_index].read();
			}
		}

		/* Buffering to speed up computations */
		/* TODO: Adjust for generic bit quantizations */
		int8_t s_weight[c_ops][c_index];
#pragma HLS array_partition variable=s_weight type=complete

		for (uint8_t s_index = 0; s_index < c_index; s_index++) {
			t_weight s_tmp_weight = i_weights[s_index].read();
			for (uint8_t s_ops = 0; s_ops < c_ops; s_ops++) {
				/* Input weights are reversed with respect to original order */
				s_weight[s_ops][c_index - 1 - s_index] = s_tmp_weight(8*(s_ops+1)-1, 8*s_ops);
			}
		}

	/* TODO: try unroll and pipeline of the inner loop */
		COMPUTE: for (uint8_t s_ops = 0; s_ops < c_ops; s_ops++) {
			uint8_t s_och = s_num_och*c_ops + s_ops;
			t_acc s_acc = 0;
			for (uint8_t s_index = 0; s_index < c_index; s_index++) {
				s_acc += s_input[s_index] * s_weight[s_ops][s_index];
			}
			if (s_ich == (c_ich - 1)) {
				s_acc += s_acc_buff[s_och];
				o_acc_stream[s_ops].write(s_acc);
			} else
				s_acc_buff[s_och] += s_acc;
		}

	}

}

template <
	class t_input,
	class t_weight,
	class t_acc,
	int c_ich,
	int c_och,
	int c_oh,
	int c_ow,
	int c_index,
	int c_str,
	int c_ops
> void ConvComp(
	hls::stream<t_input> i_input[c_index],
	hls::stream<t_weight> i_weights[c_index],
	hls::stream<t_acc> o_acc_stream[c_ops]
) {

	const int c_num_comp = c_ich*c_och;
	const int c_pipe_iter = c_num_comp/c_ops;
	const int c_o_index = c_oh*c_ow*c_pipe_iter;
	const int c_num_och = c_och/c_ops;

	t_acc s_acc_buff[c_och];
#pragma HLS array_partition variable=s_acc_buff type=complete
	t_input s_input[c_index];
#pragma HLS array_partition variable=s_input type=complete

	for (uint32_t s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
#pragma HLS pipeline

		uint16_t s_pipe_iter = s_o_index % c_pipe_iter; 
		uint8_t s_num_och = s_pipe_iter % c_num_och;
		uint8_t s_ich = s_pipe_iter / c_num_och;

		if (s_pipe_iter == 0) {
			for (uint8_t s_och = 0; s_och < c_och; s_och++)
				s_acc_buff[s_och] = 0;
		}

		if (s_num_och == 0) {
			for (uint8_t s_index = 0; s_index < c_index; s_index++) {
				s_input[s_index] = i_input[s_index].read();
			}
		}

		/* Buffering to speed up computations */
		/* TODO: Adjust for generic bit quantizations */
		int8_t s_weight[c_ops][c_index];
#pragma HLS array_partition variable=s_weight type=complete

		for (uint8_t s_index = 0; s_index < c_index; s_index++) {
			t_weight s_tmp_weight = i_weights[s_index].read();
			for (uint8_t s_ops = 0; s_ops < c_ops; s_ops++) {
				/* Input weights are reversed with respect to original order */
				s_weight[s_ops][c_index - 1 - s_index] = s_tmp_weight(8*(s_ops+1)-1, 8*s_ops);
			}
		}

	/* TODO: try unroll and pipeline of the inner loop */
		COMPUTE: for (uint8_t s_ops = 0; s_ops < c_ops; s_ops++) {
			uint8_t s_och = s_num_och*c_ops + s_ops;
			t_acc s_acc = 0;
			for (uint8_t s_index = 0; s_index < c_index; s_index++) {
				s_acc += s_input[s_index] * s_weight[s_ops][s_index];
			}
			if (s_ich == (c_ich - 1)) {
				s_acc += s_acc_buff[s_och];
				o_acc_stream[s_ops].write(s_acc);
			} else
				s_acc_buff[s_och] += s_acc;
		}

#ifndef __SYNTHESIS__
#ifdef DEBUG
				std::cout << "OUTPUT VALUES" << std::endl;
#endif
#endif

	}

}

template <
	class t_output,
	class t_acc,
	int c_ich,
	int c_och,
	int c_oh,
	int c_ow,
	int c_index,
	int c_ops,
	int c_relu,
	int c_quant,
	int c_shift_h,
	int c_shift_l
> void StreamOutput(
	hls::stream<t_acc> i_acc[c_ops],
	hls::stream<t_output> &o_data
) {

	const int c_num_comp = c_oh*c_ow*c_och;
	const int c_pipe_iter = c_num_comp/c_ops;

	for (uint32_t s_pipe_iter = 0; s_pipe_iter < c_pipe_iter; s_pipe_iter++) {
#pragma HLS pipeline
		for (uint8_t s_ops = 0; s_ops < c_ops; s_ops++) {
			t_acc s_acc = c_quant;

			/* 1 subtraction for quantization */
			s_acc += i_acc[s_ops].read();

			t_output s_output;

			if (c_relu == 1) {
				s_acc = ReluOp<t_acc>(s_acc);
				/* TODO: write generic version for different bit quantization*/
				/* if ((s_acc(c_shift_h, c_shift_l)) >= 256) */
				if ((s_acc >> c_shift_l) >= 256)
					s_output = 255;
				else {
					/* s_output = (uint8_t)(s_acc && ((128 << c_shift) -1)) > (32 << c_shift); */
					s_output = s_acc(c_shift_h, c_shift_l);
				}
			} else {
				/* s_output = s_acc(c_shift_h, c_shift_l); */
				s_output = s_acc;
			}

			o_data.write(s_output); 
		}
	}

}

template <
	class t_input,
	class t_output,
	class t_acc,
	int c_ich,
	int c_och,
	int c_index,
	int c_ops,
	int c_relu,
	int c_quant,
	int c_shift_h,
	int c_shift_l
> void StreamOutput(
	hls::stream<t_acc> i_acc[c_ops],
	hls::stream<t_input> &i_bias,
	hls::stream<t_output> &o_data
) {

	const int c_num_comp = c_och;
	const int c_pipe_iter = c_num_comp/c_ops;

	uint8_t s_ops = 0;

	for (uint8_t s_pipe_iter = 0; s_pipe_iter < c_pipe_iter; s_pipe_iter++) {
		for (uint8_t s_ops = 0; s_ops < c_ops; s_ops++) {
			t_acc s_acc = (i_bias.read() << c_shift_l) + c_quant;

			/* 1 subtraction for quantization */
			s_acc += i_acc[s_ops].read();

#ifndef __SYNTHESIS__
#ifdef DEBUG_LINE
			/* if (s_o_index == 0) */
				/* std::cout << (ap_int<32>)(s_acc) << " "; */
#endif
#endif

			t_output s_output;

			if (c_relu == 1) {
				s_acc = ReluOp<t_acc>(s_acc);
				/* TODO: write generic version for different bit quantization*/
				if ((s_acc >> c_shift_l) >= 256)
					s_output = 255;
				else {
					/* s_output = (uint8_t)(s_acc && ((128 << c_shift) -1)) > (32 << c_shift); */
					s_output = s_acc(c_shift_h, c_shift_l);
				}
			} else {
				s_output = s_acc(c_shift_h, c_shift_l);
			}

#ifndef __SYNTHESIS__
#ifdef DEBUG
			std::cout << (ap_int<32>)(s_acc) << " ";
			std::cout << (ap_uint<8>)(s_output) << " ";
#endif
#ifdef DEBUG_LINE
			/* if (s_o_index == 0) */
				/* std::cout << (ap_uint<8>)(s_output) << " "; */
#endif
#endif
			o_data.write(s_output); 
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
	int c_scale,
	int c_bypass
> void ConvOp(
	hls::stream<t_input> i_data[c_fh*c_fw],
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_input> &i_bias,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;
	const int c_o_index = c_oh*c_ow;
	/* TODO: handle different bit width with quantization */
	/* const int c_quant = -1 * 128; */
	const int c_quant = 0;
	const int c_shift_l = 7 + c_scale;
	const int c_shift_h = c_shift_l + 7;

	/* while(1) { */

#ifndef __SYNTHESIS__
#ifdef DEBUG
		std::cout << "CONVOP_"; 
		std::cout << c_ich << "_";
		std::cout << c_och << "_";
		std::cout << c_ih << "_";
		std::cout << c_iw << std::endl; 
#endif
		for (uint8_t s_index = 0; s_index < c_index; s_index++)
			while(i_data[s_index].empty());
		for (uint8_t s_index = 0; s_index < c_index; s_index++)
			while(i_weights[s_index].empty());
		while(i_bias.empty());
#endif

		for (uint16_t s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
#pragma HLS dataflow
			hls::stream<t_acc> s_acc_stream[c_ops];
#pragma HLS STREAM variable=s_acc_stream depth=c_och

			ConvComp <
				t_input,
				t_weight,
				t_acc,
				c_ich,
				c_och,
				c_index,
				c_str,
				c_ops
			> (
				i_data,
				i_weights,
				s_acc_stream
			);

#ifndef __SYNTHESIS__
#ifdef DEBUG
				std::cout << "OUTPUT VALUES" << std::endl;
#endif
#endif

			StreamOutput <
				t_input,
				t_output,
				t_acc,
				c_ich,
				c_och,
				c_index,
				c_ops,
				c_relu,
				c_quant,
				c_shift_h,
				c_shift_l
			> (
				s_acc_stream,
				i_bias,
				o_data
			);

#ifndef __SYNTHESIS__
#ifdef DEBUG_LINE
			if (s_o_index == 0) {
				std::cout << std::endl;
				std::cout << "CONVOP: " << c_ih << " " << c_iw << " " << c_ich << " " << c_str << " " << c_pad << " " << std::endl;
			}
#endif
#endif

#ifndef __SYNTHESIS__
#ifdef DEBUG
			std::cout << std::endl;
#endif
#endif

		}

#ifndef __SYNTHESIS__

#ifdef DEBUG
		std::cout << std::endl;
#endif

#endif

		/* 	break; */

#ifndef __SYNTHESIS__

#ifdef DEBUG
		std::cout << "Starting new image" << std::endl;
#endif

#endif

	/* } */

#ifndef __SYNTHESIS__
	for (uint8_t s_index = 0; s_index < c_index; s_index++)
		EmptyStream<t_input>(i_data[s_index]);
	for (uint8_t s_index = 0; s_index < c_index; s_index++)
		EmptyStream<t_weight>(i_weights[s_index]);

#ifdef DEBUG
	std::cout << "BIAS INFO" << std::endl;
#endif

	EmptyStream<t_input>(i_bias);

#ifdef DEBUG
	std::cout << "CONVOP: " << c_ih << " " << c_iw << " " << c_ich << " " << c_str << " " << c_pad << " " << std::endl;
#endif
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
	int c_scale,
	int c_bypass
> void ConvOp(
	hls::stream<t_input> i_data[c_fh*c_fw],
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_output> &o_data
) {

#pragma HLS inline

	const int c_index = c_fh*c_fw;
	/* TODO: handle different bit width with quantization */
	/* const int c_quant = -1 * 128; */
	const int c_quant = 0;
	const int c_shift_l = 7 + c_scale;
	const int c_shift_h = c_shift_l + 7;


	/* while(1) { */
#ifndef __SYNTHESIS__
	#ifdef DEBUG
		std::cout << "CONVOP_"; 
		std::cout << c_ich << "_";
		std::cout << c_och << "_";
		std::cout << c_ih << "_";
		std::cout << c_iw << std::endl; 
	#endif
		for (uint8_t s_index = 0; s_index < c_index; s_index++)
			while(i_data[s_index].empty());
		for (uint8_t s_index = 0; s_index < c_index; s_index++)
			while(i_weights[s_index].empty());
#endif


	hls::stream<t_acc> s_acc_stream[c_ops];
#pragma HLS STREAM variable=s_acc_stream depth=c_och

	ConvComp <
		t_input,
		t_weight,
		t_acc,
		c_ich,
		c_och,
		c_oh,
		c_ow,
		c_index,
		c_str,
		c_ops
	> (
		i_data,
		i_weights,
		s_acc_stream
	);

	StreamOutput <
		t_output,
		t_acc,
		c_ich,
		c_och,
		c_oh,
		c_ow,
		c_index,
		c_ops,
		c_relu,
		c_quant,
		c_shift_h,
		c_shift_l
	> (
		s_acc_stream,
		o_data
	);

#ifndef __SYNTHESIS__

#ifdef DEBUG
		std::cout << std::endl;
#endif

#endif

		/* 	break; */

#ifndef __SYNTHESIS__

#ifdef DEBUG
		std::cout << "Starting new image" << std::endl;
#endif

#endif

	/* } */

#ifndef __SYNTHESIS__
	for (uint8_t s_index = 0; s_index < c_index; s_index++)
		EmptyStream<t_input>(i_data[s_index]);
	for (uint8_t s_index = 0; s_index < c_index; s_index++)
		EmptyStream<t_weight>(i_weights[s_index]);
#ifdef DEBUG
	std::cout << "CONVOP: " << c_ih << " " << c_iw << " " << c_ich << " " << c_str << " " << c_pad << " " << std::endl;
#endif
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
	int c_scale,
	int c_ops
> void ConvKernel1x1(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_input> &o_forward,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;

#pragma HLS inline

	hls::stream<t_input> s_compute[c_fh*c_fw];
	#pragma HLS STREAM variable=s_compute depth=10


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
		c_scale,
		0
	> (
		s_compute,
		i_weights,
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
	int c_scale,
	int c_ops
> void ConvKernel1x1(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_output> &o_data
) {

	const int c_index = c_fh*c_fw;

#pragma HLS inline

	hls::stream<t_input> s_compute[c_fh*c_fw];
	#pragma HLS STREAM variable=s_compute depth=10


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
		c_scale,
		0
	> (
		s_compute,
		i_weights,
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
	int c_scale,
	int c_ops
> void ConvKernel3x3(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
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
		c_scale,
		0
	> (
		s_compute,
		i_weights,
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
	int c_scale,
	int c_ops
> void ConvKernel3x3_1x1(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights3x3[c_fh*c_fw],
	hls::stream<t_weight> i_weights1x1[1],
	hls::stream<t_output> &o_data3x3,
	hls::stream<t_output> &o_data1x1
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

	hls::stream<t_input> s_data1x1[2];
	#pragma HLS STREAM variable=s_data[0] depth=10
	#pragma HLS STREAM variable=s_data[1] depth=10

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

	hls::stream<t_input> s_compute1x1[1];
	#pragma HLS STREAM variable=s_compute1x1[0] depth=10

#pragma HLS inline


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
		s_compute[8],
		s_data1x1[0]
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
		c_scale,
		0
	> (
		s_compute,
		i_weights3x3,
		o_data3x3
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
		s_data1x1[0],
		s_data1x1[1]
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
		c_scale,
		0
	> (
		s_compute1x1,
		i_weights1x1,
		o_data1x1
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
	int c_scale,
	int c_ops
> void ConvKernel3x3(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
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
		c_scale,
		0
	> (
		s_compute,
		i_weights,
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
	int c_scale,
	int c_bias
> void ConvKernel3x3(
	hls::stream<t_input> &i_data,
	hls::stream<t_input> &i_bias,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
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
		c_scale,
		0
	> (
		s_compute,
		i_weights,
		i_bias,
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
	int c_scale,
	int c_ops
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_input> &i_bias,
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
		c_scale,
		1
	> (
		s_data_in_stream,
		i_bias,
		i_weights,
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
	int c_scale,
	int c_ops
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_input> &i_bias,
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
		c_scale,
		1
	> (
		s_data_in_stream,
		i_bias,
		i_weights,
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
	int c_scale,
	int c_ops
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[1],
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
		c_scale,
		c_ops
	> (
		i_data,
		i_weights,
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
	int c_pad,
	int c_scale,
	int c_ops
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
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
		c_scale,
		c_ops
	> (
		s_data_in_stream,
		i_weights,
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
	int c_scale,
	int c_ops
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[1],
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
		c_scale,
		c_ops
	> (
		i_data,
		i_weights,
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
	int c_scale,
	int c_ops
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
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
		c_scale,
		c_ops
	> (
		s_data_in_stream,
		i_weights,
		o_data
	);

}

/* ADDED FOR RESIDUAL LAYERS */
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
	int c_scale,
	int c_ops
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights3x3[c_fh*c_fw],
	hls::stream<t_weight> i_weights1x1[1],
	hls::stream<t_output> &o_data3x3,
	hls::stream<t_output> &o_data1x1
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
		if (i_weights3x3[s_index].empty()) {
			return;
		}
	}

	for (int s_index = 0; s_index < 1; s_index++) {
		if (i_weights1x1[s_index].empty()) {
			return;
		}
	}

#endif


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
		s_data_in_stream
	);

	ConvKernel3x3_1x1 <
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
		c_scale,
		c_ops
	> (
		s_data_in_stream,
		i_weights3x3,
		i_weights1x1,
		o_data3x3,
		o_data1x1
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
	int c_scale,
	int c_ops
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> i_weights[c_fh*c_fw],
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
		c_scale,
		c_ops
	> (
		s_data_in_stream,
		i_weights,
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
		o_data
	);

}

#endif
