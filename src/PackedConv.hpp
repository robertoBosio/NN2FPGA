#ifndef __PACKEDCONV__
#define __PACKEDCONV__

#include "ap_int.h"
#include "hls_stream.h"
#include "Utils.hpp"
#include "ActivationStreams.hpp"
#include "Debug.hpp"

template <
	class t_input_struct,
	class t_input,
	class t_weight,
	class t_acc_struct,
	class t_acc,
	int c_ich,
	int c_och,
	int c_oh,
	int c_ow,
	int c_index,
	int c_str,
	int c_ops
> void ConvComp(
	hls::stream<t_input_struct> i_input[c_index],
	hls::stream<t_weight> i_weights[c_index],
	hls::stream<t_acc_struct> o_acc_stream[c_ops]
) {
#pragma HLS inline

	const int c_num_comp = c_ich*c_och;
	const int c_pipe_iter = c_num_comp/c_ops;
	const int c_o_index = c_oh*c_ow*c_pipe_iter;
	const int c_num_och = c_och/c_ops;

	t_acc s_acc_buff[c_och];
#pragma HLS array_partition variable=s_acc_buff type=complete
	t_input s_input[c_index];
#pragma HLS array_partition variable=s_input type=complete
	bool s_last;

	for (uint32_t s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
#pragma HLS pipeline style=frp

		uint16_t s_pipe_iter = s_o_index % c_pipe_iter; 
		uint8_t s_num_och = s_pipe_iter % c_num_och;
		uint8_t s_ich = s_pipe_iter / c_num_och;

		if (s_pipe_iter == 0) {
			for (uint8_t s_och = 0; s_och < c_och; s_och++)
				s_acc_buff[s_och] = 0;
		}

		if (s_num_och == 0) {
			for (uint8_t s_index = 0; s_index < c_index; s_index++) {
				t_input_struct s_input_stuct = i_input[s_index].read();
				s_input[s_index] = s_input_stuct.data;
				/* Sending last only at the bottom left data */
				if (s_index == 0)
					s_last = s_input_stuct.last;
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
				s_weight[s_ops][s_index] = s_tmp_weight[s_ops];
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
				t_acc_struct s_acc_struct;
				s_acc_struct.data = s_acc;
				if (s_och == (c_och-1))
					s_acc_struct.last = s_last;
				else
					s_acc_struct.last = false;
				o_acc_stream[s_ops].write(s_acc_struct);
			} else
				s_acc_buff[s_och] += s_acc;
		}

	}

}

template <
	class t_output_struct,
	class t_acc_struct,
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
	hls::stream<t_acc_struct> i_acc[c_ops],
	hls::stream<t_output_struct> &o_data
) {
#pragma HLS inline

	const int c_num_comp = c_oh*c_ow*c_och;
	const int c_pipe_iter = c_num_comp;

	for (uint32_t s_pipe_iter = 0; s_pipe_iter < c_pipe_iter; s_pipe_iter++) {
#pragma HLS pipeline style=frp
		uint8_t s_ops = s_pipe_iter % c_ops;
		t_acc_struct s_acc_struct = i_acc[s_ops].read();
		t_acc s_acc = c_quant;

		/* 1 subtraction for quantization */
		s_acc += s_acc_struct.data;

		t_output_struct s_output;

		if (c_relu == 1) {
			s_acc = ReluOp<t_acc>(s_acc);
			/* TODO: write generic version for different bit quantization*/
			/* if ((s_acc(c_shift_h, c_shift_l)) >= 256) */
			if ((s_acc >> c_shift_l) >= 256)
				s_output.data = 255;
			else {
				/* s_output = (uint8_t)(s_acc && ((128 << c_shift) -1)) > (32 << c_shift); */
				s_output.data = s_acc(c_shift_h, c_shift_l);
			}
		} else {
			/* s_output = s_acc(c_shift_h, c_shift_l); */
			s_output.data = s_acc;
		}
		s_output.last = s_acc_struct.last;

		o_data.write(s_output); 
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

			o_data.write(s_output); 
		}
	}

}

template <
	class t_input,
	class t_weight,
	class t_output,
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
	int c_pad,
	int c_ops,
	int c_scale
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

	}

}

template <
	class t_input_struct,
	class t_input,
	class t_weight,
	class t_output_struct,
	class t_output,
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
	int c_pad,
	int c_ops,
	int c_scale
> void ConvOp(
	hls::stream<t_input_struct> i_data[c_fh*c_fw],
	hls::stream<t_weight> i_weights[c_fh*c_fw],
	hls::stream<t_output_struct> &o_data
) {

#pragma HLS inline

	const int c_index = c_fh*c_fw;
	/* TODO: handle different bit width with quantization */
	/* const int c_quant = -1 * 128; */
	const int c_quant = 0;
	const int c_shift_l = 7 + c_scale;
	const int c_shift_h = c_shift_l + 7;


	typedef struct {
		t_acc data;
		bool last;
	} t_acc_struct;

	hls::stream<t_acc_struct> s_acc_stream[c_ops];
#pragma HLS STREAM variable=s_acc_stream depth=c_och

	ConvComp <
		t_input_struct,
		t_input,
		t_weight,
		t_acc_struct,
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
		t_output_struct,
		t_acc_struct,
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

}

#endif
