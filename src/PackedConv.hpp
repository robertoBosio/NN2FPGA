#ifndef __PACKEDCONV__
#define __PACKEDCONV__

#include "ap_int.h"
#include "hls_stream.h"
#include "Utils.hpp"
#include "ActivationStreams.hpp"
#include "Debug.hpp"

//////////////////////////////////////////////////////////////////////////////
/* Version for Off-Chip memory weights */

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
	int c_ops,
	int c_reuse
> void ConvComp(
	hls::stream<t_input_struct> i_input[c_index],
	hls::stream<t_weight> i_weights[c_index],
	hls::stream<t_acc_struct> o_acc_stream[c_ops]
) {
/* #pragma HLS inline */

	const auto c_o_index = c_oh*c_ow/c_reuse;
	const auto c_num_och = c_och/c_ops;
	const auto c_iter    = c_reuse*c_num_och;

	t_acc s_acc_buff[c_reuse][c_och];
#pragma HLS array_partition variable=s_acc_buff type=complete dim=0
	t_input s_input[c_reuse][c_index];
#pragma HLS array_partition variable=s_input type=complete
	bool s_last = false;
	int8_t s_weight[c_ops][c_index];
#pragma HLS array_partition variable=s_weight type=complete

		/* for (auto s_reuse = 0; s_reuse < c_reuse; s_reuse++) */
		/* 	for (auto s_och = 0; s_och < c_och; s_och++) */
		/* 		s_acc_buff[s_reuse][s_och] = 0; */

	for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
		for (auto s_ich = 0; s_ich < c_ich; s_ich++) {
/* #if c_iter==1 */
/* #pragma HLS pipeline style=flp */
/* #endif */
			for (auto s_iter = 0; s_iter < c_iter; s_iter++) {
/* #if c_iter>1 */
/* #pragma HLS pipeline style=flp */
/* #endif */
#pragma HLS pipeline style=flp

				auto s_reuse = s_iter % c_reuse;
				auto s_num_och = s_iter / c_reuse;

				if (s_num_och == 0) {
					for (auto s_index = 0; s_index < c_index; s_index++) {
						t_input_struct s_input_struct = i_input[s_index].read();
						s_input[s_reuse][s_index] = s_input_struct.data;
						/* Sending last only at the bottom right data */
						if (s_index == 0)
							s_last = s_input_struct.last;
					}
				}

				/* Buffering to speed up computations */
				/* TODO: Adjust for generic bit quantizations */
				if (s_reuse == 0) {
					for (auto s_index = 0; s_index < c_index; s_index++) {
						t_weight s_tmp_weight = i_weights[s_index].read();
						for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
							/* Input weights are reversed with respect to original order */
							s_weight[s_ops][s_index] = s_tmp_weight[s_ops];
						}
					}
				}

				COMPUTE: for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
					auto s_och = s_num_och*c_ops + s_ops;
					t_acc s_acc;
					t_acc s_acc_base;
					t_acc_struct s_acc_struct;
					s_acc_struct.last = s_last & (s_och == (c_och-1));

					s_acc = 0;

					s_acc_base = s_acc_buff[s_reuse][s_och];

					for (auto s_index = 0; s_index < c_index; s_index++) {
						s_acc += s_input[s_reuse][s_index] * s_weight[s_ops][s_index];
					}

					if (s_ich != 0)
						s_acc += s_acc_base;

					s_acc_struct.data = s_acc;

					if (s_ich == (c_ich - 1)) {
						o_acc_stream[s_ops].write(s_acc_struct);
					} else {
						s_acc_buff[s_reuse][s_och] = s_acc;
					}
				}
			}
		}
	}

}

template <
	class t_input_struct,
	class t_input,
	class t_weight,
	class t_bias,
	class t_acc_struct,
	class t_acc,
	int c_ich,
	int c_och,
	int c_oh,
	int c_ow,
	int c_index,
	int c_str,
	int c_ops,
	int c_reuse
> void ConvComp(
	hls::stream<t_input_struct> i_input[c_index],
	hls::stream<t_weight> i_weights[c_index],
	hls::stream<t_bias> i_bias[1],
	hls::stream<t_acc_struct> o_acc_stream[c_ops]
) {
/* #pragma HLS inline */

	const auto c_o_index = c_oh*c_ow/c_reuse;
	const auto c_num_och = c_och/c_ops;
	const auto c_iter    = c_reuse*c_num_och;

	t_acc s_acc_buff[c_reuse][c_och];
#pragma HLS array_partition variable=s_acc_buff type=complete dim=0
	t_input s_input[c_reuse][c_index];
#pragma HLS array_partition variable=s_input type=complete
	bool s_last = false;
	int8_t s_weight[c_ops][c_index];
#pragma HLS array_partition variable=s_weight type=complete
	t_bias s_bias;
#pragma HLS array_partition variable=s_bias type=complete

		/* for (auto s_reuse = 0; s_reuse < c_reuse; s_reuse++) */
		/* 	for (auto s_och = 0; s_och < c_och; s_och++) */
		/* 		s_acc_buff[s_reuse][s_och] = 0; */

	for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
		for (auto s_ich = 0; s_ich < c_ich; s_ich++) {
			for (auto s_iter = 0; s_iter < c_iter; s_iter++) {
#pragma HLS pipeline style=flp

				auto s_reuse = s_iter % c_reuse;
				auto s_num_och = s_iter / c_reuse;

				if (s_num_och == 0) {
					for (auto s_index = 0; s_index < c_index; s_index++) {
						t_input_struct s_input_struct = i_input[s_index].read();
						s_input[s_reuse][s_index] = s_input_struct.data;
						/* Sending last only at the bottom right data */
						if (s_index == 0)
							s_last = s_input_struct.last;
					}
				}

				/* Buffering to speed up computations */
				/* TODO: Adjust for generic bit quantizations */
				if (s_reuse == 0) {
					for (auto s_index = 0; s_index < c_index; s_index++) {
						t_weight s_tmp_weight = i_weights[s_index].read();
						for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
							/* Input weights are reversed with respect to original order */
							s_weight[s_ops][s_index] = s_tmp_weight[s_ops];
						}
					}
				}
        if ((s_ich == 0) & (s_reuse == 0))
          s_bias = i_bias[0].read();

				COMPUTE: for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
					auto s_och = s_num_och*c_ops + s_ops;
					t_acc s_acc;
					t_acc s_acc_base;
					t_acc_struct s_acc_struct;
					s_acc_struct.last = s_last & (s_och == (c_och-1));

					if (s_ich == 0)
						s_acc = s_bias[s_ops];
					else
						s_acc = 0;

					s_acc_base = s_acc_buff[s_reuse][s_och];

					for (auto s_index = 0; s_index < c_index; s_index++) {
						s_acc += s_input[s_reuse][s_index] * s_weight[s_ops][s_index];
					}

					if (s_ich != 0)
						s_acc += s_acc_base;

					s_acc_struct.data = s_acc;

					if (s_ich == (c_ich - 1)) {
						o_acc_stream[s_ops].write(s_acc_struct);
					} else {
						s_acc_buff[s_reuse][s_och] = s_acc;
					}
				}
			}
		}
	}

}

template <
	class t_input_struct,
	class t_input,
	class t_weight,
	class t_bias,
	class t_weight_1x1,
	class t_bias_1x1,
	class t_acc_struct,
	class t_acc,
	class t_acc_1x1_struct,
	class t_acc_1x1,
	int c_ich,
	int c_och,
	int c_oh,
	int c_ow,
	int c_index,
	int c_str,
	int c_ops,
	int c_reuse
> void ConvComp(
	hls::stream<t_input_struct> i_input[c_index],
	hls::stream<t_weight> i_weights[c_index],
	hls::stream<t_bias> i_bias[1],
	hls::stream<t_weight_1x1> i_weights_1x1[1],
	hls::stream<t_bias_1x1> i_bias_1x1[1],
	hls::stream<t_acc_struct> o_acc_stream[c_ops],
	hls::stream<t_acc_1x1_struct> o_acc_1x1_stream[c_ops]
) {
/* #pragma HLS inline */

	const auto c_o_index = c_oh*c_ow/c_reuse;
	const auto c_num_och = c_och/c_ops;
	const auto c_iter    = c_reuse*c_num_och;

	t_acc s_acc_buff[c_reuse][c_och];
#pragma HLS array_partition variable=s_acc_buff type=complete dim=0
	t_acc_1x1 s_acc_1x1_buff[c_reuse][c_och];
#pragma HLS array_partition variable=s_acc_1x1_buff type=complete dim=0
	t_input s_input[c_reuse][c_index];
#pragma HLS array_partition variable=s_input type=complete
	bool s_last = false;
	int8_t s_weight[c_ops][c_index];
#pragma HLS array_partition variable=s_weight type=complete
	int8_t s_weight_1x1[c_ops][1];
#pragma HLS array_partition variable=s_weight_1x1 type=complete
	t_bias s_bias;
#pragma HLS array_partition variable=s_bias type=complete
	t_bias_1x1 s_bias_1x1;
#pragma HLS array_partition variable=s_bias_1x1 type=complete

		/* for (auto s_reuse = 0; s_reuse < c_reuse; s_reuse++) */
		/* 	for (auto s_och = 0; s_och < c_och; s_och++) */
		/* 		s_acc_buff[s_reuse][s_och] = 0; */

	for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
		for (auto s_ich = 0; s_ich < c_ich; s_ich++) {
			for (auto s_iter = 0; s_iter < c_iter; s_iter++) {
#pragma HLS pipeline style=flp

				auto s_reuse = s_iter % c_reuse;
				auto s_num_och = s_iter / c_reuse;

				if (s_num_och == 0) {
					for (auto s_index = 0; s_index < c_index; s_index++) {
						t_input_struct s_input_struct = i_input[s_index].read();
						s_input[s_reuse][s_index] = s_input_struct.data;
						/* Sending last only at the bottom right data */
						if (s_index == 0)
							s_last = s_input_struct.last;
					}
				}

				/* Buffering to speed up computations */
				/* TODO: Adjust for generic bit quantizations */
				if (s_reuse == 0) {
					for (auto s_index = 0; s_index < c_index; s_index++) {
						t_weight s_tmp_weight = i_weights[s_index].read();
						for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
							/* Input weights are reversed with respect to original order */
							s_weight[s_ops][s_index] = s_tmp_weight[s_ops];
						}
					}
					t_weight_1x1 s_tmp_weight_1x1 = i_weights_1x1[0].read();
					for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
						/* Input weights are reversed with respect to original order */
						s_weight_1x1[s_ops][0] = s_tmp_weight_1x1[s_ops];
					}
					if (s_ich == 0)
						s_bias = i_bias[0].read();
					if (s_ich == 0)
						s_bias_1x1 = i_bias_1x1[0].read();
				}

				COMPUTE: for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
					auto s_och = s_num_och*c_ops + s_ops;
					t_acc s_acc;
					t_acc s_acc_base;
					t_acc_struct s_acc_struct;
					s_acc_struct.last = s_last & (s_och == (c_och-1));

					t_acc_1x1 s_acc_1x1;
					t_acc_1x1 s_acc_1x1_base;
					t_acc_1x1_struct s_acc_1x1_struct;
					s_acc_1x1_struct.last = s_last & (s_och == (c_och-1));

					if (s_ich == 0)
						s_acc = s_bias[s_ops];
					else
						s_acc = 0;

					s_acc_base = s_acc_buff[s_reuse][s_och];

					for (auto s_index = 0; s_index < c_index; s_index++) {
						s_acc += s_input[s_reuse][s_index] * s_weight[s_ops][s_index];
					}

					if (s_ich == 0)
						s_acc_1x1 = s_bias_1x1[s_ops];
					else
						s_acc_1x1 = 0;

					s_acc_1x1_base = s_acc_1x1_buff[s_reuse][s_och];

					s_acc_1x1 += s_input[s_reuse][c_index/2] * s_weight_1x1[s_ops][0];

					if (s_ich != 0)
						s_acc += s_acc_base;

					s_acc_struct.data = s_acc;

					if (s_ich != 0)
						s_acc_1x1 += s_acc_1x1_base;

					s_acc_1x1_struct.data = s_acc_1x1;

					if (s_ich == (c_ich - 1)) {
						o_acc_stream[s_ops].write(s_acc_struct);
						o_acc_1x1_stream[s_ops].write(s_acc_1x1_struct);
					} else {
						s_acc_buff[s_reuse][s_och] = s_acc;
						s_acc_1x1_buff[s_reuse][s_och] = s_acc_1x1;
					}
				}
			}
		}
	}

}

template <
	class t_input_struct,
	class t_input,
	class t_weight,
	class t_bias,
	class t_acc_struct,
	class t_acc,
	int c_ich,
	int c_och,
	int c_oh,
	int c_ow,
	int c_index,
	int c_str,
	int c_ops,
	int c_reuse,
	int c_shift_h,
	int c_shift_l
> void ConvComp(
	hls::stream<t_input_struct> i_input[c_index],
	hls::stream<t_weight> i_weights[c_index],
	hls::stream<t_bias> i_bias[1],
	hls::stream<t_input_struct> &o_forward,
	hls::stream<t_acc_struct> o_acc_stream[c_ops]
) {
/* #pragma HLS inline */

	const auto c_o_index = c_oh*c_ow/c_reuse;
	const auto c_num_och = c_och/c_ops;
	const auto c_iter    = c_reuse*c_num_och;

	t_acc s_acc_buff[c_reuse][c_och];
#pragma HLS array_partition variable=s_acc_buff type=complete dim=0
	t_input s_input[c_reuse][c_index];
#pragma HLS array_partition variable=s_input type=complete
	bool s_last = false;
	int8_t s_weight[c_ops][c_index];
#pragma HLS array_partition variable=s_weight type=complete
	t_bias s_bias;
#pragma HLS array_partition variable=s_bias type=complete

		/* for (auto s_reuse = 0; s_reuse < c_reuse; s_reuse++) */
		/* 	for (auto s_och = 0; s_och < c_och; s_och++) */
		/* 		s_acc_buff[s_reuse][s_och] = 0; */

	for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
		for (auto s_ich = 0; s_ich < c_ich; s_ich++) {
			for (auto s_iter = 0; s_iter < c_iter; s_iter++) {
#pragma HLS pipeline style=flp

				auto s_reuse = s_iter % c_reuse;
				auto s_num_och = s_iter / c_reuse;

				if (s_num_och == 0) {
					for (auto s_index = 0; s_index < c_index; s_index++) {
						t_input_struct s_input_struct = i_input[s_index].read();
						s_input[s_reuse][s_index] = s_input_struct.data;
						/* Sending last only at the bottom right data */
						if (s_index == 0)
							s_last = s_input_struct.last;
					}
				}

				/* Buffering to speed up computations */
				/* TODO: Adjust for generic bit quantizations */
				if (s_reuse == 0) {
					for (auto s_index = 0; s_index < c_index; s_index++) {
						t_weight s_tmp_weight = i_weights[s_index].read();
						for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
							/* Input weights are reversed with respect to original order */
							s_weight[s_ops][s_index] = s_tmp_weight[s_ops];
						}
					}
					if (s_ich == 0)
						s_bias = i_bias[0].read();
				}

				COMPUTE: for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
					auto s_och = s_num_och*c_ops + s_ops;
					t_acc s_acc;
					t_acc s_acc_base;
					t_acc_struct s_acc_struct;
					s_acc_struct.last = s_last & (s_och == (c_och-1));

					if (s_ich == 0)
						s_acc = s_bias[s_ops];
					else
						s_acc = 0;

					s_acc_base = s_acc_buff[s_reuse][s_och];

					for (auto s_index = 0; s_index < c_index; s_index++) {
						t_input s_data = s_input[s_reuse][s_index];
						if (c_shift_l != 0)
							s_data = QuantAct<t_input,c_shift_l,c_shift_h,t_input>(s_data);
						s_acc += s_data * s_weight[s_ops][s_index];
					}

					if (s_ich != 0)
						s_acc += s_acc_base;

					s_acc_struct.data = s_acc;

					if (s_ich == (c_ich - 1)) {
						o_acc_stream[s_ops].write(s_acc_struct);
					} else {
						s_acc_buff[s_reuse][s_och] = s_acc;
					}
				}
				if (s_num_och == (c_num_och-1)) {
					t_input_struct s_forward;
					s_forward.data = s_input[s_reuse][c_index/2];
					s_forward.last = false;
					o_forward.write(s_forward);
				}
			}
		}
	}

}

//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
/* Version for On-Chip memory weights */

template <
	class t_input_struct,
	class t_input,
	class t_weight,
	class t_bias,
	class t_add_struct,
	class t_acc_struct,
	class t_acc,
	int c_ich,
	int c_och,
	int c_oh,
	int c_ow,
	int c_index,
	int c_str,
	int c_ops,
	int c_reuse,
	int c_shift_l
> void ConvComp(
	hls::stream<t_input_struct> i_input[c_index],
	hls::stream<t_weight> i_weights[c_index],
	hls::stream<t_bias> i_bias[1],
	hls::stream<t_add_struct> &i_add,
	hls::stream<t_acc_struct> o_acc_stream[c_ops]
) {
/* #pragma HLS inline */

	const auto c_o_index = c_oh*c_ow/c_reuse;
	const auto c_num_och = c_och/c_ops;
	const auto c_iter    = c_reuse*c_num_och;

	t_acc s_acc_buff[c_reuse][c_och];
#pragma HLS array_partition variable=s_acc_buff type=complete dim=0
	t_input s_input[c_reuse][c_index];
#pragma HLS array_partition variable=s_input type=complete
	bool s_last = false;
	int8_t s_weight[c_ops][c_index];
#pragma HLS array_partition variable=s_weight type=complete
	t_bias s_bias;
#pragma HLS array_partition variable=s_bias type=complete

		/* for (auto s_reuse = 0; s_reuse < c_reuse; s_reuse++) */
		/* 	for (auto s_och = 0; s_och < c_och; s_och++) */
		/* 		s_acc_buff[s_reuse][s_och] = 0; */

	t_add_struct s_add;
	for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
		for (auto s_ich = 0; s_ich < c_ich; s_ich++) {
			for (auto s_iter = 0; s_iter < c_iter; s_iter++) {
#pragma HLS pipeline style=flp

				auto s_reuse = s_iter % c_reuse;
				auto s_num_och = s_iter / c_reuse;

				if (s_num_och == 0) {
					for (auto s_index = 0; s_index < c_index; s_index++) {
						t_input_struct s_input_struct = i_input[s_index].read();
						s_input[s_reuse][s_index] = s_input_struct.data;
						/* Sending last only at the bottom right data */
						if (s_index == 0)
							s_last = s_input_struct.last;
					}
				}

				/* Buffering to speed up computations */
				/* TODO: Adjust for generic bit quantizations */
				if (s_reuse == 0) {
					for (auto s_index = 0; s_index < c_index; s_index++) {
						t_weight s_tmp_weight = i_weights[s_index].read();
						for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
							/* Input weights are reversed with respect to original order */
							s_weight[s_ops][s_index] = s_tmp_weight[s_ops];
						}
					}
					if (s_ich == 0)
						s_bias = i_bias[0].read();
				}

				// Done to avoid partitioning of the stream and resource wasting
				// Theoretically with the add the input and output dimensions should
				// be equal, so we could perform the addition in different iterations
				if (s_iter == 0)
					s_add = i_add.read();

				COMPUTE: for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
					auto s_och = s_num_och*c_ops + s_ops;
					t_acc s_acc;
					t_acc s_acc_base;
					t_acc_struct s_acc_struct;
					s_acc_struct.last = s_last & (s_och == (c_och-1));

					if (s_ich == 0) { 
						s_acc = s_bias[s_ops];
					}
					else {
						s_acc = 0;
					}

					if (s_ich == s_och) {
						s_acc += (t_acc)(s_add.data) << c_shift_l;
					}

					s_acc_base = s_acc_buff[s_reuse][s_och];

					for (auto s_index = 0; s_index < c_index; s_index++) {
						s_acc += s_input[s_reuse][s_index] * s_weight[s_ops][s_index];
					}

					if (s_ich != 0)
						s_acc += s_acc_base;

					s_acc_struct.data = s_acc;

					if (s_ich == (c_ich - 1)) {
						o_acc_stream[s_ops].write(s_acc_struct);
					} else {
						s_acc_buff[s_reuse][s_och] = s_acc;
					}
				}
			}
		}
	}

}


template <
	class t_input_struct,
	class t_input,
	class t_weight,
	class t_bias,
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
	hls::stream<t_bias> i_bias[1],
	hls::stream<t_acc_struct> o_acc_stream[c_ops]
) {
/* #pragma HLS inline */

	const int c_num_comp = c_ich*c_och;
	const int c_pipe_iter = c_num_comp/c_ops;
	const int c_o_index = c_oh*c_ow*c_pipe_iter;
	const int c_num_och = c_och/c_ops;

	t_acc s_acc_buff[c_och];
#pragma HLS array_partition variable=s_acc_buff type=complete
	t_input s_input[c_index];
#pragma HLS array_partition variable=s_input type=complete
	t_bias s_bias;
#pragma HLS array_partition variable=s_bias type=complete
	bool s_last;

	for (uint32_t s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
#pragma HLS pipeline style=flp

		uint16_t s_pipe_iter = s_o_index % c_pipe_iter; 
		uint8_t s_num_och = s_pipe_iter % c_num_och;
		uint8_t s_ich = s_pipe_iter / c_num_och;

		if (s_pipe_iter == 0) {
			for (uint8_t s_och = 0; s_och < c_och; s_och++)
				s_acc_buff[s_och] = 0;
		}

		if (s_num_och == 0) {
			for (uint8_t s_index = 0; s_index < c_index; s_index++) {
				t_input_struct s_input_struct = i_input[s_index].read();
				s_input[s_index] = s_input_struct.data;
				/* Sending last only at the bottom right data */
				if (s_index == 0)
					s_last = s_input_struct.last;
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

		if (s_ich == 0)
			s_bias = i_bias[0].read();

		COMPUTE: for (uint8_t s_ops = 0; s_ops < c_ops; s_ops++) {
			uint8_t s_och = s_num_och*c_ops + s_ops;
			t_acc s_acc;

			if (s_ich == 0)
				s_acc = s_bias[s_ops];
			else
				s_acc = 0;

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

//////////////////////////////////////////////////////////////////////////////

template <
	class t_output_struct,
	class t_output,
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
	int c_mask,
	int c_shift_h,
	int c_shift_l
> void QuantStream(
	hls::stream<t_acc_struct> i_acc[c_ops],
	int s_ops,
	hls::stream<t_output_struct> &o_data
) {
#pragma HLS inline
	t_acc_struct s_acc_struct = i_acc[s_ops].read();
	t_acc s_acc = c_quant;

	/* 1 subtraction for quantization */
	s_acc += s_acc_struct.data;

	t_output_struct s_output;

	if (c_relu == 1) {
		s_acc = ReluOp<t_acc>(s_acc);
	}
	s_output.data = QuantAct<t_acc,c_shift_l,c_shift_h,c_mask,t_output>(s_acc);
	s_output.last = s_acc_struct.last;

	o_data.write(s_output); 
}

template <
	class t_output_struct,
	class t_output,
	class t_output_1x1_struct,
	class t_output_1x1,
	class t_acc_struct,
	class t_acc,
	class t_acc_1x1_struct,
	class t_acc_1x1,
	int c_ich,
	int c_och,
	int c_oh,
	int c_ow,
	int c_index,
	int c_ops,
	int c_relu,
	int c_quant,
	int c_mask,
	int c_shift_h,
	int c_shift_l,
	int c_mask_1x1,
	int c_shift_h_1x1,
	int c_shift_l_1x1
> void StreamOutput(
	hls::stream<t_acc_struct> i_acc[c_ops],
	hls::stream<t_acc_1x1_struct> i_acc_1x1[c_ops],
	hls::stream<t_output_struct> &o_data,
	hls::stream<t_output_1x1_struct> &o_data_1x1
) {
/* #pragma HLS inline */

	const int c_num_comp = c_oh*c_ow*c_och;
	const int c_pipe_iter = c_num_comp;

	for (auto s_pipe_iter = 0; s_pipe_iter < c_pipe_iter; s_pipe_iter++) {
#pragma HLS pipeline style=frp
    auto s_ops = s_pipe_iter % c_ops;
    QuantStream <
      t_output_struct,
      t_output,
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
      c_mask,
      c_shift_h,
      c_shift_l
    > (
      i_acc,
      s_ops,
      o_data
    );


    QuantStream <
      t_output_1x1_struct,
      t_output_1x1,
      t_acc_1x1_struct,
      t_acc_1x1,
      c_ich,
      c_och,
      c_oh,
      c_ow,
      1,
      1,
      0,
      c_quant,
      c_mask_1x1,
      c_shift_h_1x1,
      c_shift_l_1x1
    > (
      i_acc_1x1,
      s_ops,
      o_data_1x1
    );
	}

}

template <
	class t_output_struct,
	class t_output,
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
	int c_mask,
	int c_shift_h,
	int c_shift_l
> void StreamOutput(
	hls::stream<t_acc_struct> i_acc[c_ops],
	hls::stream<t_output_struct> &o_data
) {
/* #pragma HLS inline */

	const int c_num_comp = c_oh*c_ow*c_och;
	const int c_pipe_iter = c_num_comp;

	for (auto s_pipe_iter = 0; s_pipe_iter < c_pipe_iter; s_pipe_iter++) {
#pragma HLS pipeline style=frp
    auto s_ops = s_pipe_iter % c_ops;
    QuantStream <
      t_output_struct,
      t_output,
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
      c_mask,
      c_shift_h,
      c_shift_l
    > (
      i_acc,
      s_ops,
      o_data
    );
	}

}

#endif
