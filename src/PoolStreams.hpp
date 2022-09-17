#ifndef __POOLSTREAM__
#define __POOLSTREAM__

#include "ap_int.h"
#include "hls_stream.h"

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

	if (i_data.empty())
		return;

	const int c_ih_padded = c_ih + 2*(1 - c_pad)*(c_ih % c_fh);
	const int c_ih_start  = (1 - c_pad)*(c_ih % c_fh);
	const int c_ih_loop_start = (c_pad)*(c_ih % c_fh) + c_ih_start;
	const int c_ih_loop_end = (c_pad)*(c_ih % c_fh) + c_ih;

	const int c_iw_padded = c_iw + 2*c_pad*(c_iw % c_fw);
	const int c_iw_start  = (1 - c_pad)*(c_iw % c_fw);
	const int c_iw_loop_start = (c_pad)*(c_iw % c_fw) + c_iw_start;
	const int c_iw_loop_end = (c_pad)*(c_iw % c_fw) + c_iw;

	const int c_average = c_fh*c_fw;

	t_input s_data_buffer[c_ich][c_fh][c_iw_padded] = {0};

	if (c_pad == 0) {
		/* Filling the buffers for padding case */
		for (uint8_t s_fh = 0; s_fh < (c_fh - 1); s_fh++) {
			for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					s_data_buffer[s_ich][s_fh][s_iw] = i_data.read();
				}
			}
		}
	}

	for (uint8_t s_ih = c_ih_loop_start; s_ih < c_ih_loop_end; s_ih+=c_str) {

		if (c_pad == 0) {
			/* Filling the buffers for padding case along w axes */
			for (uint8_t s_iw = 0; s_iw < c_fw; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					s_data_buffer[s_ich][s_ih % c_fh][s_iw] = i_data.read();
				}
			}
		}

		for (uint8_t s_iw = c_iw_loop_start; s_iw < c_iw_loop_end; s_iw+=c_str) {

			for (uint8_t s_str = 0; s_str < c_str; s_str++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					s_data_buffer[s_ich][s_ih % c_fh][s_iw + s_str] = i_data.read();
				}
			}

			for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
				t_acc s_acc = 0;

				for (uint8_t s_fh = 0; s_fh < c_fh; s_fh++) {
					for (uint8_t s_fw =0; s_fw < c_fw; s_fw++) {
						t_input s_data = s_data_buffer[s_ich][(s_ih + s_fh) % c_fh][s_iw + s_fw];
						s_acc += s_data;
					}
				}
				s_acc /= c_average;
				o_data.write((t_output)(s_acc));
			}
		}

		/* This loop fills the new line of the buffer if stride is present */
		for (uint8_t s_str = 0; s_str < (c_str - 1); s_str++) {
			/* Filling the buffers for padding case along w axes */
			for (uint8_t s_iw = c_iw_loop_start; s_iw < c_iw_loop_end; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					s_data_buffer[s_ich][(s_ih + s_str) % c_fh][s_iw] = i_data.read();
				}
			}
		}

	}


}

#endif

