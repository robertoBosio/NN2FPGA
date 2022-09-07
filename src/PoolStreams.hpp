#ifndef __POOLSTREAM__
#define __POOLSTREAM__

#include "ap_int.h"
#include "hls_stream.h"

template <
	class t_input,
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
> void AveragePoolKernel(
	t_input i_data,
	int s_oh,
	int s_ow,
	t_acc s_acc_buffer[c_fh*2][c_ow][c_och]
) {

	t_weight s_weight_buffer;
	int s_ow_index;
	int s_oh_index;

	if (c_pad == 1) {
		/* FORMULA IS S_OH_INDEX = (S_OH + C_FH - S_FH - C_OHT) */
		for (int s_fh = c_fh - 1; s_fh > -1; s_fh-=c_str) {
			/* s_oh_index = (s_oh + s_fh - c_oht) % c_fh; */
			s_oh_index = (s_oh + c_fh - s_fh - 1);

			/* Module done comparing against constant */
			/* This works iff s_oh never higher than c_fh*2 */
			if (s_oh_index >= (c_fh << 1))
				s_oh_index -= (c_fh << 1);

			for (int s_fw = c_fw - 1; s_fw > -1; s_fw-=c_str) {
				/* s_ow_index = (s_ow + s_fw - c_owt) % c_fw; */
				s_ow_index = (s_ow + c_fw - s_fw - 1);
				for (int s_och = 0; s_och < c_och; s_och++) {
					if (s_ow_index < (c_ow))
						s_acc_buffer[s_oh_index][s_ow_index][s_och] += i_data;
#ifndef __SYNTHESIS__
					/* std::cout << s_oh_index << " " << s_ow_index << "\n"; */
					/* std::cout << (unsigned int)(s_in_buffer & 0xff) << " " << (unsigned int)(s_weight_buffer & 0xff) << "\n"; */
#endif
				}
			}
		}
	}

	/* Not padded computation */
	if (c_pad == 0) {
		/* FORMULA IS S_OH_INDEX = (S_OH + C_FH - S_FH - C_OHT) */
		for (int s_fh = 0; s_fh < c_fh; s_fh+=c_str) {
			/* s_oh_index = (s_oh + s_fh - c_oht) % c_fh; */
			s_oh_index = (s_oh - s_fh);

			/* Module done comparing against constant */
			/* This works iff s_oh never higher than c_fh*2 */
			if (s_oh_index < 0)
				s_oh_index += (c_fh << 1);

			for (int s_fw = 0; s_fw < c_fw; s_fw+=c_str) {
				/* s_ow_index = (s_ow + s_fw - c_owt) % c_fw; */
				s_ow_index = s_ow - s_fw;
				for (int s_och = 0; s_och < c_och; s_och++) {
					if ((s_ow_index > -1) & (s_oh_index > -1))
						s_acc_buffer[s_oh_index][s_ow_index][s_och] += i_data * s_weight_buffer;
#ifndef __SYNTHESIS__
					/* std::cout << s_oh_index << " " << s_ow_index << "\n"; */
					/* std::cout << (unsigned int)(s_in_buffer & 0xff) << " " << (unsigned int)(s_weight_buffer & 0xff) << "\n"; */
#endif
				}
			}
		}
	}

}


/* Not padded version of the packed average pooling with Accumulation buffers */
/* What changes is the association between the average window indices and the */ 
/* input features map, this translates in a different initialization of the loops */
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
	hls::stream<t_output> &o_data,
	ap_uint<1> *i_last,
	ap_uint<1> *o_last
) {

	if (i_data.empty())
		return;

	t_acc s_acc_buffer[c_fh * 2][c_ow][c_och] = {0};

	/* Computing offsets for padding */
	const int c_average = c_fh * c_fw;
	const int c_pad_index_oh = (c_fh - 1) * (1 - c_pad);
	const int c_pad_index_ow = c_iw * (1 - c_pad);
	int s_oh, s_ow;
	int s_oh_write, s_ow_write;

	s_oh = 0;
	s_ow = 0;
	s_oh_write = 0;
	s_ow_write = 0;

	/* TODO: Check array indices */ 
	/* Initializing accumulator in case of no padding, handles by c_pad_index */
	for (uint8_t s_ih = 0; s_ih < c_pad_index_oh; s_ih+=c_str) {

		for (uint8_t s_str_h = 0; s_str_h < c_str; s_str_h++) {

			for (uint8_t s_iw = 0; s_iw < c_pad_index_ow; s_iw+=c_str) {

				for (uint8_t s_str_w = 0; s_str_w < c_str; s_str_w++) {

					t_input s_in_buffer;

					s_in_buffer = i_data.read(); 

					AveragePoolKernel<
						t_input,
						t_acc,
						c_ich,
						c_och,
						c_iw,
						c_ih,
						c_ow,
						c_oh,
						c_fw,
						c_fh,
						c_str,
						c_pad
					> (
						s_in_buffer,
						s_oh + s_str_h,
						s_ow + s_str_w,
						s_acc_buffer
					);

				}

			}

			/* If the line is over then the new one starts */
			if (s_ow == (c_ow - 1)) {
				s_oh++;
				s_ow = 0;
			} else {
				s_ow++;
			}

		}

	}

	for (uint8_t s_ih = c_pad_index_oh; s_ih < c_ih; s_ih+=c_str) {

		for (uint8_t s_str_h = 0; s_str_h < c_str; s_str_h++) {

			for (uint8_t s_iw = 0; s_iw < c_pad_index_ow - 1; s_iw+=c_str) {

				for (uint8_t s_str_w = 0; s_str_w < c_str; s_str_w++) {

					t_input s_in_buffer;

					s_in_buffer = i_data.read(); 

					AveragePoolKernel<
						t_input,
						t_acc,
						c_ich,
						c_och,
						c_iw,
						c_ih,
						c_ow,
						c_oh,
						c_fw,
						c_fh,
						c_str,
						c_pad
					> (
						s_in_buffer,
						s_oh + s_str_h,
						s_ow + s_str_w,
						s_acc_buffer
					);

				}

				/* If the line is over then the new one starts */
				if (s_ow == (c_ow - 1)) {
					s_oh++;
					s_ow = 0;
				} else {
					s_ow++;
				}

			}

			for (uint8_t s_iw = c_pad_index_ow - 1; s_iw < c_iw; s_iw+=c_str) {

				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {

					for (uint8_t s_str_w = 0; s_str_w < c_str; s_str_w++) {

						/* At each cycle a value is read and computed everywhere it is needed */
						/* All the weights must be read accordingly */
						t_input s_in_buffer;

						s_in_buffer = i_data.read(); 

						AveragePoolKernel<
							t_input,
							t_acc,
							c_ich,
							c_och,
							c_iw,
							c_ih,
							c_ow,
							c_oh,
							c_fw,
							c_fh,
							c_str,
							c_pad
						> (
							s_in_buffer,
							s_oh + s_str_h,
							s_ow + s_str_w,
							s_acc_buffer
						);

					}

				}


				/* Apply division for average */

				for (int s_och = 0; s_och < c_och; s_och++) {

					s_acc_buffer[s_oh_write][s_ow_write][s_och] /= c_average;

				}

				WriteOutputKernel<
					t_acc,
					t_output,
					c_och,
					c_ow,
					c_fh,
					c_pad
				> (
					s_oh_write,
					s_ow_write,
					s_acc_buffer,
					o_data
				);


				/* If the line is over then the new one starts */
				if (s_ow == (c_ow - 1)) {
					s_oh++;
					if (s_oh >= (c_fh))
						s_oh = 0;
					s_ow = 0;
				} else {
					s_ow++;
				}

				/* If the line is over then the new one starts */
				if (s_ow_write == (c_ow - 1)) {
					s_oh_write++;
					if (s_oh_write >= (c_fh))
						s_oh_write = 0;
					s_ow_write = 0;
				} else {
					s_ow_write++;
				}
				
			}

		}

	}

}

#endif

