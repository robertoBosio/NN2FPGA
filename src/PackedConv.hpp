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
	int c_iw,
	int c_ih,
	int c_ow,
	int c_oh,
	int c_fw,
	int c_fh,
	int c_str,
	int c_pad
> void PackedConvKernel(
	t_input i_data,
	hls::stream<t_weight> &i_weights,
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
			if (s_oh_index >= (c_fh * 2))
				s_oh_index -= (c_fh*2);

			for (int s_fw = c_fw - 1; s_fw > -1; s_fw-=c_str) {
				/* s_ow_index = (s_ow + s_fw - c_owt) % c_fw; */
				s_ow_index = (s_ow + c_fw - s_fw - 1);
				for (int s_och = 0; s_och < c_och; s_och++) {
					s_weight_buffer = i_weights.read();
					if (s_ow_index < (c_ow))
						s_acc_buffer[s_oh_index][s_ow_index][s_och] += i_data * s_weight_buffer;
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
			s_oh_index = (s_oh + s_fh);

			/* Module done comparing against constant */
			/* This works iff s_oh never higher than c_fh*2 */
			if (s_oh_index >= (c_fh * 2))
				s_oh_index -= (c_fh*2);

			for (int s_fw = 0; s_fw < c_fw; s_fw+=c_str) {
				/* s_ow_index = (s_ow + s_fw - c_owt) % c_fw; */
				s_ow_index = s_ow + s_fw;
				for (int s_och = 0; s_och < c_och; s_och++) {
					s_weight_buffer = i_weights.read();
					if (s_ow_index < (c_ow))
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
	int c_pad,
	int c_str
> void PackedConv(
	hls::stream<t_input> &i_data,
	t_weight *i_weights,
	hls::stream<t_output> &o_data
) {

	t_input s_line_buffer[c_ich][c_fh][c_iw];
	/* t_weight s_weight_buffer[c_och][c_ich]; */
	t_acc s_acc_buffer[c_och];

	/* INIT LINE BUFFER */
	for (int s_fh = 0; s_fh < c_fh-1; s_fh++) {
		for (int s_iw = 0; s_iw < c_iw; s_iw++) {
			for (int s_ich = 0; s_ich < c_ich; s_ich++) {
				i_data.read(s_line_buffer[s_ich][s_fh][s_iw]);
			}
		}
	}

	for (int s_oh = 0; s_oh < c_oh; s_oh++) {

		for (int s_ow = 0; s_ow < c_ow; s_ow++) {

			/* INIT ACC BUFFER */
			for (int s_och = 0; s_och < c_och; s_och++) 
				s_acc_buffer[s_och] = 0;

			for (int s_fh = 0; s_fh < c_fh; s_fh++) {

				for (int s_fw = 0; s_fw < c_fw; s_fw++) {

					int s_ih = (s_oh + s_fh) % c_fh;
					int s_iw = s_ow + s_fw;

					ap_uint<1> s_read = (s_fh == (c_fh - 1)) & (s_fw == (c_fw - 1));

					for (int s_ich = 0; s_ich < c_ich; s_ich++) {

						if (s_read) {
							i_data.read(s_line_buffer[s_ich][s_ih][s_iw]);
						}

						for (int s_och = 0; s_och < c_och; s_och++) {

							#pragma HLS bind_op variable=s_acc_buffer op=add impl=dsp //Implement in DSPs
							#pragma HLS bind_op variable=s_acc_buffer op=mul impl=dsp //Implement in DSPs
							int s_windex = s_och * c_ich * c_fw * c_fh + s_ich * c_fw * c_fh + s_fh * c_fw + s_fw;
							s_acc_buffer[s_och] += s_line_buffer[s_ich][s_ih][s_iw] * i_weights[s_windex];

							/* OUTPUT WRITE */
							ap_uint<1> s_wrt = (s_ich == (c_ich - 1)) & (s_fh == (c_fh - 1)) & (s_fw == (c_fw - 1)); 
							if (s_wrt) {
								o_data.write((t_output)(s_acc_buffer[s_och]));
							}

						}

					}

				}

			}

		}

	}

}

/* Config padded version of the packed convolution with Accumulation buffers */
/* What changes is the association between the filters kernel indices and the */ 
/* input features map, this translates in a different initialization of the loops */
/* Version with stream forward for skip connections*/
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
	hls::stream<t_input> &o_forward,
	ap_uint<1> *i_last,
	ap_uint<1> *o_last_forward,
	ap_uint<1> *o_last
) {

	if (i_weights.empty())
		return;

	if (i_data.empty())
		return;

	t_acc s_acc_buffer[c_fh * 2][c_ow][c_och] = {0};

	/* Computing offsets for padding */
	const int c_index_off_w = (c_ow % c_fw) * c_och;
	const int c_index_off_h = (c_oh % c_fh) * c_ow * c_och;
	const int c_index_off = c_index_off_w + c_index_off_h;
	const int c_acc_index = c_och*c_fh*c_fw;
	const int c_oht = c_oh % c_fh;
	const int c_owt = c_ow % c_fw;
	const int c_index = c_ih*c_iw;
	uint16_t s_oh, s_ow;
	uint16_t s_oh_index_w;

	s_oh = 0;
	s_ow = 0;

	for (int s_index = 0; s_index < c_index; s_index+=c_str) {

		for (uint8_t s_str = 0; s_str < c_str; s_str++) {

			/* At each cycle a value is read and computed everywhere it is needed */
			/* All the weights must be read accordingly */
			t_input s_in_buffer;

			/* FIX INDEXING OFFSET */
			/* IT COULD BE ALREADY CORRECT */
			/* Padded computation, padding is done through loop iterations management */
			for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
				s_in_buffer = i_data.read(); 

				 PackedConvKernel<
						t_input,
						t_weight,
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
						i_weights,
						s_oh,
						s_ow,
						s_acc_buffer
					);


				/* This is useful for skip connection propagation, the stream can be read and */
				/* written just once, this allows to propagate the stream towards the next block */
				o_forward.write(s_in_buffer);

			}

		}

		WriteOutputKernel<
			t_acc,
			t_output,
			c_och,
			c_ow,
			c_fh,
			c_pad
		> (
			s_oh,
			s_ow,
			s_acc_buffer,
			o_data
		);

		/* If the line is over then the new one starts */
		if (s_ow == (c_ow - 1)) {
			s_oh++;
			if (s_oh >= (c_fh * 2))
				s_oh = 0;
			s_ow = 0;
		} else {
			s_ow++;
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
	int c_str,
	int c_pad
> void PackedConvBuffAcc(
	hls::stream<t_input> &i_data,
	hls::stream<t_weight> &i_weights,
	hls::stream<t_output> &o_data,
	ap_uint<1> *i_last,
	ap_uint<1> *o_last
) {

	if (i_weights.empty())
		return;

	if (i_data.empty())
		return;

	t_acc s_acc_buffer[c_fh * 2][c_ow][c_och] = {0};

	/* Computing offsets for padding */
	const int c_index_off_w = (c_ow % c_fw) * c_och;
	const int c_index_off_h = (c_oh % c_fh) * c_ow * c_och;
	const int c_index_off = c_index_off_w + c_index_off_h;
	const int c_acc_index = c_och*c_fh*c_fw;
	const int c_oht = c_oh % c_fh;
	const int c_owt = c_ow % c_fw;
	const int c_index = c_ih*c_iw;
	uint16_t s_oh, s_ow;
	uint16_t s_oh_index_w;

	s_oh = 0;
	s_ow = 0;

	for (int s_index = 0; s_index < c_index; s_index+=c_str) {

		for (uint8_t s_str = 0; s_str < c_str; s_str++) {

			/* At each cycle a value is read and computed everywhere it is needed */
			/* All the weights must be read accordingly */
			t_input s_in_buffer;

			for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
				s_in_buffer = i_data.read(); 

				 PackedConvKernel<
						t_input,
						t_weight,
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
						i_weights,
						s_oh,
						s_ow,
						s_acc_buffer
					);

			}

		}

		WriteOutputKernel<
			t_acc,
			t_output,
			c_och,
			c_ow,
			c_fh,
			c_pad
		> (
			s_oh,
			s_ow,
			s_acc_buffer,
			o_data
		);

		/* If the line is over then the new one starts */
		if (s_ow == (c_ow - 1)) {
			s_oh++;
			if (s_oh >= (c_fh * 2))
				s_oh = 0;
			s_ow = 0;
		} else {
			s_ow++;
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
	hls::stream<t_output> &o_data,
	ap_uint<1> *i_last,
	ap_uint<1> *o_last
) {

	if (i_weights.empty())
		return;

	if (i_data.empty())
		return;

	t_acc s_acc_buffer[c_fh * 2][c_ow][c_och] = {0};

	/* Computing offsets for padding */
	const int c_index_off_w = (c_ow % c_fw) * c_och;
	const int c_index_off_h = (c_oh % c_fh) * c_ow * c_och;
	const int c_index_off = c_index_off_w + c_index_off_h;
	const int c_acc_index = c_och*c_fh*c_fw;
	const int c_oht = c_oh % c_fh;
	const int c_owt = c_ow % c_fw;
	const int c_index = c_ih*c_iw;
	uint16_t s_oh, s_ow;
	uint16_t s_oh_index_w;
	bool s_start = true;

	s_oh = 0;
	s_ow = 0;

	for (int s_index = 0; s_index < c_index; s_index+=c_str) {

		for (uint8_t s_str = 0; s_str < c_str; s_str++) {

			/* At each cycle a value is read and computed everywhere it is needed */
			/* All the weights must be read accordingly */
			t_input s_in_buffer;

			for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
				s_in_buffer = i_data.read(); 

				 PackedConvKernel<
						t_input,
						t_weight,
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
						i_weights,
						s_oh,
						s_ow,
						s_acc_buffer
					);

			}

			/* If the filter indices are both 0 it means that the values accumulated on the */
			/* buffer are ready to be written out as soon as the input channel values are */
			/* all considered */
			/* s_oh_index_w = s_oh % c_fh; */
			s_oh_index_w = s_oh;
			for (uint8_t s_och = 0; s_och < c_och; s_och++) {

				/* Module done comparing against constant */
				/* This works iff s_oh never higher than c_fh*2 */
				if (s_oh_index_w >= (c_fh << 1))
					s_oh_index_w = 0;

				/* BIAS ADDITION*/
				t_bias s_bias;
				i_bias.read(s_bias);
				s_acc_buffer[s_oh_index_w][s_ow][s_och] += s_bias;

			}

		}

		WriteOutputKernel<
			t_acc,
			t_output,
			c_och,
			c_ow,
			c_fh,
			c_pad
		> (
			s_oh,
			s_ow,
			s_acc_buffer,
			o_data
		);

		/* If the line is over then the new one starts */
		if (s_ow == (c_ow - 1)) {
			s_oh++;
			if (s_oh >= (c_fh << 1))
				s_oh = 0;
			s_ow = 0;
		} else {
			s_ow++;
		}

	}

}

#endif
