#ifndef __POOLSTREAM__
#define __POOLSTREAM__

#include "ap_int.h"
#include "hls_stream.h"

/* Not padded version of the packed average pooling with Accumulation buffers */
/* What changes is the association between the average window indices and the */ 
/* input features map, this translates in a different initialization of the loops */
template <
	class t_input,
	class t_output,
	class t_acc,
	int c_och,
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
	const int c_index_off_w = (c_ow % c_fw) * c_och;
	const int c_index_off_h = (c_oh % c_fh) * c_ow * c_och;
	const int c_index_off = c_index_off_w + c_index_off_h;
	const int c_acc_index = c_och*c_fh*c_fw;
	const int c_oht = c_oh % c_fh;
	const int c_owt = c_ow % c_fw;
	const int c_average = c_fh * c_fw;
	const int c_index = c_och*c_oh*c_ow;
	int s_oh, s_ow;
	int s_oh_index;
	int s_oh_index_w;
	int s_ow_index;

	s_oh = 0;
	s_ow = 0;

	for (int s_index = 0; s_index < c_index; s_index++) {

		/* At each cycle a value is read and computed everywhere it is needed */
		/* All the weights must be read accordingly */
		t_input s_in_buffer;

		s_in_buffer = i_data.read(); 
		if (c_pad == 1) {
			/* FORMULA IS S_OH_INDEX = (S_OH + C_FH - S_FH - C_OHT) */
			for (int s_fh = c_fh - 1; s_fh > -1; s_fh-=c_str) {
				/* s_oh_index = (s_oh + s_fh - c_oht) % c_fh; */
				s_oh_index = (s_oh + c_fh - s_fh - 1);

				/* Module done comparing against constant */
				/* This works iff s_oh never higher than c_fh*2 */
				if (s_oh_index >= (c_fh))
					s_oh_index -= (c_fh);

				for (int s_fw = c_fw - 1; s_fw > -1; s_fw-=c_str) {
					s_ow_index = (s_ow + c_fw - s_fw - 1);
					for (int s_och = 0; s_och < c_och; s_och++) {
						if (s_ow_index < (c_ow))
							s_acc_buffer[s_oh_index][s_ow_index][s_och] += s_in_buffer;
#ifndef __SYNTHESIS__
						/* std::cout << s_oh_index << " " << s_ow_index << "\n"; */
						/* std::cout << (unsigned int)(s_in_buffer & 0xff) << "\n"; */
#endif
					}
				}
			}
		}

		if (c_pad == 0) {
			for (int s_fh = 0; s_fh < c_fh; s_fh+=c_str) {
				s_oh_index = (s_oh + s_fh);

				/* Module done comparing against constant */
				/* This works iff s_oh never higher than c_fh*2 */
				if (s_oh_index >= (c_fh))
					s_oh_index -= (c_fh);

				for (int s_fw = 0; s_fw < c_fw; s_fw+=c_str) {
					s_ow_index = s_ow + s_fw;
					for (int s_och = 0; s_och < c_och; s_och++) {
						if (s_ow_index < (c_ow))
							s_acc_buffer[s_oh_index][s_ow_index][s_och] += s_in_buffer;
#ifndef __SYNTHESIS__
						/* std::cout << s_oh_index << " " << s_ow_index << "\n"; */
						/* std::cout << (unsigned int)(s_in_buffer & 0xff) << "\n"; */
#endif
					}
				}
			}
		}

		s_oh_index_w = s_oh;
		for (int s_och = 0; s_och < c_och; s_och++) {

			/* Module done comparing against constant */
			/* This works iff s_oh never higher than c_fh*2 */
			if (s_oh_index_w >= (c_fh))
				s_oh_index_w -= (c_fh);

			t_output s_out_buffer = (t_output)(s_acc_buffer[s_oh_index_w][s_ow][s_och]);

			/* Apply division for average */

			s_out_buffer /= c_average;

			o_data.write(s_out_buffer);
			s_acc_buffer[s_oh_index_w][s_ow][s_och] = 0;
		}

		/* If the line is over then the new one starts */
		if (s_ow == (c_ow - 1)) {
			s_oh++;
			if (s_oh >= (c_fh))
				s_oh = 0;
			s_ow = 0;
		} else {
			s_ow++;
		}

	}

}

#endif

