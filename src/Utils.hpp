#ifndef __UTILS__
#define __UTILS__

//////////////////////////// FROM POINTER TO STREAM /////////////////////////// 
// For input activations
template <
	class t_input,
	class t_output,
	int c_ich,
	int c_iw,
	int c_ih
> void ProduceStream(
	t_input *i_data,
	hls::stream<t_output> &s_i_data,
	ap_uint<1> *i_last
) {

	i_last[0] = 0;

	int s_index = 0;

	PRODSTR: for (int s_ih = 0; s_ih < c_ih; s_ih++) {
		for (int s_iw = 0; s_iw < c_iw; s_iw++) {
			for (int s_ich = 0; s_ich < c_ich; s_ich++) {
				s_i_data.write((t_output)(i_data[s_index]));
				s_index++;
			}
		}
	}

	i_last[0] = 1;

}

// For input weights
template <
	class t_input,
	class t_output,
	int c_ich,
	int c_och,
	int c_iw,
	int c_ih,
	int c_ow,
	int c_oh,
	int c_str
> void ProduceStream(
	t_input *i_data,
	hls::stream<t_output> &s_i_data,
	ap_uint<1> *i_last
) {

	for (int s_oh = 0; s_oh < c_oh; s_oh++) {
		for (int s_ow = 0; s_ow < c_ow; s_ow++) {

			int s_index = 0;

			for (int s_ich = 0; s_ich < c_ich; s_ich++) {
				for (int s_ih = 0; s_ih < c_ih; s_ih+=c_str) {
					for (int s_iw = 0; s_iw < c_iw; s_iw+=c_str) {
						for (int s_och = 0; s_och < c_och; s_och++) {
							s_i_data.write((t_output)(i_data[s_index]));
							s_index++;
						}
					}
				}
			}
		}
	}

}

// For input weights
template <
	class t_input,
	class t_output,
	int c_ich,
	int c_och,
	int c_iw,
	int c_ih,
	int c_ow,
	int c_oh,
	int c_str
> void ProduceStream(
	t_input *i_data,
	hls::stream<t_output> &s_i_data
) {

	for (int s_oh = 0; s_oh < c_oh; s_oh++) {
		for (int s_ow = 0; s_ow < c_ow; s_ow++) {

			int s_index = 0;

			for (int s_ich = 0; s_ich < c_ich; s_ich++) {
				for (int s_ih = 0; s_ih < c_ih; s_ih+=c_str) {
					for (int s_iw = 0; s_iw < c_iw; s_iw+=c_str) {
						for (int s_och = 0; s_och < c_och; s_och++) {
							s_i_data.write((t_output)(i_data[s_index]));
							s_index++;
						}
					}
				}
			}
		}
	}

}

// For input weights
template <
	class t_input,
	class t_output,
	int c_ich,
	int c_och,
	int c_iw,
	int c_ih,
	int c_ow,
	int c_oh,
	int c_str
> void ProduceStream(
	const t_input i_data[c_och*c_ich*c_iw*c_ih],
	hls::stream<t_output> &s_i_data
) {

	for (int s_oh = 0; s_oh < c_oh; s_oh++) {
		for (int s_ow = 0; s_ow < c_ow; s_ow++) {

			int s_index = 0;

			for (int s_ich = 0; s_ich < c_ich; s_ich++) {
				for (int s_ih = 0; s_ih < c_ih; s_ih+=c_str) {
					for (int s_iw = 0; s_iw < c_iw; s_iw+=c_str) {
						for (int s_och = 0; s_och < c_och; s_och++) {
							s_i_data.write((t_output)(i_data[s_index]));
							s_index++;
						}
					}
				}
			}
		}
	}

}

///////////////////////////// FROM STREAM TO POINTER ////////////////////////// 

// For output activations
template <
	class t_input,
	class t_output,
	int c_och,
	int c_ow,
	int c_oh
> void ConsumeStream(
	hls::stream<t_input> &i_data,
	t_output *o_data,
	ap_uint<1> *i_last
) {

	if (i_data.empty())
		return;

	t_input s_read;
	bool s_start = true;

	const int c_index = c_och*c_oh*c_ow;

	for (int s_index = 0; s_index < c_index; s_index++) {

		s_read = i_data.read();
		o_data[s_index] = (t_output)(s_read);

	}

	if (i_last[0] == 1) {
		s_start = true;
		i_last[0] = 0;
	} else {
		s_start = false;
	}

}

///////////////////////////// FROM STREAM TO STREAM ///////////////////////////

template <
	class t_input,
	class t_output,
	int c_ich,
	int c_och,
	int c_iw,
	int c_ih,
	int c_ow,
	int c_oh,
	int c_pad
> void PadStream(
	hls::stream<t_input> &i_data,
	hls::stream<t_output> &o_data
) {

	if (c_pad == 0) {
		for (uint8_t s_ih = 0; s_ih < c_ih; s_ih++) {
			for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
						o_data.write((t_output)(i_data.read()));
				}
			}
		}
	}

	if (c_pad == 1) {
		for (uint8_t s_ih = 0; s_ih < c_ih; s_ih++) {
			for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
						o_data.write((t_output)(i_data.read()));
				}
			}
		}
	}

}

//////////////////////////// BLOCK INTERFACES /////////////////////////////////

template <
	class t_acc,
	class t_output,
	int c_och,
	int c_ow,
	int c_fh,
	int c_pad
> void WriteOutputKernel(
	uint16_t s_oh,
	uint16_t s_ow,
	t_acc s_acc_buffer[c_fh*2][c_ow][c_och],
	hls::stream<t_output> &o_data
) {

	for (uint8_t s_och = 0; s_och < c_och; s_och++) {

			t_output s_out_buffer = (t_output)(s_acc_buffer[s_oh][s_ow][s_och]);
			o_data.write(s_out_buffer);
			s_acc_buffer[s_oh][s_ow][s_och] = 0;

	}

}

#endif
