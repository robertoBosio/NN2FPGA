#ifndef __UTILS__
#define __UTILS__

///////////////////////////// FROM POINTER TO STREAM /////////////////////////// 
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

template <
	class t_input,
	class t_output,
	int c_och,
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

	PRODSTR: for (int s_ich = 0; s_ich < c_ich; s_ich++) {
		for (int s_ih = 0; s_ih < c_ih; s_ih++) {
			for (int s_iw = 0; s_iw < c_iw; s_iw++) {
				for (int s_och = 0; s_och < c_och; s_och++) {
					s_i_data.write((t_output)(i_data[s_index]));
					s_index++;
				}
			}
		}
	}

	i_last[0] = 1;

}

template <
	class t_input,
	class t_output,
	int c_och,
	int c_ich,
	int c_iw,
	int c_ih
> void ProduceStream(
	t_input *i_data,
	hls::stream<t_output> &s_i_data
) {

	int s_index = 0;

	PRODSTR: for (int s_ich = 0; s_ich < c_ich; s_ich++) {
		for (int s_ih = 0; s_ih < c_ih; s_ih++) {
			for (int s_iw = 0; s_iw < c_iw; s_iw++) {
				for (int s_och = 0; s_och < c_och; s_och++) {
					s_i_data.write((t_output)(i_data[s_index]));
					s_index++;
				}
			}
		}
	}

}

template <
	class t_input,
	class t_output,
	int c_och,
	int c_ich,
	int c_iw,
	int c_ih
> void ProduceStream(
	const t_input i_data[c_och*c_ich*c_iw*c_ih],
	hls::stream<t_output> &s_i_data
) {

	int s_index = 0;

	PRODSTR: for (int s_ich = 0; s_ich < c_ich; s_ich++) {
		for (int s_ih = 0; s_ih < c_ih; s_ih++) {
			for (int s_iw = 0; s_iw < c_iw; s_iw++) {
				for (int s_och = 0; s_och < c_och; s_och++) {
					s_i_data.write((t_output)(i_data[s_index]));
					s_index++;
				}
			}
		}
	}

}

///////////////////////////// FROM STREAM TO POINTER /////////////////////////// 

template <
	class t_input,
	class t_output,
	int c_och,
	int c_ow,
	int c_oh
> void ConsumeStream(
	hls::stream<t_input> &s_i_data,
	t_output *o_data,
	ap_uint<1> *i_last
) {

	int s_index = 0;
	t_input s_read;
	bool s_start = true;

	if (s_start) {
		s_index = 0;
	}

	s_read = s_i_data.read();
	o_data[s_index] = (t_output)(s_read);
	s_index++;

	if (i_last[0] == 1) {
		s_start = true;
		i_last[0] = 0;
	} else {
		s_start = false;
	}

}

#endif
