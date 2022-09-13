#ifndef __ACTIVATIONSTREAM__
#define __ACTIVATIONSTREAM__

#include "ap_int.h"
#include "hls_stream.h"
template <
	class t_input
> t_input ReluOp(
	t_input i_data
) {
#pragma HLS inline
	if (i_data > 0)
		return i_data
	else
		return 0	
}

template <
	class t_input,
	class t_output,
	const int c_ich,
	const int c_ih,
	const int c_iw
> void ReluStreams(
	hls::stream<t_input> &i_data,
	hls::stream<t_output> &o_data
) {

	if (i_data.empty())
		return;

	t_input s_data;
	t_output s_o_data; 

	const int c_index = c_ich*c_ih*c_iw;

	for (int s_index = 0; s_index < c_index; s_index++) {

		i_data.read(s_data);

		if (s_data > 0)
			s_o_data = s_data;
		else
			s_o_data = 0;

		o_data.write(s_o_data);

	}

}

#endif
