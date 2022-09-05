#ifndef __ACTIVATIONSTREAM__
#define __ACTIVATIONSTREAM__

#include "ap_int.h"
#include "hls_stream.h"

template <
	class t_input,
	class t_output
> void ReluStreams(
	hls::stream<t_input> &i_data,
	hls::stream<t_output> &o_data,
	ap_uint<1> i_last[1],
	ap_uint<1> o_last[1]
) {

	t_input s_data;
	t_output s_o_data; 

	i_data.read(s_data);

	if (s_data > 0)
		s_o_data = s_data;
	else
		s_o_data = 0;

	o_data.write(s_o_data);

	o_last[0] = i_last[0];

}

#endif
