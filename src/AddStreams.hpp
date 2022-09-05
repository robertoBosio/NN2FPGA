#ifndef __ADDSTREAM__
#define __ADDSTREAM__

#include "ap_int.h"
#include "hls_stream.h"

template <
	class t_input,
	class t_output
> void AddStreams(
	hls::stream<t_input> &i_data1,
	hls::stream<t_input> &i_data2,
	hls::stream<t_output> &o_data,
	ap_uint<1> i_last[1],
	ap_uint<1> o_last[1]
) {

	t_input s_data1, s_data2;
	t_output s_o_data; 

	i_data1.read(s_data1);
	i_data2.read(s_data2);

	s_o_data = s_data1 + s_data2;

	o_data.write(s_o_data);

	o_last[0] = i_last[0];

}

#endif
