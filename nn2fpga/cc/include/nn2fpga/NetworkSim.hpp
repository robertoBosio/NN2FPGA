#ifndef __NETWORKSIM__
#define __NETWORKSIM__
#include "Network.hpp"

void NetworkSim(
	hls::stream<t_inp_1> &inp_1,
	hls::stream<t_o_outp1> &o_outp1
) {
	Network(
		inp_1,
		o_outp1
	);
}

#endif