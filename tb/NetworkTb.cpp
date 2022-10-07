#include "../src/Network.hpp"
#include "hls_stream.h"

int main() {

	hls::stream<t_i_data> i_data;
	/* t_o_data o_data_exp[12]; */
	hls::stream<t_o_data> o_data_sim;
	#pragma HLS interface axis port=i_data
	#pragma HLS interface axis port=o_data_sim

	const int c_index = c_input_ich * c_input_ih * c_input_iw;
	// INIT DATA
	for (int s_index = 0; s_index < c_index; s_index++) {
		t_i_data s_data;
		s_data.data = rand() % 256;

		if (s_index == (c_index - 1))
			s_data.last = true;
		else
			s_data.last = false;
		i_data.write(s_data);
		/* std::cout << i_data[s_index] << "\n"; */
	}

	std::cout << "--------------------- KERNEL -----------------------" << "\n";
	Network(
		i_data,
		o_data_sim
	);

	t_o_data s_o_data;
	do {
		s_o_data = o_data_sim.read();
	} while(!s_o_data.last);
	/* while(o_last == 0); */
	/* while(o_last == 1); */
	
	/* std::cout << "EXP: " << o_data_exp[0] << "\n"; */

	return 0;

}

