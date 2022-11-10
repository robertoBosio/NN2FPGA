#include "../src/Network.hpp"
#include "../cifar-10/include/cifar/cifar10_reader.hpp"
#include "hls_stream.h"
#include <unistd.h>
char *getcwd(char *buf, size_t size);

int main() {

	hls::stream<t_i_data> i_data;
	/* t_o_data o_data_exp[12]; */
	hls::stream<t_o_data> o_data_sim;
	#pragma HLS interface axis port=i_data
	#pragma HLS interface axis port=o_data_sim

	const int c_par = c_i_data/8;
	const int c_index = (c_input_ich*c_input_ih*c_input_iw)/c_par;
	const int c_labels = 10;

	char cwd[100];
	std::cout << "CURRENT WORKING DIRECTORY" << std::endl;
	std::cout << getcwd(cwd, sizeof(cwd)) << std::endl;
  auto dataset = cifar::read_dataset<std::vector, std::vector, char, char>();

	/* for (int i = 0; i < dataset.test_images.size(); i++) { */
	/* 	std::cout << dataset.test_images.at(i) << ' '; */
	/* } */
	/* const int c_batch = dataset.test_images.size(); */
	const int c_batch = 1;
	const int n_bytes = c_batch*c_index*c_par;
	std::cout << "SENDING " << c_batch << " IMAGES" << std::endl;
	std::cout << "SENDING " << n_bytes << " BYTES" << std::endl;
	int s_index = 0; 

	for (auto it = dataset.test_images.begin(); it != dataset.test_images.end(); ++it) {
		for (auto itt = it->begin(); itt != it->end(); ++itt) {
			t_i_data s_data;
			int s_par = (s_index % c_par);
			s_data.data(8*(s_par+1)-1,8*s_par) = (ap_uint<8>)(*itt);

			std::cout << (ap_uint<8>)(*itt) << " ";

			if (s_index == (n_bytes-1))
				s_data.last = true;
			else
				s_data.last = false;
			if (s_par == (c_par-1))
				i_data.write(s_data);
			if (s_index == (n_bytes-1))
				break;
			s_index++;
		}
		if (s_index == (n_bytes-1))
			break;
	}

	std::cout << std::endl;
	// INIT DATA

	std::cout << "--------------------- KERNEL -----------------------" << "\n";
	Network(
		i_data,
		o_data_sim
	);

	t_o_data s_o_data;
	do {
		s_o_data = o_data_sim.read();
		std::cout << "COMPUTED LABEL " << s_o_data.data << std::endl;
	} while(!s_o_data.last);

	const int n_bytes_labels = c_batch;
	s_index = 0;
	for (auto it = dataset.test_labels.begin(); it != dataset.test_labels.end(); ++it) {
		std::cout << "EXPECTED LABEL " << (int)*it << std::endl;
		if (s_index == (n_bytes_labels-1))
			break;
		s_index++;
	}
	/* while(o_last == 0); */
	/* while(o_last == 1); */
	
	/* std::cout << "EXP: " << o_data_exp[0] << "\n"; */

	return 0;

}

