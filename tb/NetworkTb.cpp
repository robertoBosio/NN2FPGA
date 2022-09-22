#include "../src/Network.hpp"

int main() {

	t_i_data i_data[c_input_ich*c_input_ih*c_input_iw];
	t_weight i_weight[10];
	t_o_data o_data_exp[10];
	t_o_data o_data_sim[10];
	int o_last;

	// INIT DATA
	int s_index = 0;

	for (int s_input_ich = 0; s_input_ich < c_input_ich; s_input_ich++) {
		for (int s_input_ih = 0; s_input_ih < c_input_ih; s_input_ih++) {
			for (int s_input_iw = 0; s_input_iw < c_input_iw; s_input_iw++) {
				i_data[s_index] = rand() % 256;
				/* std::cout << i_data[s_index] << "\n"; */
				s_index++;
			}
		}
	}

	std::cout << "--------------------- KERNEL -----------------------" << "\n";
	Network(
		i_data,
		o_data_sim
	);

	/* while(o_last == 0); */
	/* while(o_last == 1); */
	
	std::cout << "EXP: " << o_data_exp[0] << "\n";

	return 0;

}

