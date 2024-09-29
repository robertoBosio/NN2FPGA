#include "nn2fpga/mm2s.h"
#include "params.h"

extern "C++" {
	void mm2s_activations(
		const t_in_mem* inp_1,
		const unsigned int n_inp,
		hls::stream<t_inp_1>& c_inp_1_stream
	) {

		nn2fpga::mm2s<t_in_mem, t_inp_1>
			(inp_1, n_inp, c_inp_1_stream);

	}
}