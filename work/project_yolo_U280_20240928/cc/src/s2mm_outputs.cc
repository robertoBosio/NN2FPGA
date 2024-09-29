#include "nn2fpga/s2mm.h"
#include "params.h"

extern "C++" {
	void s2mm_outputs(
		t_out_mem* o_outp1,
		const unsigned int n_out,
		hls::stream<t_o_outp1>& c_outp1_stream
	) {

		nn2fpga::s2mm<t_out_mem, t_o_outp1>
			(o_outp1, n_out, c_outp1_stream);

	}
}