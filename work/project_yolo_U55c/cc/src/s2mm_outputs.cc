#include "nn2fpga/s2mm.h"
#include "params.h"

extern "C++" {
	void s2mm_outputs(
		t_out_mem1* o_outp1,
		t_out_mem2* o_outp2,
		const unsigned int n_out1,
		const unsigned int n_out2,
		hls::stream<t_o_outp1>& c_outp1_stream
		hls::stream<t_o_outp2>& c_outp2_stream
	) {

		nn2fpga::s2mm<t_out_mem1, t_o_outp1>
			(o_outp1, n_out1, c_outp1_stream);
		
		nn2fpga::s2mm<t_out_mem2, t_o_outp2>
			(o_outp2, n_out2, c_outp2_stream);
		

	}
}