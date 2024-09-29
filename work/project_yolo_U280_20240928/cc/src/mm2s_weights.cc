#include "nn2fpga/mm2s.h"
#include "params.h"

extern "C++" {
	void mm2s_weights(
		const t_params_st* c_params,
		const unsigned int c_params_dim,
		hls::stream<t_params_axi_stream>& c_params_stream
	) {

		nn2fpga::mm2s<t_params_st, t_params_axi_stream>
			(c_params, c_params_dim, c_params_stream);

	}
}