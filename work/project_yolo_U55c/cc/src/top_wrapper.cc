#include <iostream>
#include <fstream>
#include "yolo.h"
#include "params.h"

extern "C++" {
        void top_wrapper(
                        const t_in_mem *inp_1,
                        const t_params_st *c_params,
                        t_out_mem1 * o_outp1,
			t_out_mem2 * o_outp2)
        {
#ifndef __SYNTHESIS__
                std::cout << "Starting top_wrapper" << std::endl;
#endif
#pragma HLS interface mode = ap_ctrl_hs port = return
#pragma HLS dataflow disable_start_propagation

#pragma HLS INTERFACE mode = m_axi port = c_params bundle = m_axi_w depth = 8649648
#pragma HLS INTERFACE mode = m_axi port = inp_1 bundle = m_axi_a depth = 519168 / 8 // 416 * 416 * 3 
#pragma HLS INTERFACE mode = m_axi port = o_outp1 bundle = m_axi_o depth = 86528
#pragma HLS INTERFACE mode = m_axi port = o_outp2 bundle = m_axi_o2 depth = 173056 

                yolo(
                        inp_1,
                        c_params,
                        o_outp1,
			o_outp2);
        }
}
