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
                yolo(
                        inp_1,
                        c_params,
                        o_outp1,
			o_outp2);
        }
}
