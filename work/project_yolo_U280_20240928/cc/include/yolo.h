#ifndef __YOLO__H__
#define __YOLO__H__
#include "params.h"

void yolo(
	hls::stream<t_inp_1> &i_inp_1,
	hls::stream<t_params_axi_stream> &i_data_params,
	hls::stream<t_o_outp1> &o_outp1,
	hls::stream<t_o_outp1> &o_outp2
);

#endif  /*__YOLO__H__ */