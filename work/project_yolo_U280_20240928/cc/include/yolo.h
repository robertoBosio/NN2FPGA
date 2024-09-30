#ifndef __YOLO__H__
#define __YOLO__H__
#include "params.h"

void yolo(
	hls::stream<t_in_mem> &i_inp_1,
	hls::stream<t_params_stream> &i_data_params,
	hls::stream<t_net_19> &o_outp1,
	hls::stream<t_net_25> &o_outp2
);

#endif  /*__YOLO__H__ */