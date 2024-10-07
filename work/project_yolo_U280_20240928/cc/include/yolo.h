#ifndef YOLO_H
#define YOLO_H

#include "params.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "nn2fpga/packed_conv.h"
#include "nn2fpga/pool_streams.h"
#include "nn2fpga/utils.h"
#include "nn2fpga/activations_utils.h"
#include "nn2fpga/block_interface.h"
#include "nn2fpga/line_buffer_utils.h"
#include "nn2fpga/quantisation.h"
#include "nn2fpga/weights_utils.h"
#include "nn2fpga/mm2s.h"
#include "nn2fpga/s2mm.h"

#include "params.h"


template<typename T_mem, typename T_stream, size_t size>
void
mm2s_params(const T_mem* mem, hls::stream<T_stream>& stream, bool &s_init)
{
	if (!s_init)
	{
	MM2S_PARAMS_LOOP:
		for (auto it = 0; it < size; ++it)
		{
#pragma HLS pipeline II = 1
			T_stream s_data;
			auto data = mem[it];
			s_data = data;
			stream.write(s_data);
		}
		s_init = true;
	}
}

extern "C++" {

void yolo(
	// hls::stream<t_in_mem> &i_inp_1,
	// hls::stream<t_params_stream> &i_data_params,
	// hls::stream<t_net_19> &o_outp1,
	// hls::stream<t_net_25> &o_outp2
	const t_in_mem* inp_1,
	const t_params_st* c_params,
	t_out_mem1* o_outp1,
	t_out_mem2* o_outp2
) {
	#pragma HLS interface mode = ap_ctrl_hs port = return
	#pragma HLS INTERFACE mode = m_axi port = c_params bundle = m_axi_w depth=8649648
	#pragma HLS INTERFACE mode = m_axi port = inp_1 bundle = m_axi_a depth=519168
	#pragma HLS INTERFACE mode = m_axi port = o_outp1 bundle = m_axi_o depth=78432
	#pragma HLS INTERFACE mode = m_axi port = o_outp1 bundle = m_axi_o depth=173056
	// #pragma HLS interface mode=ap_ctrl_none port=return
	
	
	static bool s_axi_to_stream_init_flag = false;
	hls::stream<t_params_stream> s_axi_to_stream_out[1];
	#pragma HLS stream variable=s_axi_to_stream_out depth=2 type=fifo
	// #pragma HLS interface port=i_data_params mode=axis
	hls::stream<t_net_5_struct> s_net_5[1];
	#pragma HLS stream variable=s_net_5 depth=3 type=fifo
	// #pragma HLS interface port=i_inp_1 mode=axis
	#pragma HLS dataflow disable_start_propagation
	hls::stream<t_net_5_lb_struct> s_net_5_data[8];
	hls::stream<std::nullptr_t> s_net_5_null[1];
	hls::stream<t_net_5_lb_struct> s_net_5_pre_pad[9];
	#pragma HLS stream variable=s_net_5_pre_pad depth=10 type=fifo
	#pragma HLS bind_storage variable=s_net_5_pre_pad impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_5_data[0] depth=1 type=fifo
	#pragma HLS bind_storage variable=s_net_5_data[0] impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_5_data[1] depth=1 type=fifo
	#pragma HLS stream variable=s_net_5_data[2] depth=416 type=fifo
	#pragma HLS stream variable=s_net_5_data[3] depth=1 type=fifo
	#pragma HLS stream variable=s_net_5_data[4] depth=1 type=fifo
	#pragma HLS stream variable=s_net_5_data[5] depth=416 type=fifo
	#pragma HLS stream variable=s_net_5_data[6] depth=1 type=fifo
	#pragma HLS stream variable=s_net_5_data[7] depth=1 type=fifo
	hls::stream<t_net_5_window_struct> s_net_5_compute[1];
	#pragma HLS stream variable=s_net_5_compute depth=2 type=fifo
	hls::stream<t_net_6_struct> s_net_6[1];
	t_node_conv_1_weight_mem static c_node_conv_1_weight[9][1][48];
	static bool s_node_conv_1_init_flag = false;
	hls::stream<t_params_stream> s_node_conv_1_out[1];
	#pragma HLS array_reshape variable=c_node_conv_1_weight dim=3 type=complete
	#pragma HLS array_reshape variable=c_node_conv_1_weight dim=1 type=complete
	#pragma HLS array_partition variable=c_node_conv_1_weight off=true
	#pragma HLS stream variable=s_net_6 depth=2 type=fifo
	hls::stream<t_net_6_lb_struct> s_net_6_data[3];
	hls::stream<std::nullptr_t> s_net_6_null[1];
	hls::stream<t_net_6_lb_struct> s_net_6_pre_pad[4];
	#pragma HLS stream variable=s_net_6_pre_pad depth=65 type=fifo
	#pragma HLS bind_storage variable=s_net_6_pre_pad impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_6_data[0] depth=1 type=fifo
	#pragma HLS bind_storage variable=s_net_6_data[0] impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_6_data[1] depth=416 type=fifo
	#pragma HLS stream variable=s_net_6_data[2] depth=1 type=fifo
	hls::stream<t_net_6_window_struct> s_net_6_compute[1];
	#pragma HLS stream variable=s_net_6_compute depth=2 type=fifo
	hls::stream<t_net_7_struct> s_net_7[1];
	#pragma HLS stream variable=s_net_7 depth=1 type=fifo
	hls::stream<t_net_7_adj_struct> s_net_7_adj[2];
	#pragma HLS stream variable=s_net_7_adj depth=2 type=fifo
	hls::stream<t_net_7_lb_struct> s_net_7_data[10];
	hls::stream<std::nullptr_t> s_net_7_null[2];
	hls::stream<t_net_7_lb_struct> s_net_7_pre_pad[12];
	#pragma HLS stream variable=s_net_7_pre_pad depth=10 type=fifo
	#pragma HLS bind_storage variable=s_net_7_pre_pad impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_7_data[0] depth=2 type=fifo
	#pragma HLS bind_storage variable=s_net_7_data[0] impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_7_data[1] depth=2 type=fifo
	#pragma HLS stream variable=s_net_7_data[2] depth=208 type=fifo
	#pragma HLS stream variable=s_net_7_data[3] depth=208 type=fifo
	#pragma HLS stream variable=s_net_7_data[4] depth=2 type=fifo
	#pragma HLS stream variable=s_net_7_data[5] depth=2 type=fifo
	#pragma HLS stream variable=s_net_7_data[6] depth=208 type=fifo
	#pragma HLS stream variable=s_net_7_data[7] depth=208 type=fifo
	#pragma HLS stream variable=s_net_7_data[8] depth=2 type=fifo
	#pragma HLS stream variable=s_net_7_data[9] depth=2 type=fifo
	hls::stream<t_net_7_window_struct> s_net_7_compute[1];
	#pragma HLS stream variable=s_net_7_compute depth=2 type=fifo
	hls::stream<t_net_8_struct> s_net_8[2];
	t_node_conv_3_weight_mem static c_node_conv_3_weight[9][8][64];
	static bool s_node_conv_3_init_flag = false;
	hls::stream<t_params_stream> s_node_conv_3_out[1];
	#pragma HLS array_reshape variable=c_node_conv_3_weight dim=3 type=complete
	#pragma HLS array_reshape variable=c_node_conv_3_weight dim=1 type=complete
	#pragma HLS array_partition variable=c_node_conv_3_weight off=true
	#pragma HLS stream variable=s_net_8 depth=5 type=fifo
	hls::stream<t_net_8_adj_struct> s_net_8_adj[1];
	#pragma HLS stream variable=s_net_8_adj depth=3 type=fifo
	hls::stream<t_net_8_lb_struct> s_net_8_data[3];
	hls::stream<std::nullptr_t> s_net_8_null[1];
	hls::stream<t_net_8_lb_struct> s_net_8_pre_pad[4];
	#pragma HLS stream variable=s_net_8_pre_pad depth=129 type=fifo
	#pragma HLS bind_storage variable=s_net_8_pre_pad impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_8_data[0] depth=2 type=fifo
	#pragma HLS bind_storage variable=s_net_8_data[0] impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_8_data[1] depth=416 type=fifo
	#pragma HLS stream variable=s_net_8_data[2] depth=2 type=fifo
	hls::stream<t_net_8_window_struct> s_net_8_compute[1];
	#pragma HLS stream variable=s_net_8_compute depth=2 type=fifo
	hls::stream<t_net_9_struct> s_net_9[1];
	#pragma HLS stream variable=s_net_9 depth=2 type=fifo
	hls::stream<t_net_9_adj_struct> s_net_9_adj[2];
	#pragma HLS stream variable=s_net_9_adj depth=3 type=fifo
	hls::stream<t_net_9_lb_struct> s_net_9_data[10];
	hls::stream<std::nullptr_t> s_net_9_null[2];
	hls::stream<t_net_9_lb_struct> s_net_9_pre_pad[12];
	#pragma HLS stream variable=s_net_9_pre_pad depth=10 type=fifo
	#pragma HLS bind_storage variable=s_net_9_pre_pad impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_9_data[0] depth=4 type=fifo
	#pragma HLS bind_storage variable=s_net_9_data[0] impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_9_data[1] depth=4 type=fifo
	#pragma HLS stream variable=s_net_9_data[2] depth=208 type=fifo
	#pragma HLS stream variable=s_net_9_data[3] depth=208 type=fifo
	#pragma HLS stream variable=s_net_9_data[4] depth=4 type=fifo
	#pragma HLS stream variable=s_net_9_data[5] depth=4 type=fifo
	#pragma HLS stream variable=s_net_9_data[6] depth=208 type=fifo
	#pragma HLS stream variable=s_net_9_data[7] depth=208 type=fifo
	#pragma HLS stream variable=s_net_9_data[8] depth=4 type=fifo
	#pragma HLS stream variable=s_net_9_data[9] depth=4 type=fifo
	hls::stream<t_net_9_window_struct> s_net_9_compute[1];
	#pragma HLS stream variable=s_net_9_compute depth=2 type=fifo
	hls::stream<t_net_10_struct> s_net_10[2];
	t_node_conv_5_weight_mem static c_node_conv_5_weight[9][32][64];
	static bool s_node_conv_5_init_flag = false;
	hls::stream<t_params_stream> s_node_conv_5_out[1];
	#pragma HLS array_reshape variable=c_node_conv_5_weight dim=3 type=complete
	#pragma HLS array_reshape variable=c_node_conv_5_weight dim=1 type=complete
	#pragma HLS array_partition variable=c_node_conv_5_weight off=true
	#pragma HLS stream variable=s_net_10 depth=17 type=fifo
	hls::stream<t_net_10_adj_struct> s_net_10_adj[1];
	#pragma HLS stream variable=s_net_10_adj depth=9 type=fifo
	hls::stream<t_net_10_lb_struct> s_net_10_data[3];
	hls::stream<std::nullptr_t> s_net_10_null[1];
	hls::stream<t_net_10_lb_struct> s_net_10_pre_pad[4];
	#pragma HLS stream variable=s_net_10_pre_pad depth=257 type=fifo
	#pragma HLS bind_storage variable=s_net_10_pre_pad impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_10_data[0] depth=8 type=fifo
	#pragma HLS bind_storage variable=s_net_10_data[0] impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_10_data[1] depth=832 type=fifo
	#pragma HLS stream variable=s_net_10_data[2] depth=8 type=fifo
	hls::stream<t_net_10_window_struct> s_net_10_compute[1];
	#pragma HLS stream variable=s_net_10_compute depth=2 type=fifo
	hls::stream<t_net_11_struct> s_net_11[1];
	#pragma HLS stream variable=s_net_11 depth=8 type=fifo
	hls::stream<t_net_11_adj_struct> s_net_11_adj[2];
	#pragma HLS stream variable=s_net_11_adj depth=9 type=fifo
	hls::stream<t_net_11_lb_struct> s_net_11_data[10];
	hls::stream<std::nullptr_t> s_net_11_null[2];
	hls::stream<t_net_11_lb_struct> s_net_11_pre_pad[12];
	#pragma HLS stream variable=s_net_11_pre_pad depth=10 type=fifo
	#pragma HLS bind_storage variable=s_net_11_pre_pad impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_11_data[0] depth=8 type=fifo
	#pragma HLS bind_storage variable=s_net_11_data[0] impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_11_data[1] depth=8 type=fifo
	#pragma HLS stream variable=s_net_11_data[2] depth=208 type=fifo
	#pragma HLS stream variable=s_net_11_data[3] depth=208 type=fifo
	#pragma HLS stream variable=s_net_11_data[4] depth=8 type=fifo
	#pragma HLS stream variable=s_net_11_data[5] depth=8 type=fifo
	#pragma HLS stream variable=s_net_11_data[6] depth=208 type=fifo
	#pragma HLS stream variable=s_net_11_data[7] depth=208 type=fifo
	#pragma HLS stream variable=s_net_11_data[8] depth=8 type=fifo
	#pragma HLS stream variable=s_net_11_data[9] depth=8 type=fifo
	hls::stream<t_net_11_window_struct> s_net_11_compute[1];
	#pragma HLS stream variable=s_net_11_compute depth=2 type=fifo
	hls::stream<t_net_12_struct> s_net_12[2];
	t_node_conv_7_weight_mem static c_node_conv_7_weight[9][128][64];
	static bool s_node_conv_7_init_flag = false;
	hls::stream<t_params_stream> s_node_conv_7_out[1];
	#pragma HLS array_reshape variable=c_node_conv_7_weight dim=3 type=complete
	#pragma HLS array_reshape variable=c_node_conv_7_weight dim=1 type=complete
	#pragma HLS array_partition variable=c_node_conv_7_weight off=true
	#pragma HLS stream variable=s_net_12 depth=33 type=fifo
	hls::stream<t_net_12_adj_struct> s_net_12_adj[1];
	#pragma HLS stream variable=s_net_12_adj depth=17 type=fifo
	hls::stream<t_net_12_lb_struct> s_net_12_data[3];
	hls::stream<std::nullptr_t> s_net_12_null[1];
	hls::stream<t_net_12_lb_struct> s_net_12_pre_pad[4];
	#pragma HLS stream variable=s_net_12_pre_pad depth=513 type=fifo
	#pragma HLS bind_storage variable=s_net_12_pre_pad impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_12_data[0] depth=16 type=fifo
	#pragma HLS bind_storage variable=s_net_12_data[0] impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_12_data[1] depth=832 type=fifo
	#pragma HLS stream variable=s_net_12_data[2] depth=16 type=fifo
	hls::stream<t_net_12_window_struct> s_net_12_compute[1];
	#pragma HLS stream variable=s_net_12_compute depth=2 type=fifo
	hls::stream<t_net_13_struct> s_net_13[1];
	#pragma HLS stream variable=s_net_13 depth=16 type=fifo
	hls::stream<t_net_13_adj_struct> s_net_13_adj[2];
	#pragma HLS stream variable=s_net_13_adj depth=17 type=fifo
	hls::stream<t_net_13_lb_struct> s_net_13_data[10];
	hls::stream<std::nullptr_t> s_net_13_null[2];
	hls::stream<t_net_13_lb_struct> s_net_13_pre_pad[12];
	#pragma HLS stream variable=s_net_13_pre_pad depth=10 type=fifo
	#pragma HLS bind_storage variable=s_net_13_pre_pad impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_13_data[0] depth=16 type=fifo
	#pragma HLS bind_storage variable=s_net_13_data[0] impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_13_data[1] depth=16 type=fifo
	#pragma HLS stream variable=s_net_13_data[2] depth=208 type=fifo
	#pragma HLS stream variable=s_net_13_data[3] depth=208 type=fifo
	#pragma HLS stream variable=s_net_13_data[4] depth=16 type=fifo
	#pragma HLS stream variable=s_net_13_data[5] depth=16 type=fifo
	#pragma HLS stream variable=s_net_13_data[6] depth=208 type=fifo
	#pragma HLS stream variable=s_net_13_data[7] depth=208 type=fifo
	#pragma HLS stream variable=s_net_13_data[8] depth=16 type=fifo
	#pragma HLS stream variable=s_net_13_data[9] depth=16 type=fifo
	hls::stream<t_net_13_window_struct> s_net_13_compute[1];
	#pragma HLS stream variable=s_net_13_compute depth=2 type=fifo
	hls::stream<t_net_14_struct> s_net_14[2];
	t_node_conv_9_weight_mem static c_node_conv_9_weight[9][512][64];
	static bool s_node_conv_9_init_flag = false;
	hls::stream<t_params_stream> s_node_conv_9_out[1];
	#pragma HLS array_reshape variable=c_node_conv_9_weight dim=3 type=complete
	#pragma HLS array_reshape variable=c_node_conv_9_weight dim=1 type=complete
	#pragma HLS array_partition variable=c_node_conv_9_weight off=true
	#pragma HLS stream variable=s_net_14 depth=65 type=fifo
	hls::stream<t_net_14_dup_0_struct> s_net_14_dup_0[2];
	hls::stream<t_net_14_dup_1_struct> s_net_14_dup_1[2];
	hls::stream<t_net_20_struct> s_net_14_dup_1_adj[1];
	#pragma HLS stream variable=s_net_14_dup_0 depth=33 type=fifo
	#pragma HLS stream variable=s_net_14_dup_1 depth=2260 type=fifo
	hls::stream<t_net_14_dup_0_adj_struct> s_net_14_dup_0_adj[1];
	#pragma HLS stream variable=s_net_14_dup_0_adj depth=33 type=fifo
	hls::stream<t_net_14_dup_0_lb_struct> s_net_14_dup_0_data[3];
	hls::stream<std::nullptr_t> s_net_14_dup_0_null[1];
	hls::stream<t_net_14_dup_0_lb_struct> s_net_14_dup_0_pre_pad[4];
	#pragma HLS stream variable=s_net_14_dup_0_pre_pad depth=1025 type=fifo
	#pragma HLS bind_storage variable=s_net_14_dup_0_pre_pad impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_14_dup_0_data[0] depth=32 type=fifo
	#pragma HLS bind_storage variable=s_net_14_dup_0_data[0] impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_14_dup_0_data[1] depth=832 type=fifo
	#pragma HLS stream variable=s_net_14_dup_0_data[2] depth=32 type=fifo
	hls::stream<t_net_14_dup_0_window_struct> s_net_14_dup_0_compute[1];
	#pragma HLS stream variable=s_net_14_dup_0_compute depth=2 type=fifo
	hls::stream<t_net_15_struct> s_net_15[1];
	#pragma HLS stream variable=s_net_15 depth=32 type=fifo
	hls::stream<t_net_15_lb_struct> s_net_15_data[8];
	hls::stream<std::nullptr_t> s_net_15_null[1];
	hls::stream<t_net_15_lb_struct> s_net_15_pre_pad[9];
	#pragma HLS stream variable=s_net_15_pre_pad depth=10 type=fifo
	#pragma HLS bind_storage variable=s_net_15_pre_pad impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_15_data[0] depth=32 type=fifo
	#pragma HLS bind_storage variable=s_net_15_data[0] impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_15_data[1] depth=32 type=fifo
	#pragma HLS stream variable=s_net_15_data[2] depth=416 type=fifo
	#pragma HLS stream variable=s_net_15_data[3] depth=32 type=fifo
	#pragma HLS stream variable=s_net_15_data[4] depth=32 type=fifo
	#pragma HLS stream variable=s_net_15_data[5] depth=416 type=fifo
	#pragma HLS stream variable=s_net_15_data[6] depth=32 type=fifo
	#pragma HLS stream variable=s_net_15_data[7] depth=32 type=fifo
	hls::stream<t_net_15_window_struct> s_net_15_compute[1];
	#pragma HLS stream variable=s_net_15_compute depth=2 type=fifo
	hls::stream<t_net_16_struct> s_net_16[1];
	t_node_conv_11_weight_mem static c_node_conv_11_weight[9][1024][128];
	static bool s_node_conv_11_init_flag = false;
	hls::stream<t_params_stream> s_node_conv_11_out[1];
	#pragma HLS bind_storage variable=c_node_conv_11_weight impl=uram type=ram_s2p
	#pragma HLS array_reshape variable=c_node_conv_11_weight dim=3 type=complete
	#pragma HLS array_reshape variable=c_node_conv_11_weight dim=1 type=complete
	#pragma HLS array_partition variable=c_node_conv_11_weight off=true
	#pragma HLS stream variable=s_net_16 depth=17 type=fifo
	hls::stream<t_net_16_lb_struct> s_net_16_data[8];
	hls::stream<std::nullptr_t> s_net_16_null[1];
	hls::stream<t_net_16_lb_struct> s_net_16_pre_pad[9];
	#pragma HLS stream variable=s_net_16_pre_pad depth=10 type=fifo
	#pragma HLS bind_storage variable=s_net_16_pre_pad impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_16_data[0] depth=16 type=fifo
	#pragma HLS bind_storage variable=s_net_16_data[0] impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_16_data[1] depth=16 type=fifo
	#pragma HLS stream variable=s_net_16_data[2] depth=208 type=fifo
	#pragma HLS stream variable=s_net_16_data[3] depth=16 type=fifo
	#pragma HLS stream variable=s_net_16_data[4] depth=16 type=fifo
	#pragma HLS stream variable=s_net_16_data[5] depth=208 type=fifo
	#pragma HLS stream variable=s_net_16_data[6] depth=16 type=fifo
	#pragma HLS stream variable=s_net_16_data[7] depth=16 type=fifo
	hls::stream<t_net_16_window_struct> s_net_16_compute[1];
	#pragma HLS stream variable=s_net_16_compute depth=2 type=fifo
	hls::stream<t_net_17_struct> s_net_17[1];
	t_node_conv_12_weight_mem static c_node_conv_12_weight[9][1024][512];
	static bool s_node_conv_12_init_flag = false;
	hls::stream<t_params_stream> s_node_conv_12_out[1];
	#pragma HLS bind_storage variable=c_node_conv_12_weight impl=uram type=ram_s2p
	#pragma HLS array_reshape variable=c_node_conv_12_weight dim=3 type=complete
	#pragma HLS array_reshape variable=c_node_conv_12_weight dim=1 type=complete
	#pragma HLS array_partition variable=c_node_conv_12_weight off=true
	#pragma HLS stream variable=s_net_17 depth=65 type=fifo
	hls::stream<t_net_17_lb_struct> s_net_17_data[0];
	hls::stream<std::nullptr_t> s_net_17_null[1];
	hls::stream<t_net_17_lb_struct> s_net_17_pre_pad[1];
	#pragma HLS stream variable=s_net_17_pre_pad depth=2 type=fifo
	#pragma HLS bind_storage variable=s_net_17_pre_pad impl=AUTO type=fifo
	hls::stream<t_net_18_struct> s_net_18[1];
	t_node_conv_13_weight_mem static c_node_conv_13_weight[1][1024][256];
	static bool s_node_conv_13_init_flag = false;
	hls::stream<t_params_stream> s_node_conv_13_out[1];
	#pragma HLS bind_storage variable=c_node_conv_13_weight impl=uram type=ram_s2p
	#pragma HLS array_reshape variable=c_node_conv_13_weight dim=3 type=complete
	#pragma HLS array_reshape variable=c_node_conv_13_weight dim=1 type=complete
	#pragma HLS array_partition variable=c_node_conv_13_weight off=true
	#pragma HLS stream variable=s_net_18 depth=17 type=fifo
	hls::stream<t_net_18_dup_0_struct> s_net_18_dup_0[1];
	hls::stream<t_net_18_dup_1_struct> s_net_18_dup_1[1];
	#pragma HLS stream variable=s_net_18_dup_0 depth=17 type=fifo
	#pragma HLS stream variable=s_net_18_dup_1 depth=17 type=fifo
	hls::stream<t_net_18_dup_0_lb_struct> s_net_18_dup_0_data[8];
	hls::stream<std::nullptr_t> s_net_18_dup_0_null[1];
	hls::stream<t_net_18_dup_0_lb_struct> s_net_18_dup_0_pre_pad[9];
	#pragma HLS stream variable=s_net_18_dup_0_pre_pad depth=10 type=fifo
	#pragma HLS bind_storage variable=s_net_18_dup_0_pre_pad impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_18_dup_0_data[0] depth=32 type=fifo
	#pragma HLS bind_storage variable=s_net_18_dup_0_data[0] impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_18_dup_0_data[1] depth=32 type=fifo
	#pragma HLS stream variable=s_net_18_dup_0_data[2] depth=416 type=fifo
	#pragma HLS stream variable=s_net_18_dup_0_data[3] depth=32 type=fifo
	#pragma HLS stream variable=s_net_18_dup_0_data[4] depth=32 type=fifo
	#pragma HLS stream variable=s_net_18_dup_0_data[5] depth=416 type=fifo
	#pragma HLS stream variable=s_net_18_dup_0_data[6] depth=32 type=fifo
	#pragma HLS stream variable=s_net_18_dup_0_data[7] depth=32 type=fifo
	hls::stream<t_net_18_dup_0_window_struct> s_net_18_dup_0_compute[1];
	#pragma HLS stream variable=s_net_18_dup_0_compute depth=2 type=fifo
	hls::stream<t_net_19_struct> s_net_19[1];
	t_node_conv_14_weight_mem static c_node_conv_14_weight[9][1024][128];
	static bool s_node_conv_14_init_flag = false;
	hls::stream<t_params_stream> s_node_conv_14_out[1];
	#pragma HLS bind_storage variable=c_node_conv_14_weight impl=uram type=ram_s2p
	#pragma HLS array_reshape variable=c_node_conv_14_weight dim=3 type=complete
	#pragma HLS array_reshape variable=c_node_conv_14_weight dim=1 type=complete
	#pragma HLS array_partition variable=c_node_conv_14_weight off=true
	#pragma HLS stream variable=s_net_19 depth=33 type=fifo
	hls::stream<t_net_18_dup_1_lb_struct> s_net_18_dup_1_data[0];
	hls::stream<std::nullptr_t> s_net_18_dup_1_null[1];
	hls::stream<t_net_18_dup_1_lb_struct> s_net_18_dup_1_pre_pad[1];
	#pragma HLS stream variable=s_net_18_dup_1_pre_pad depth=2 type=fifo
	#pragma HLS bind_storage variable=s_net_18_dup_1_pre_pad impl=AUTO type=fifo
	hls::stream<t_net_20_struct> s_net_20[1];
	t_node_conv_15_weight_mem static c_node_conv_15_weight[1][1024][32];
	static bool s_node_conv_15_init_flag = false;
	hls::stream<t_params_stream> s_node_conv_15_out[1];
	#pragma HLS bind_storage variable=c_node_conv_15_weight impl=uram type=ram_s2p
	#pragma HLS array_reshape variable=c_node_conv_15_weight dim=3 type=complete
	#pragma HLS array_reshape variable=c_node_conv_15_weight dim=1 type=complete
	#pragma HLS array_partition variable=c_node_conv_15_weight off=true
	#pragma HLS stream variable=s_net_20 depth=33 type=fifo
	hls::stream<t_net_20_struct> s_net_23[1];
	hls::stream<t_net_20_struct> s_net_24_pre_adj[1];
	hls::stream<t_net_24_struct> s_net_24[1];
	const int c_node_concat_17_ich[2] = {128,256};
	hls::stream<t_net_24_lb_struct> s_net_24_data[8];
	hls::stream<std::nullptr_t> s_net_24_null[1];
	hls::stream<t_net_24_lb_struct> s_net_24_pre_pad[9];
	#pragma HLS stream variable=s_net_24_pre_pad depth=10 type=fifo
	#pragma HLS bind_storage variable=s_net_24_pre_pad impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_24_data[0] depth=16 type=fifo
	#pragma HLS bind_storage variable=s_net_24_data[0] impl=AUTO type=fifo
	#pragma HLS stream variable=s_net_24_data[1] depth=16 type=fifo
	#pragma HLS stream variable=s_net_24_data[2] depth=416 type=fifo
	#pragma HLS stream variable=s_net_24_data[3] depth=16 type=fifo
	#pragma HLS stream variable=s_net_24_data[4] depth=16 type=fifo
	#pragma HLS stream variable=s_net_24_data[5] depth=416 type=fifo
	#pragma HLS stream variable=s_net_24_data[6] depth=16 type=fifo
	#pragma HLS stream variable=s_net_24_data[7] depth=16 type=fifo
	hls::stream<t_net_24_window_struct> s_net_24_compute[1];
	#pragma HLS stream variable=s_net_24_compute depth=2 type=fifo
	hls::stream<t_net_25_struct> s_net_25[1];
	t_node_conv_18_weight_mem static c_node_conv_18_weight[9][256][384];
	static bool s_node_conv_18_init_flag = false;
	hls::stream<t_params_stream> s_node_conv_18_out[1];
	#pragma HLS array_reshape variable=c_node_conv_18_weight dim=3 type=complete
	#pragma HLS array_reshape variable=c_node_conv_18_weight dim=1 type=complete
	#pragma HLS array_partition variable=c_node_conv_18_weight off=true
	#pragma HLS stream variable=s_net_25 depth=17 type=fifo
	// #pragma HLS interface port=o_outp1 mode=axis
	// #pragma HLS interface port=o_outp1 mode=axis


	const int n_inp = 416 * 416 * 3 / 8;
	const int n_out2 = 256 * 26 * 26;
	const int n_out1 = 512 * 13 * 13;
	hls::stream<t_params_stream, 8> i_data_params;
	hls::stream<t_in_mem, 8> c_inp_1_stream;
	hls::stream<t_out_mem1, 8> c_outp1_stream;
	hls::stream<t_out_mem2, 8> c_outp2_stream;	
	static bool init = false;

	mm2s_params <t_params_st, t_params_stream, 8649648> (
		c_params,
		i_data_params,
		init
	);
	
	nn2fpga::axi_to_stream <
		t_params_stream,
		t_params_stream,
		8649648>		//tot cycles
	(
		i_data_params,
		s_axi_to_stream_init_flag,
		s_axi_to_stream_out
	);

	nn2fpga::mm2s<t_in_mem, t_in_mem> (
		inp_1,
		n_inp,
		c_inp_1_stream
	);

	nn2fpga::produce_stream <
		t_in_mem,
		t_inp_1_part,
		t_net_5_struct,
		t_net_5,
		c_node_produce_0_ich,
		c_node_produce_0_iw,
		c_node_produce_0_ih,
		c_inp_1,
		c_node_produce_0_ops,
		c_act_width,
		false>		//transform flag
	(
		c_inp_1_stream,
		s_net_5
	);


	nn2fpga::shift_op <
		t_net_5_struct,
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		c_node_conv_1_ich,
		c_node_conv_1_och,
		c_node_conv_1_ih,
		c_node_conv_1_iw,
		c_node_conv_1_oh,
		c_node_conv_1_ow,
		c_node_conv_1_fh,
		c_node_conv_1_fw,
		c_node_conv_1_stride,
		c_node_conv_1_pad,
		2,
		2,
		c_node_conv_1_ow_ops,
		3,
		3>
	(
		s_net_5[0],
		s_net_5_pre_pad[0],
		s_net_5_data[0]
	);

	nn2fpga::shift_op <
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		c_node_conv_1_ich,
		c_node_conv_1_och,
		c_node_conv_1_ih,
		c_node_conv_1_iw,
		c_node_conv_1_oh,
		c_node_conv_1_ow,
		c_node_conv_1_fh,
		c_node_conv_1_fw,
		c_node_conv_1_stride,
		c_node_conv_1_pad,
		2,
		1,
		c_node_conv_1_ow_ops,
		3,
		3>
	(
		s_net_5_data[0],
		s_net_5_pre_pad[1],
		s_net_5_data[1]
	);

	nn2fpga::shift_op <
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		c_node_conv_1_ich,
		c_node_conv_1_och,
		c_node_conv_1_ih,
		c_node_conv_1_iw,
		c_node_conv_1_oh,
		c_node_conv_1_ow,
		c_node_conv_1_fh,
		c_node_conv_1_fw,
		c_node_conv_1_stride,
		c_node_conv_1_pad,
		2,
		0,
		c_node_conv_1_ow_ops,
		3,
		3>
	(
		s_net_5_data[1],
		s_net_5_pre_pad[2],
		s_net_5_data[2]
	);

	nn2fpga::shift_op <
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		c_node_conv_1_ich,
		c_node_conv_1_och,
		c_node_conv_1_ih,
		c_node_conv_1_iw,
		c_node_conv_1_oh,
		c_node_conv_1_ow,
		c_node_conv_1_fh,
		c_node_conv_1_fw,
		c_node_conv_1_stride,
		c_node_conv_1_pad,
		1,
		2,
		c_node_conv_1_ow_ops,
		3,
		3>
	(
		s_net_5_data[2],
		s_net_5_pre_pad[3],
		s_net_5_data[3]
	);

	nn2fpga::shift_op <
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		c_node_conv_1_ich,
		c_node_conv_1_och,
		c_node_conv_1_ih,
		c_node_conv_1_iw,
		c_node_conv_1_oh,
		c_node_conv_1_ow,
		c_node_conv_1_fh,
		c_node_conv_1_fw,
		c_node_conv_1_stride,
		c_node_conv_1_pad,
		1,
		1,
		c_node_conv_1_ow_ops,
		3,
		3>
	(
		s_net_5_data[3],
		s_net_5_pre_pad[4],
		s_net_5_data[4]
	);

	nn2fpga::shift_op <
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		c_node_conv_1_ich,
		c_node_conv_1_och,
		c_node_conv_1_ih,
		c_node_conv_1_iw,
		c_node_conv_1_oh,
		c_node_conv_1_ow,
		c_node_conv_1_fh,
		c_node_conv_1_fw,
		c_node_conv_1_stride,
		c_node_conv_1_pad,
		1,
		0,
		c_node_conv_1_ow_ops,
		3,
		3>
	(
		s_net_5_data[4],
		s_net_5_pre_pad[5],
		s_net_5_data[5]
	);

	nn2fpga::shift_op <
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		c_node_conv_1_ich,
		c_node_conv_1_och,
		c_node_conv_1_ih,
		c_node_conv_1_iw,
		c_node_conv_1_oh,
		c_node_conv_1_ow,
		c_node_conv_1_fh,
		c_node_conv_1_fw,
		c_node_conv_1_stride,
		c_node_conv_1_pad,
		0,
		2,
		c_node_conv_1_ow_ops,
		3,
		3>
	(
		s_net_5_data[5],
		s_net_5_pre_pad[6],
		s_net_5_data[6]
	);

	nn2fpga::shift_op <
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		c_node_conv_1_ich,
		c_node_conv_1_och,
		c_node_conv_1_ih,
		c_node_conv_1_iw,
		c_node_conv_1_oh,
		c_node_conv_1_ow,
		c_node_conv_1_fh,
		c_node_conv_1_fw,
		c_node_conv_1_stride,
		c_node_conv_1_pad,
		0,
		1,
		c_node_conv_1_ow_ops,
		3,
		3>
	(
		s_net_5_data[6],
		s_net_5_pre_pad[7],
		s_net_5_data[7]
	);

	nn2fpga::shift_op <
		t_net_5_lb_struct,
		t_net_5_lb_struct,
		std::nullptr_t,
		c_node_conv_1_ich,
		c_node_conv_1_och,
		c_node_conv_1_ih,
		c_node_conv_1_iw,
		c_node_conv_1_oh,
		c_node_conv_1_ow,
		c_node_conv_1_fh,
		c_node_conv_1_fw,
		c_node_conv_1_stride,
		c_node_conv_1_pad,
		0,
		0,
		c_node_conv_1_ow_ops,
		3,
		3>
	(
		s_net_5_data[7],
		s_net_5_pre_pad[8],
		s_net_5_null[0]
	);

	nn2fpga::pad_input <
		t_net_5_lb_struct,
		t_net_5_window_struct,
		c_node_conv_1_ich,
		c_node_conv_1_ih,
		c_node_conv_1_iw,
		c_node_conv_1_fh,
		c_node_conv_1_fw,
		c_node_conv_1_stride,
		c_node_conv_1_pad,
		c_node_conv_1_ow_ops,
		3,
		3>
	(
		s_net_5_pre_pad,
		s_net_5_compute
	);

	nn2fpga::conv_comp_wrap <
		t_net_5_window_struct,
		t_net_5_window,
		t_net_5_reduce,
		t_net_5,
		t_node_conv_1_weight,
		t_node_conv_1_weight_mem,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_5_mod,
		t_net_5,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_6_acc_struct,
		t_net_6_acc,
		std::nullptr_t,
		std::nullptr_t,
		t_net_6_struct,
		t_net_6_vector,
		t_net_6,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_params_stream,
		c_node_conv_1_ich,
		c_node_conv_1_ich,
		c_node_conv_1_och,
		c_node_conv_1_och_1x1,
		c_node_conv_1_oh,
		c_node_conv_1_ow,
		c_node_conv_1_fh,
		c_node_conv_1_fw,
		c_node_conv_1_index,
		c_node_conv_1_stride,
		c_node_conv_1_ops,
		c_node_conv_1_ops_out,
		c_node_conv_1_ich_ops,
		1,		//add ops
		c_node_conv_1_ow_ops,
		c_node_conv_1_ow_ops_out,
		16,		//bias_ops
		c_node_conv_1_reuse,
		c_node_conv_1_ow_pack,
		c_node_conv_1_och_pack,
		8,		//act bits
		3,		//act integer bits
		8,		//weight bits
		2,		//weight integer bits
		2,		//guard bits
		4,		//accumulators in parallel
		3,		//mask bits
		0,		//act bits 1x1
		0,		//act integer bits 1x1
		0,		//weight bits 1x1
		0,		//weight integer bits 1x1
		2,		//guard bits 1x1
		1,		//accumulators in parallel 1x1
		0,		//mask bits 1x1
		8649216,		//shift cycles
		8,		//width stream
		1,		//weight reads per data
		0,		//bias reads per data
		0,		//weight reads per data 1x1
		0,		//bias reads per data 1x1
		c_node_conv_1_relu,
		0>		//depth
	(
		s_axi_to_stream_out,
		s_node_conv_1_init_flag,
		s_node_conv_1_out,
		nullptr,
		c_node_conv_1_weight,
		nullptr,
		nullptr,
		s_net_5_compute,
		(hls::stream<std::nullptr_t>*)(nullptr),
		(hls::stream<std::nullptr_t>*)(nullptr),
		s_net_6,
		(hls::stream<std::nullptr_t>*)(nullptr)
	);

	nn2fpga::shift_op <
		t_net_6_struct,
		t_net_6_lb_struct,
		t_net_6_lb_struct,
		c_node_pool_2_ich,
		c_node_pool_2_och,
		c_node_pool_2_ih,
		c_node_pool_2_iw,
		c_node_pool_2_oh,
		c_node_pool_2_ow,
		c_node_pool_2_fh,
		c_node_pool_2_fw,
		c_node_pool_2_stride,
		c_node_pool_2_pad,
		1,
		1,
		c_node_pool_2_ow_ops,
		16,
		16>
	(
		s_net_6[0],
		s_net_6_pre_pad[0],
		s_net_6_data[0]
	);

	nn2fpga::shift_op <
		t_net_6_lb_struct,
		t_net_6_lb_struct,
		t_net_6_lb_struct,
		c_node_pool_2_ich,
		c_node_pool_2_och,
		c_node_pool_2_ih,
		c_node_pool_2_iw,
		c_node_pool_2_oh,
		c_node_pool_2_ow,
		c_node_pool_2_fh,
		c_node_pool_2_fw,
		c_node_pool_2_stride,
		c_node_pool_2_pad,
		1,
		0,
		c_node_pool_2_ow_ops,
		16,
		16>
	(
		s_net_6_data[0],
		s_net_6_pre_pad[1],
		s_net_6_data[1]
	);

	nn2fpga::shift_op <
		t_net_6_lb_struct,
		t_net_6_lb_struct,
		t_net_6_lb_struct,
		c_node_pool_2_ich,
		c_node_pool_2_och,
		c_node_pool_2_ih,
		c_node_pool_2_iw,
		c_node_pool_2_oh,
		c_node_pool_2_ow,
		c_node_pool_2_fh,
		c_node_pool_2_fw,
		c_node_pool_2_stride,
		c_node_pool_2_pad,
		0,
		1,
		c_node_pool_2_ow_ops,
		16,
		16>
	(
		s_net_6_data[1],
		s_net_6_pre_pad[2],
		s_net_6_data[2]
	);

	nn2fpga::shift_op <
		t_net_6_lb_struct,
		t_net_6_lb_struct,
		std::nullptr_t,
		c_node_pool_2_ich,
		c_node_pool_2_och,
		c_node_pool_2_ih,
		c_node_pool_2_iw,
		c_node_pool_2_oh,
		c_node_pool_2_ow,
		c_node_pool_2_fh,
		c_node_pool_2_fw,
		c_node_pool_2_stride,
		c_node_pool_2_pad,
		0,
		0,
		c_node_pool_2_ow_ops,
		16,
		16>
	(
		s_net_6_data[2],
		s_net_6_pre_pad[3],
		s_net_6_null[0]
	);

	nn2fpga::pad_input <
		t_net_6_lb_struct,
		t_net_6_window_struct,
		c_node_pool_2_ich,
		c_node_pool_2_ih,
		c_node_pool_2_iw,
		c_node_pool_2_fh,
		c_node_pool_2_fw,
		c_node_pool_2_stride,
		c_node_pool_2_pad,
		c_node_pool_2_ow_ops,
		16,
		16>
	(
		s_net_6_pre_pad,
		s_net_6_compute
	);

	nn2fpga::pool_op <
		t_net_6_window_struct,
		t_net_6_window,
		t_net_7_struct,
		t_net_7,
		t_node_pool_2_acc,
		t_node_pool_2_acc,
		c_node_pool_2_ich,
		c_node_pool_2_och,
		c_node_pool_2_ih,
		c_node_pool_2_iw,
		c_node_pool_2_oh,
		c_node_pool_2_ow,
		c_node_pool_2_fw,
		c_node_pool_2_fh,
		c_node_pool_2_stride,
		c_node_pool_2_pad,
		c_node_pool_2_pool,
		c_node_pool_2_ow_ops,
		c_node_pool_2_ops,
		c_node_pool_2_in_ops>
	(
		s_net_6_compute,
		s_net_7
	);

	nn2fpga::bandwidth_adjust <
		t_net_7_struct,
		t_net_7_adj_struct,
		c_bandwidth_adjust_net_7_ich,
		c_bandwidth_adjust_net_7_iw,
		c_bandwidth_adjust_net_7_ih,
		c_bandwidth_adjust_net_7_ow_ops_in,
		c_bandwidth_adjust_net_7_ow_ops,
		c_bandwidth_adjust_net_7_old_in_ops,
		c_bandwidth_adjust_net_7_in_ops,
		false>		//skip connection flag
	(
		s_net_7,
		s_net_7_adj
	);

	nn2fpga::shift_op <
		t_net_7_adj_struct,
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		c_node_conv_3_ich,
		c_node_conv_3_och,
		c_node_conv_3_ih,
		c_node_conv_3_iw,
		c_node_conv_3_oh,
		c_node_conv_3_ow,
		c_node_conv_3_fh,
		c_node_conv_3_fw,
		c_node_conv_3_stride,
		c_node_conv_3_pad,
		2,
		3,
		c_node_conv_3_ow_ops,
		16,
		8>
	(
		s_net_7_adj[0],
		s_net_7_pre_pad[0],
		s_net_7_data[0]
	);

	nn2fpga::shift_op <
		t_net_7_adj_struct,
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		c_node_conv_3_ich,
		c_node_conv_3_och,
		c_node_conv_3_ih,
		c_node_conv_3_iw,
		c_node_conv_3_oh,
		c_node_conv_3_ow,
		c_node_conv_3_fh,
		c_node_conv_3_fw,
		c_node_conv_3_stride,
		c_node_conv_3_pad,
		2,
		2,
		c_node_conv_3_ow_ops,
		16,
		8>
	(
		s_net_7_adj[1],
		s_net_7_pre_pad[1],
		s_net_7_data[1]
	);

	nn2fpga::shift_op <
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		c_node_conv_3_ich,
		c_node_conv_3_och,
		c_node_conv_3_ih,
		c_node_conv_3_iw,
		c_node_conv_3_oh,
		c_node_conv_3_ow,
		c_node_conv_3_fh,
		c_node_conv_3_fw,
		c_node_conv_3_stride,
		c_node_conv_3_pad,
		2,
		1,
		c_node_conv_3_ow_ops,
		8,
		8>
	(
		s_net_7_data[0],
		s_net_7_pre_pad[2],
		s_net_7_data[2]
	);

	nn2fpga::shift_op <
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		c_node_conv_3_ich,
		c_node_conv_3_och,
		c_node_conv_3_ih,
		c_node_conv_3_iw,
		c_node_conv_3_oh,
		c_node_conv_3_ow,
		c_node_conv_3_fh,
		c_node_conv_3_fw,
		c_node_conv_3_stride,
		c_node_conv_3_pad,
		2,
		0,
		c_node_conv_3_ow_ops,
		8,
		8>
	(
		s_net_7_data[1],
		s_net_7_pre_pad[3],
		s_net_7_data[3]
	);

	nn2fpga::shift_op <
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		c_node_conv_3_ich,
		c_node_conv_3_och,
		c_node_conv_3_ih,
		c_node_conv_3_iw,
		c_node_conv_3_oh,
		c_node_conv_3_ow,
		c_node_conv_3_fh,
		c_node_conv_3_fw,
		c_node_conv_3_stride,
		c_node_conv_3_pad,
		1,
		3,
		c_node_conv_3_ow_ops,
		8,
		8>
	(
		s_net_7_data[2],
		s_net_7_pre_pad[4],
		s_net_7_data[4]
	);

	nn2fpga::shift_op <
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		c_node_conv_3_ich,
		c_node_conv_3_och,
		c_node_conv_3_ih,
		c_node_conv_3_iw,
		c_node_conv_3_oh,
		c_node_conv_3_ow,
		c_node_conv_3_fh,
		c_node_conv_3_fw,
		c_node_conv_3_stride,
		c_node_conv_3_pad,
		1,
		2,
		c_node_conv_3_ow_ops,
		8,
		8>
	(
		s_net_7_data[3],
		s_net_7_pre_pad[5],
		s_net_7_data[5]
	);

	nn2fpga::shift_op <
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		c_node_conv_3_ich,
		c_node_conv_3_och,
		c_node_conv_3_ih,
		c_node_conv_3_iw,
		c_node_conv_3_oh,
		c_node_conv_3_ow,
		c_node_conv_3_fh,
		c_node_conv_3_fw,
		c_node_conv_3_stride,
		c_node_conv_3_pad,
		1,
		1,
		c_node_conv_3_ow_ops,
		8,
		8>
	(
		s_net_7_data[4],
		s_net_7_pre_pad[6],
		s_net_7_data[6]
	);

	nn2fpga::shift_op <
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		c_node_conv_3_ich,
		c_node_conv_3_och,
		c_node_conv_3_ih,
		c_node_conv_3_iw,
		c_node_conv_3_oh,
		c_node_conv_3_ow,
		c_node_conv_3_fh,
		c_node_conv_3_fw,
		c_node_conv_3_stride,
		c_node_conv_3_pad,
		1,
		0,
		c_node_conv_3_ow_ops,
		8,
		8>
	(
		s_net_7_data[5],
		s_net_7_pre_pad[7],
		s_net_7_data[7]
	);

	nn2fpga::shift_op <
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		c_node_conv_3_ich,
		c_node_conv_3_och,
		c_node_conv_3_ih,
		c_node_conv_3_iw,
		c_node_conv_3_oh,
		c_node_conv_3_ow,
		c_node_conv_3_fh,
		c_node_conv_3_fw,
		c_node_conv_3_stride,
		c_node_conv_3_pad,
		0,
		3,
		c_node_conv_3_ow_ops,
		8,
		8>
	(
		s_net_7_data[6],
		s_net_7_pre_pad[8],
		s_net_7_data[8]
	);

	nn2fpga::shift_op <
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		c_node_conv_3_ich,
		c_node_conv_3_och,
		c_node_conv_3_ih,
		c_node_conv_3_iw,
		c_node_conv_3_oh,
		c_node_conv_3_ow,
		c_node_conv_3_fh,
		c_node_conv_3_fw,
		c_node_conv_3_stride,
		c_node_conv_3_pad,
		0,
		2,
		c_node_conv_3_ow_ops,
		8,
		8>
	(
		s_net_7_data[7],
		s_net_7_pre_pad[9],
		s_net_7_data[9]
	);

	nn2fpga::shift_op <
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		std::nullptr_t,
		c_node_conv_3_ich,
		c_node_conv_3_och,
		c_node_conv_3_ih,
		c_node_conv_3_iw,
		c_node_conv_3_oh,
		c_node_conv_3_ow,
		c_node_conv_3_fh,
		c_node_conv_3_fw,
		c_node_conv_3_stride,
		c_node_conv_3_pad,
		0,
		1,
		c_node_conv_3_ow_ops,
		8,
		8>
	(
		s_net_7_data[8],
		s_net_7_pre_pad[10],
		s_net_7_null[1]
	);

	nn2fpga::shift_op <
		t_net_7_lb_struct,
		t_net_7_lb_struct,
		std::nullptr_t,
		c_node_conv_3_ich,
		c_node_conv_3_och,
		c_node_conv_3_ih,
		c_node_conv_3_iw,
		c_node_conv_3_oh,
		c_node_conv_3_ow,
		c_node_conv_3_fh,
		c_node_conv_3_fw,
		c_node_conv_3_stride,
		c_node_conv_3_pad,
		0,
		0,
		c_node_conv_3_ow_ops,
		8,
		8>
	(
		s_net_7_data[9],
		s_net_7_pre_pad[11],
		s_net_7_null[0]
	);

	nn2fpga::pad_input <
		t_net_7_lb_struct,
		t_net_7_window_struct,
		c_node_conv_3_ich,
		c_node_conv_3_ih,
		c_node_conv_3_iw,
		c_node_conv_3_fh,
		c_node_conv_3_fw,
		c_node_conv_3_stride,
		c_node_conv_3_pad,
		c_node_conv_3_ow_ops,
		8,
		8>
	(
		s_net_7_pre_pad,
		s_net_7_compute
	);

	nn2fpga::conv_comp_wrap <
		t_net_7_window_struct,
		t_net_7_window,
		t_net_7_reduce,
		t_net_7,
		t_node_conv_3_weight,
		t_node_conv_3_weight_mem,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_7_mod,
		t_net_7,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_8_acc_struct,
		t_net_8_acc,
		std::nullptr_t,
		std::nullptr_t,
		t_net_8_struct,
		t_net_8_vector,
		t_net_8,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_params_stream,
		c_node_conv_3_ich,
		c_node_conv_3_ich,
		c_node_conv_3_och,
		c_node_conv_3_och_1x1,
		c_node_conv_3_oh,
		c_node_conv_3_ow,
		c_node_conv_3_fh,
		c_node_conv_3_fw,
		c_node_conv_3_index,
		c_node_conv_3_stride,
		c_node_conv_3_ops,
		c_node_conv_3_ops_out,
		c_node_conv_3_ich_ops,
		1,		//add ops
		c_node_conv_3_ow_ops,
		c_node_conv_3_ow_ops_out,
		8,		//bias_ops
		c_node_conv_3_reuse,
		c_node_conv_3_ow_pack,
		c_node_conv_3_och_pack,
		8,		//act bits
		3,		//act integer bits
		8,		//weight bits
		2,		//weight integer bits
		2,		//guard bits
		16,		//accumulators in parallel
		15,		//mask bits
		0,		//act bits 1x1
		0,		//act integer bits 1x1
		0,		//weight bits 1x1
		0,		//weight integer bits 1x1
		2,		//guard bits 1x1
		1,		//accumulators in parallel 1x1
		0,		//mask bits 1x1
		8644608,		//shift cycles
		8,		//width stream
		1,		//weight reads per data
		0,		//bias reads per data
		0,		//weight reads per data 1x1
		0,		//bias reads per data 1x1
		c_node_conv_3_relu,
		0>		//depth
	(
		s_node_conv_1_out,
		s_node_conv_3_init_flag,
		s_node_conv_3_out,
		nullptr,
		c_node_conv_3_weight,
		nullptr,
		nullptr,
		s_net_7_compute,
		(hls::stream<std::nullptr_t>*)(nullptr),
		(hls::stream<std::nullptr_t>*)(nullptr),
		s_net_8,
		(hls::stream<std::nullptr_t>*)(nullptr)
	);

	nn2fpga::bandwidth_adjust <
		t_net_8_struct,
		t_net_8_adj_struct,
		c_bandwidth_adjust_net_8_ich,
		c_bandwidth_adjust_net_8_iw,
		c_bandwidth_adjust_net_8_ih,
		c_bandwidth_adjust_net_8_ow_ops_in,
		c_bandwidth_adjust_net_8_ow_ops,
		c_bandwidth_adjust_net_8_old_in_ops,
		c_bandwidth_adjust_net_8_in_ops,
		false>		//skip connection flag
	(
		s_net_8,
		s_net_8_adj
	);

	nn2fpga::shift_op <
		t_net_8_adj_struct,
		t_net_8_lb_struct,
		t_net_8_lb_struct,
		c_node_pool_4_ich,
		c_node_pool_4_och,
		c_node_pool_4_ih,
		c_node_pool_4_iw,
		c_node_pool_4_oh,
		c_node_pool_4_ow,
		c_node_pool_4_fh,
		c_node_pool_4_fw,
		c_node_pool_4_stride,
		c_node_pool_4_pad,
		1,
		1,
		c_node_pool_4_ow_ops,
		16,
		16>
	(
		s_net_8_adj[0],
		s_net_8_pre_pad[0],
		s_net_8_data[0]
	);

	nn2fpga::shift_op <
		t_net_8_lb_struct,
		t_net_8_lb_struct,
		t_net_8_lb_struct,
		c_node_pool_4_ich,
		c_node_pool_4_och,
		c_node_pool_4_ih,
		c_node_pool_4_iw,
		c_node_pool_4_oh,
		c_node_pool_4_ow,
		c_node_pool_4_fh,
		c_node_pool_4_fw,
		c_node_pool_4_stride,
		c_node_pool_4_pad,
		1,
		0,
		c_node_pool_4_ow_ops,
		16,
		16>
	(
		s_net_8_data[0],
		s_net_8_pre_pad[1],
		s_net_8_data[1]
	);

	nn2fpga::shift_op <
		t_net_8_lb_struct,
		t_net_8_lb_struct,
		t_net_8_lb_struct,
		c_node_pool_4_ich,
		c_node_pool_4_och,
		c_node_pool_4_ih,
		c_node_pool_4_iw,
		c_node_pool_4_oh,
		c_node_pool_4_ow,
		c_node_pool_4_fh,
		c_node_pool_4_fw,
		c_node_pool_4_stride,
		c_node_pool_4_pad,
		0,
		1,
		c_node_pool_4_ow_ops,
		16,
		16>
	(
		s_net_8_data[1],
		s_net_8_pre_pad[2],
		s_net_8_data[2]
	);

	nn2fpga::shift_op <
		t_net_8_lb_struct,
		t_net_8_lb_struct,
		std::nullptr_t,
		c_node_pool_4_ich,
		c_node_pool_4_och,
		c_node_pool_4_ih,
		c_node_pool_4_iw,
		c_node_pool_4_oh,
		c_node_pool_4_ow,
		c_node_pool_4_fh,
		c_node_pool_4_fw,
		c_node_pool_4_stride,
		c_node_pool_4_pad,
		0,
		0,
		c_node_pool_4_ow_ops,
		16,
		16>
	(
		s_net_8_data[2],
		s_net_8_pre_pad[3],
		s_net_8_null[0]
	);

	nn2fpga::pad_input <
		t_net_8_lb_struct,
		t_net_8_window_struct,
		c_node_pool_4_ich,
		c_node_pool_4_ih,
		c_node_pool_4_iw,
		c_node_pool_4_fh,
		c_node_pool_4_fw,
		c_node_pool_4_stride,
		c_node_pool_4_pad,
		c_node_pool_4_ow_ops,
		16,
		16>
	(
		s_net_8_pre_pad,
		s_net_8_compute
	);

	nn2fpga::pool_op <
		t_net_8_window_struct,
		t_net_8_window,
		t_net_9_struct,
		t_net_9,
		t_node_pool_4_acc,
		t_node_pool_4_acc,
		c_node_pool_4_ich,
		c_node_pool_4_och,
		c_node_pool_4_ih,
		c_node_pool_4_iw,
		c_node_pool_4_oh,
		c_node_pool_4_ow,
		c_node_pool_4_fw,
		c_node_pool_4_fh,
		c_node_pool_4_stride,
		c_node_pool_4_pad,
		c_node_pool_4_pool,
		c_node_pool_4_ow_ops,
		c_node_pool_4_ops,
		c_node_pool_4_in_ops>
	(
		s_net_8_compute,
		s_net_9
	);

	nn2fpga::bandwidth_adjust <
		t_net_9_struct,
		t_net_9_adj_struct,
		c_bandwidth_adjust_net_9_ich,
		c_bandwidth_adjust_net_9_iw,
		c_bandwidth_adjust_net_9_ih,
		c_bandwidth_adjust_net_9_ow_ops_in,
		c_bandwidth_adjust_net_9_ow_ops,
		c_bandwidth_adjust_net_9_old_in_ops,
		c_bandwidth_adjust_net_9_in_ops,
		false>		//skip connection flag
	(
		s_net_9,
		s_net_9_adj
	);

	nn2fpga::shift_op <
		t_net_9_adj_struct,
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		c_node_conv_5_ich,
		c_node_conv_5_och,
		c_node_conv_5_ih,
		c_node_conv_5_iw,
		c_node_conv_5_oh,
		c_node_conv_5_ow,
		c_node_conv_5_fh,
		c_node_conv_5_fw,
		c_node_conv_5_stride,
		c_node_conv_5_pad,
		2,
		3,
		c_node_conv_5_ow_ops,
		16,
		8>
	(
		s_net_9_adj[0],
		s_net_9_pre_pad[0],
		s_net_9_data[0]
	);

	nn2fpga::shift_op <
		t_net_9_adj_struct,
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		c_node_conv_5_ich,
		c_node_conv_5_och,
		c_node_conv_5_ih,
		c_node_conv_5_iw,
		c_node_conv_5_oh,
		c_node_conv_5_ow,
		c_node_conv_5_fh,
		c_node_conv_5_fw,
		c_node_conv_5_stride,
		c_node_conv_5_pad,
		2,
		2,
		c_node_conv_5_ow_ops,
		16,
		8>
	(
		s_net_9_adj[1],
		s_net_9_pre_pad[1],
		s_net_9_data[1]
	);

	nn2fpga::shift_op <
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		c_node_conv_5_ich,
		c_node_conv_5_och,
		c_node_conv_5_ih,
		c_node_conv_5_iw,
		c_node_conv_5_oh,
		c_node_conv_5_ow,
		c_node_conv_5_fh,
		c_node_conv_5_fw,
		c_node_conv_5_stride,
		c_node_conv_5_pad,
		2,
		1,
		c_node_conv_5_ow_ops,
		8,
		8>
	(
		s_net_9_data[0],
		s_net_9_pre_pad[2],
		s_net_9_data[2]
	);

	nn2fpga::shift_op <
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		c_node_conv_5_ich,
		c_node_conv_5_och,
		c_node_conv_5_ih,
		c_node_conv_5_iw,
		c_node_conv_5_oh,
		c_node_conv_5_ow,
		c_node_conv_5_fh,
		c_node_conv_5_fw,
		c_node_conv_5_stride,
		c_node_conv_5_pad,
		2,
		0,
		c_node_conv_5_ow_ops,
		8,
		8>
	(
		s_net_9_data[1],
		s_net_9_pre_pad[3],
		s_net_9_data[3]
	);

	nn2fpga::shift_op <
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		c_node_conv_5_ich,
		c_node_conv_5_och,
		c_node_conv_5_ih,
		c_node_conv_5_iw,
		c_node_conv_5_oh,
		c_node_conv_5_ow,
		c_node_conv_5_fh,
		c_node_conv_5_fw,
		c_node_conv_5_stride,
		c_node_conv_5_pad,
		1,
		3,
		c_node_conv_5_ow_ops,
		8,
		8>
	(
		s_net_9_data[2],
		s_net_9_pre_pad[4],
		s_net_9_data[4]
	);

	nn2fpga::shift_op <
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		c_node_conv_5_ich,
		c_node_conv_5_och,
		c_node_conv_5_ih,
		c_node_conv_5_iw,
		c_node_conv_5_oh,
		c_node_conv_5_ow,
		c_node_conv_5_fh,
		c_node_conv_5_fw,
		c_node_conv_5_stride,
		c_node_conv_5_pad,
		1,
		2,
		c_node_conv_5_ow_ops,
		8,
		8>
	(
		s_net_9_data[3],
		s_net_9_pre_pad[5],
		s_net_9_data[5]
	);

	nn2fpga::shift_op <
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		c_node_conv_5_ich,
		c_node_conv_5_och,
		c_node_conv_5_ih,
		c_node_conv_5_iw,
		c_node_conv_5_oh,
		c_node_conv_5_ow,
		c_node_conv_5_fh,
		c_node_conv_5_fw,
		c_node_conv_5_stride,
		c_node_conv_5_pad,
		1,
		1,
		c_node_conv_5_ow_ops,
		8,
		8>
	(
		s_net_9_data[4],
		s_net_9_pre_pad[6],
		s_net_9_data[6]
	);

	nn2fpga::shift_op <
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		c_node_conv_5_ich,
		c_node_conv_5_och,
		c_node_conv_5_ih,
		c_node_conv_5_iw,
		c_node_conv_5_oh,
		c_node_conv_5_ow,
		c_node_conv_5_fh,
		c_node_conv_5_fw,
		c_node_conv_5_stride,
		c_node_conv_5_pad,
		1,
		0,
		c_node_conv_5_ow_ops,
		8,
		8>
	(
		s_net_9_data[5],
		s_net_9_pre_pad[7],
		s_net_9_data[7]
	);

	nn2fpga::shift_op <
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		c_node_conv_5_ich,
		c_node_conv_5_och,
		c_node_conv_5_ih,
		c_node_conv_5_iw,
		c_node_conv_5_oh,
		c_node_conv_5_ow,
		c_node_conv_5_fh,
		c_node_conv_5_fw,
		c_node_conv_5_stride,
		c_node_conv_5_pad,
		0,
		3,
		c_node_conv_5_ow_ops,
		8,
		8>
	(
		s_net_9_data[6],
		s_net_9_pre_pad[8],
		s_net_9_data[8]
	);

	nn2fpga::shift_op <
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		c_node_conv_5_ich,
		c_node_conv_5_och,
		c_node_conv_5_ih,
		c_node_conv_5_iw,
		c_node_conv_5_oh,
		c_node_conv_5_ow,
		c_node_conv_5_fh,
		c_node_conv_5_fw,
		c_node_conv_5_stride,
		c_node_conv_5_pad,
		0,
		2,
		c_node_conv_5_ow_ops,
		8,
		8>
	(
		s_net_9_data[7],
		s_net_9_pre_pad[9],
		s_net_9_data[9]
	);

	nn2fpga::shift_op <
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		std::nullptr_t,
		c_node_conv_5_ich,
		c_node_conv_5_och,
		c_node_conv_5_ih,
		c_node_conv_5_iw,
		c_node_conv_5_oh,
		c_node_conv_5_ow,
		c_node_conv_5_fh,
		c_node_conv_5_fw,
		c_node_conv_5_stride,
		c_node_conv_5_pad,
		0,
		1,
		c_node_conv_5_ow_ops,
		8,
		8>
	(
		s_net_9_data[8],
		s_net_9_pre_pad[10],
		s_net_9_null[1]
	);

	nn2fpga::shift_op <
		t_net_9_lb_struct,
		t_net_9_lb_struct,
		std::nullptr_t,
		c_node_conv_5_ich,
		c_node_conv_5_och,
		c_node_conv_5_ih,
		c_node_conv_5_iw,
		c_node_conv_5_oh,
		c_node_conv_5_ow,
		c_node_conv_5_fh,
		c_node_conv_5_fw,
		c_node_conv_5_stride,
		c_node_conv_5_pad,
		0,
		0,
		c_node_conv_5_ow_ops,
		8,
		8>
	(
		s_net_9_data[9],
		s_net_9_pre_pad[11],
		s_net_9_null[0]
	);

	nn2fpga::pad_input <
		t_net_9_lb_struct,
		t_net_9_window_struct,
		c_node_conv_5_ich,
		c_node_conv_5_ih,
		c_node_conv_5_iw,
		c_node_conv_5_fh,
		c_node_conv_5_fw,
		c_node_conv_5_stride,
		c_node_conv_5_pad,
		c_node_conv_5_ow_ops,
		8,
		8>
	(
		s_net_9_pre_pad,
		s_net_9_compute
	);

	nn2fpga::conv_comp_wrap <
		t_net_9_window_struct,
		t_net_9_window,
		t_net_9_reduce,
		t_net_9,
		t_node_conv_5_weight,
		t_node_conv_5_weight_mem,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_9_mod,
		t_net_9,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_10_acc_struct,
		t_net_10_acc,
		std::nullptr_t,
		std::nullptr_t,
		t_net_10_struct,
		t_net_10_vector,
		t_net_10,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_params_stream,
		c_node_conv_5_ich,
		c_node_conv_5_ich,
		c_node_conv_5_och,
		c_node_conv_5_och_1x1,
		c_node_conv_5_oh,
		c_node_conv_5_ow,
		c_node_conv_5_fh,
		c_node_conv_5_fw,
		c_node_conv_5_index,
		c_node_conv_5_stride,
		c_node_conv_5_ops,
		c_node_conv_5_ops_out,
		c_node_conv_5_ich_ops,
		1,		//add ops
		c_node_conv_5_ow_ops,
		c_node_conv_5_ow_ops_out,
		8,		//bias_ops
		c_node_conv_5_reuse,
		c_node_conv_5_ow_pack,
		c_node_conv_5_och_pack,
		8,		//act bits
		3,		//act integer bits
		8,		//weight bits
		1,		//weight integer bits
		2,		//guard bits
		16,		//accumulators in parallel
		15,		//mask bits
		0,		//act bits 1x1
		0,		//act integer bits 1x1
		0,		//weight bits 1x1
		0,		//weight integer bits 1x1
		2,		//guard bits 1x1
		1,		//accumulators in parallel 1x1
		0,		//mask bits 1x1
		8626176,		//shift cycles
		8,		//width stream
		1,		//weight reads per data
		0,		//bias reads per data
		0,		//weight reads per data 1x1
		0,		//bias reads per data 1x1
		c_node_conv_5_relu,
		0>		//depth
	(
		s_node_conv_3_out,
		s_node_conv_5_init_flag,
		s_node_conv_5_out,
		nullptr,
		c_node_conv_5_weight,
		nullptr,
		nullptr,
		s_net_9_compute,
		(hls::stream<std::nullptr_t>*)(nullptr),
		(hls::stream<std::nullptr_t>*)(nullptr),
		s_net_10,
		(hls::stream<std::nullptr_t>*)(nullptr)
	);

	nn2fpga::bandwidth_adjust <
		t_net_10_struct,
		t_net_10_adj_struct,
		c_bandwidth_adjust_net_10_ich,
		c_bandwidth_adjust_net_10_iw,
		c_bandwidth_adjust_net_10_ih,
		c_bandwidth_adjust_net_10_ow_ops_in,
		c_bandwidth_adjust_net_10_ow_ops,
		c_bandwidth_adjust_net_10_old_in_ops,
		c_bandwidth_adjust_net_10_in_ops,
		false>		//skip connection flag
	(
		s_net_10,
		s_net_10_adj
	);

	nn2fpga::shift_op <
		t_net_10_adj_struct,
		t_net_10_lb_struct,
		t_net_10_lb_struct,
		c_node_pool_6_ich,
		c_node_pool_6_och,
		c_node_pool_6_ih,
		c_node_pool_6_iw,
		c_node_pool_6_oh,
		c_node_pool_6_ow,
		c_node_pool_6_fh,
		c_node_pool_6_fw,
		c_node_pool_6_stride,
		c_node_pool_6_pad,
		1,
		1,
		c_node_pool_6_ow_ops,
		8,
		8>
	(
		s_net_10_adj[0],
		s_net_10_pre_pad[0],
		s_net_10_data[0]
	);

	nn2fpga::shift_op <
		t_net_10_lb_struct,
		t_net_10_lb_struct,
		t_net_10_lb_struct,
		c_node_pool_6_ich,
		c_node_pool_6_och,
		c_node_pool_6_ih,
		c_node_pool_6_iw,
		c_node_pool_6_oh,
		c_node_pool_6_ow,
		c_node_pool_6_fh,
		c_node_pool_6_fw,
		c_node_pool_6_stride,
		c_node_pool_6_pad,
		1,
		0,
		c_node_pool_6_ow_ops,
		8,
		8>
	(
		s_net_10_data[0],
		s_net_10_pre_pad[1],
		s_net_10_data[1]
	);

	nn2fpga::shift_op <
		t_net_10_lb_struct,
		t_net_10_lb_struct,
		t_net_10_lb_struct,
		c_node_pool_6_ich,
		c_node_pool_6_och,
		c_node_pool_6_ih,
		c_node_pool_6_iw,
		c_node_pool_6_oh,
		c_node_pool_6_ow,
		c_node_pool_6_fh,
		c_node_pool_6_fw,
		c_node_pool_6_stride,
		c_node_pool_6_pad,
		0,
		1,
		c_node_pool_6_ow_ops,
		8,
		8>
	(
		s_net_10_data[1],
		s_net_10_pre_pad[2],
		s_net_10_data[2]
	);

	nn2fpga::shift_op <
		t_net_10_lb_struct,
		t_net_10_lb_struct,
		std::nullptr_t,
		c_node_pool_6_ich,
		c_node_pool_6_och,
		c_node_pool_6_ih,
		c_node_pool_6_iw,
		c_node_pool_6_oh,
		c_node_pool_6_ow,
		c_node_pool_6_fh,
		c_node_pool_6_fw,
		c_node_pool_6_stride,
		c_node_pool_6_pad,
		0,
		0,
		c_node_pool_6_ow_ops,
		8,
		8>
	(
		s_net_10_data[2],
		s_net_10_pre_pad[3],
		s_net_10_null[0]
	);

	nn2fpga::pad_input <
		t_net_10_lb_struct,
		t_net_10_window_struct,
		c_node_pool_6_ich,
		c_node_pool_6_ih,
		c_node_pool_6_iw,
		c_node_pool_6_fh,
		c_node_pool_6_fw,
		c_node_pool_6_stride,
		c_node_pool_6_pad,
		c_node_pool_6_ow_ops,
		8,
		8>
	(
		s_net_10_pre_pad,
		s_net_10_compute
	);

	nn2fpga::pool_op <
		t_net_10_window_struct,
		t_net_10_window,
		t_net_11_struct,
		t_net_11,
		t_node_pool_6_acc,
		t_node_pool_6_acc,
		c_node_pool_6_ich,
		c_node_pool_6_och,
		c_node_pool_6_ih,
		c_node_pool_6_iw,
		c_node_pool_6_oh,
		c_node_pool_6_ow,
		c_node_pool_6_fw,
		c_node_pool_6_fh,
		c_node_pool_6_stride,
		c_node_pool_6_pad,
		c_node_pool_6_pool,
		c_node_pool_6_ow_ops,
		c_node_pool_6_ops,
		c_node_pool_6_in_ops>
	(
		s_net_10_compute,
		s_net_11
	);

	nn2fpga::bandwidth_adjust <
		t_net_11_struct,
		t_net_11_adj_struct,
		c_bandwidth_adjust_net_11_ich,
		c_bandwidth_adjust_net_11_iw,
		c_bandwidth_adjust_net_11_ih,
		c_bandwidth_adjust_net_11_ow_ops_in,
		c_bandwidth_adjust_net_11_ow_ops,
		c_bandwidth_adjust_net_11_old_in_ops,
		c_bandwidth_adjust_net_11_in_ops,
		false>		//skip connection flag
	(
		s_net_11,
		s_net_11_adj
	);

	nn2fpga::shift_op <
		t_net_11_adj_struct,
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		c_node_conv_7_ich,
		c_node_conv_7_och,
		c_node_conv_7_ih,
		c_node_conv_7_iw,
		c_node_conv_7_oh,
		c_node_conv_7_ow,
		c_node_conv_7_fh,
		c_node_conv_7_fw,
		c_node_conv_7_stride,
		c_node_conv_7_pad,
		2,
		3,
		c_node_conv_7_ow_ops,
		8,
		8>
	(
		s_net_11_adj[0],
		s_net_11_pre_pad[0],
		s_net_11_data[0]
	);

	nn2fpga::shift_op <
		t_net_11_adj_struct,
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		c_node_conv_7_ich,
		c_node_conv_7_och,
		c_node_conv_7_ih,
		c_node_conv_7_iw,
		c_node_conv_7_oh,
		c_node_conv_7_ow,
		c_node_conv_7_fh,
		c_node_conv_7_fw,
		c_node_conv_7_stride,
		c_node_conv_7_pad,
		2,
		2,
		c_node_conv_7_ow_ops,
		8,
		8>
	(
		s_net_11_adj[1],
		s_net_11_pre_pad[1],
		s_net_11_data[1]
	);

	nn2fpga::shift_op <
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		c_node_conv_7_ich,
		c_node_conv_7_och,
		c_node_conv_7_ih,
		c_node_conv_7_iw,
		c_node_conv_7_oh,
		c_node_conv_7_ow,
		c_node_conv_7_fh,
		c_node_conv_7_fw,
		c_node_conv_7_stride,
		c_node_conv_7_pad,
		2,
		1,
		c_node_conv_7_ow_ops,
		8,
		8>
	(
		s_net_11_data[0],
		s_net_11_pre_pad[2],
		s_net_11_data[2]
	);

	nn2fpga::shift_op <
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		c_node_conv_7_ich,
		c_node_conv_7_och,
		c_node_conv_7_ih,
		c_node_conv_7_iw,
		c_node_conv_7_oh,
		c_node_conv_7_ow,
		c_node_conv_7_fh,
		c_node_conv_7_fw,
		c_node_conv_7_stride,
		c_node_conv_7_pad,
		2,
		0,
		c_node_conv_7_ow_ops,
		8,
		8>
	(
		s_net_11_data[1],
		s_net_11_pre_pad[3],
		s_net_11_data[3]
	);

	nn2fpga::shift_op <
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		c_node_conv_7_ich,
		c_node_conv_7_och,
		c_node_conv_7_ih,
		c_node_conv_7_iw,
		c_node_conv_7_oh,
		c_node_conv_7_ow,
		c_node_conv_7_fh,
		c_node_conv_7_fw,
		c_node_conv_7_stride,
		c_node_conv_7_pad,
		1,
		3,
		c_node_conv_7_ow_ops,
		8,
		8>
	(
		s_net_11_data[2],
		s_net_11_pre_pad[4],
		s_net_11_data[4]
	);

	nn2fpga::shift_op <
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		c_node_conv_7_ich,
		c_node_conv_7_och,
		c_node_conv_7_ih,
		c_node_conv_7_iw,
		c_node_conv_7_oh,
		c_node_conv_7_ow,
		c_node_conv_7_fh,
		c_node_conv_7_fw,
		c_node_conv_7_stride,
		c_node_conv_7_pad,
		1,
		2,
		c_node_conv_7_ow_ops,
		8,
		8>
	(
		s_net_11_data[3],
		s_net_11_pre_pad[5],
		s_net_11_data[5]
	);

	nn2fpga::shift_op <
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		c_node_conv_7_ich,
		c_node_conv_7_och,
		c_node_conv_7_ih,
		c_node_conv_7_iw,
		c_node_conv_7_oh,
		c_node_conv_7_ow,
		c_node_conv_7_fh,
		c_node_conv_7_fw,
		c_node_conv_7_stride,
		c_node_conv_7_pad,
		1,
		1,
		c_node_conv_7_ow_ops,
		8,
		8>
	(
		s_net_11_data[4],
		s_net_11_pre_pad[6],
		s_net_11_data[6]
	);

	nn2fpga::shift_op <
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		c_node_conv_7_ich,
		c_node_conv_7_och,
		c_node_conv_7_ih,
		c_node_conv_7_iw,
		c_node_conv_7_oh,
		c_node_conv_7_ow,
		c_node_conv_7_fh,
		c_node_conv_7_fw,
		c_node_conv_7_stride,
		c_node_conv_7_pad,
		1,
		0,
		c_node_conv_7_ow_ops,
		8,
		8>
	(
		s_net_11_data[5],
		s_net_11_pre_pad[7],
		s_net_11_data[7]
	);

	nn2fpga::shift_op <
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		c_node_conv_7_ich,
		c_node_conv_7_och,
		c_node_conv_7_ih,
		c_node_conv_7_iw,
		c_node_conv_7_oh,
		c_node_conv_7_ow,
		c_node_conv_7_fh,
		c_node_conv_7_fw,
		c_node_conv_7_stride,
		c_node_conv_7_pad,
		0,
		3,
		c_node_conv_7_ow_ops,
		8,
		8>
	(
		s_net_11_data[6],
		s_net_11_pre_pad[8],
		s_net_11_data[8]
	);

	nn2fpga::shift_op <
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		c_node_conv_7_ich,
		c_node_conv_7_och,
		c_node_conv_7_ih,
		c_node_conv_7_iw,
		c_node_conv_7_oh,
		c_node_conv_7_ow,
		c_node_conv_7_fh,
		c_node_conv_7_fw,
		c_node_conv_7_stride,
		c_node_conv_7_pad,
		0,
		2,
		c_node_conv_7_ow_ops,
		8,
		8>
	(
		s_net_11_data[7],
		s_net_11_pre_pad[9],
		s_net_11_data[9]
	);

	nn2fpga::shift_op <
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		std::nullptr_t,
		c_node_conv_7_ich,
		c_node_conv_7_och,
		c_node_conv_7_ih,
		c_node_conv_7_iw,
		c_node_conv_7_oh,
		c_node_conv_7_ow,
		c_node_conv_7_fh,
		c_node_conv_7_fw,
		c_node_conv_7_stride,
		c_node_conv_7_pad,
		0,
		1,
		c_node_conv_7_ow_ops,
		8,
		8>
	(
		s_net_11_data[8],
		s_net_11_pre_pad[10],
		s_net_11_null[1]
	);

	nn2fpga::shift_op <
		t_net_11_lb_struct,
		t_net_11_lb_struct,
		std::nullptr_t,
		c_node_conv_7_ich,
		c_node_conv_7_och,
		c_node_conv_7_ih,
		c_node_conv_7_iw,
		c_node_conv_7_oh,
		c_node_conv_7_ow,
		c_node_conv_7_fh,
		c_node_conv_7_fw,
		c_node_conv_7_stride,
		c_node_conv_7_pad,
		0,
		0,
		c_node_conv_7_ow_ops,
		8,
		8>
	(
		s_net_11_data[9],
		s_net_11_pre_pad[11],
		s_net_11_null[0]
	);

	nn2fpga::pad_input <
		t_net_11_lb_struct,
		t_net_11_window_struct,
		c_node_conv_7_ich,
		c_node_conv_7_ih,
		c_node_conv_7_iw,
		c_node_conv_7_fh,
		c_node_conv_7_fw,
		c_node_conv_7_stride,
		c_node_conv_7_pad,
		c_node_conv_7_ow_ops,
		8,
		8>
	(
		s_net_11_pre_pad,
		s_net_11_compute
	);

	nn2fpga::conv_comp_wrap <
		t_net_11_window_struct,
		t_net_11_window,
		t_net_11_reduce,
		t_net_11,
		t_node_conv_7_weight,
		t_node_conv_7_weight_mem,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_11_mod,
		t_net_11,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_12_acc_struct,
		t_net_12_acc,
		std::nullptr_t,
		std::nullptr_t,
		t_net_12_struct,
		t_net_12_vector,
		t_net_12,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_params_stream,
		c_node_conv_7_ich,
		c_node_conv_7_ich,
		c_node_conv_7_och,
		c_node_conv_7_och_1x1,
		c_node_conv_7_oh,
		c_node_conv_7_ow,
		c_node_conv_7_fh,
		c_node_conv_7_fw,
		c_node_conv_7_index,
		c_node_conv_7_stride,
		c_node_conv_7_ops,
		c_node_conv_7_ops_out,
		c_node_conv_7_ich_ops,
		1,		//add ops
		c_node_conv_7_ow_ops,
		c_node_conv_7_ow_ops_out,
		8,		//bias_ops
		c_node_conv_7_reuse,
		c_node_conv_7_ow_pack,
		c_node_conv_7_och_pack,
		8,		//act bits
		3,		//act integer bits
		8,		//weight bits
		1,		//weight integer bits
		2,		//guard bits
		16,		//accumulators in parallel
		15,		//mask bits
		0,		//act bits 1x1
		0,		//act integer bits 1x1
		0,		//weight bits 1x1
		0,		//weight integer bits 1x1
		2,		//guard bits 1x1
		1,		//accumulators in parallel 1x1
		0,		//mask bits 1x1
		8552448,		//shift cycles
		8,		//width stream
		1,		//weight reads per data
		0,		//bias reads per data
		0,		//weight reads per data 1x1
		0,		//bias reads per data 1x1
		c_node_conv_7_relu,
		0>		//depth
	(
		s_node_conv_5_out,
		s_node_conv_7_init_flag,
		s_node_conv_7_out,
		nullptr,
		c_node_conv_7_weight,
		nullptr,
		nullptr,
		s_net_11_compute,
		(hls::stream<std::nullptr_t>*)(nullptr),
		(hls::stream<std::nullptr_t>*)(nullptr),
		s_net_12,
		(hls::stream<std::nullptr_t>*)(nullptr)
	);

	nn2fpga::bandwidth_adjust <
		t_net_12_struct,
		t_net_12_adj_struct,
		c_bandwidth_adjust_net_12_ich,
		c_bandwidth_adjust_net_12_iw,
		c_bandwidth_adjust_net_12_ih,
		c_bandwidth_adjust_net_12_ow_ops_in,
		c_bandwidth_adjust_net_12_ow_ops,
		c_bandwidth_adjust_net_12_old_in_ops,
		c_bandwidth_adjust_net_12_in_ops,
		false>		//skip connection flag
	(
		s_net_12,
		s_net_12_adj
	);

	nn2fpga::shift_op <
		t_net_12_adj_struct,
		t_net_12_lb_struct,
		t_net_12_lb_struct,
		c_node_pool_8_ich,
		c_node_pool_8_och,
		c_node_pool_8_ih,
		c_node_pool_8_iw,
		c_node_pool_8_oh,
		c_node_pool_8_ow,
		c_node_pool_8_fh,
		c_node_pool_8_fw,
		c_node_pool_8_stride,
		c_node_pool_8_pad,
		1,
		1,
		c_node_pool_8_ow_ops,
		8,
		8>
	(
		s_net_12_adj[0],
		s_net_12_pre_pad[0],
		s_net_12_data[0]
	);

	nn2fpga::shift_op <
		t_net_12_lb_struct,
		t_net_12_lb_struct,
		t_net_12_lb_struct,
		c_node_pool_8_ich,
		c_node_pool_8_och,
		c_node_pool_8_ih,
		c_node_pool_8_iw,
		c_node_pool_8_oh,
		c_node_pool_8_ow,
		c_node_pool_8_fh,
		c_node_pool_8_fw,
		c_node_pool_8_stride,
		c_node_pool_8_pad,
		1,
		0,
		c_node_pool_8_ow_ops,
		8,
		8>
	(
		s_net_12_data[0],
		s_net_12_pre_pad[1],
		s_net_12_data[1]
	);

	nn2fpga::shift_op <
		t_net_12_lb_struct,
		t_net_12_lb_struct,
		t_net_12_lb_struct,
		c_node_pool_8_ich,
		c_node_pool_8_och,
		c_node_pool_8_ih,
		c_node_pool_8_iw,
		c_node_pool_8_oh,
		c_node_pool_8_ow,
		c_node_pool_8_fh,
		c_node_pool_8_fw,
		c_node_pool_8_stride,
		c_node_pool_8_pad,
		0,
		1,
		c_node_pool_8_ow_ops,
		8,
		8>
	(
		s_net_12_data[1],
		s_net_12_pre_pad[2],
		s_net_12_data[2]
	);

	nn2fpga::shift_op <
		t_net_12_lb_struct,
		t_net_12_lb_struct,
		std::nullptr_t,
		c_node_pool_8_ich,
		c_node_pool_8_och,
		c_node_pool_8_ih,
		c_node_pool_8_iw,
		c_node_pool_8_oh,
		c_node_pool_8_ow,
		c_node_pool_8_fh,
		c_node_pool_8_fw,
		c_node_pool_8_stride,
		c_node_pool_8_pad,
		0,
		0,
		c_node_pool_8_ow_ops,
		8,
		8>
	(
		s_net_12_data[2],
		s_net_12_pre_pad[3],
		s_net_12_null[0]
	);

	nn2fpga::pad_input <
		t_net_12_lb_struct,
		t_net_12_window_struct,
		c_node_pool_8_ich,
		c_node_pool_8_ih,
		c_node_pool_8_iw,
		c_node_pool_8_fh,
		c_node_pool_8_fw,
		c_node_pool_8_stride,
		c_node_pool_8_pad,
		c_node_pool_8_ow_ops,
		8,
		8>
	(
		s_net_12_pre_pad,
		s_net_12_compute
	);

	nn2fpga::pool_op <
		t_net_12_window_struct,
		t_net_12_window,
		t_net_13_struct,
		t_net_13,
		t_node_pool_8_acc,
		t_node_pool_8_acc,
		c_node_pool_8_ich,
		c_node_pool_8_och,
		c_node_pool_8_ih,
		c_node_pool_8_iw,
		c_node_pool_8_oh,
		c_node_pool_8_ow,
		c_node_pool_8_fw,
		c_node_pool_8_fh,
		c_node_pool_8_stride,
		c_node_pool_8_pad,
		c_node_pool_8_pool,
		c_node_pool_8_ow_ops,
		c_node_pool_8_ops,
		c_node_pool_8_in_ops>
	(
		s_net_12_compute,
		s_net_13
	);

	nn2fpga::bandwidth_adjust <
		t_net_13_struct,
		t_net_13_adj_struct,
		c_bandwidth_adjust_net_13_ich,
		c_bandwidth_adjust_net_13_iw,
		c_bandwidth_adjust_net_13_ih,
		c_bandwidth_adjust_net_13_ow_ops_in,
		c_bandwidth_adjust_net_13_ow_ops,
		c_bandwidth_adjust_net_13_old_in_ops,
		c_bandwidth_adjust_net_13_in_ops,
		false>		//skip connection flag
	(
		s_net_13,
		s_net_13_adj
	);

	nn2fpga::shift_op <
		t_net_13_adj_struct,
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		c_node_conv_9_ich,
		c_node_conv_9_och,
		c_node_conv_9_ih,
		c_node_conv_9_iw,
		c_node_conv_9_oh,
		c_node_conv_9_ow,
		c_node_conv_9_fh,
		c_node_conv_9_fw,
		c_node_conv_9_stride,
		c_node_conv_9_pad,
		2,
		3,
		c_node_conv_9_ow_ops,
		8,
		8>
	(
		s_net_13_adj[0],
		s_net_13_pre_pad[0],
		s_net_13_data[0]
	);

	nn2fpga::shift_op <
		t_net_13_adj_struct,
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		c_node_conv_9_ich,
		c_node_conv_9_och,
		c_node_conv_9_ih,
		c_node_conv_9_iw,
		c_node_conv_9_oh,
		c_node_conv_9_ow,
		c_node_conv_9_fh,
		c_node_conv_9_fw,
		c_node_conv_9_stride,
		c_node_conv_9_pad,
		2,
		2,
		c_node_conv_9_ow_ops,
		8,
		8>
	(
		s_net_13_adj[1],
		s_net_13_pre_pad[1],
		s_net_13_data[1]
	);

	nn2fpga::shift_op <
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		c_node_conv_9_ich,
		c_node_conv_9_och,
		c_node_conv_9_ih,
		c_node_conv_9_iw,
		c_node_conv_9_oh,
		c_node_conv_9_ow,
		c_node_conv_9_fh,
		c_node_conv_9_fw,
		c_node_conv_9_stride,
		c_node_conv_9_pad,
		2,
		1,
		c_node_conv_9_ow_ops,
		8,
		8>
	(
		s_net_13_data[0],
		s_net_13_pre_pad[2],
		s_net_13_data[2]
	);

	nn2fpga::shift_op <
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		c_node_conv_9_ich,
		c_node_conv_9_och,
		c_node_conv_9_ih,
		c_node_conv_9_iw,
		c_node_conv_9_oh,
		c_node_conv_9_ow,
		c_node_conv_9_fh,
		c_node_conv_9_fw,
		c_node_conv_9_stride,
		c_node_conv_9_pad,
		2,
		0,
		c_node_conv_9_ow_ops,
		8,
		8>
	(
		s_net_13_data[1],
		s_net_13_pre_pad[3],
		s_net_13_data[3]
	);

	nn2fpga::shift_op <
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		c_node_conv_9_ich,
		c_node_conv_9_och,
		c_node_conv_9_ih,
		c_node_conv_9_iw,
		c_node_conv_9_oh,
		c_node_conv_9_ow,
		c_node_conv_9_fh,
		c_node_conv_9_fw,
		c_node_conv_9_stride,
		c_node_conv_9_pad,
		1,
		3,
		c_node_conv_9_ow_ops,
		8,
		8>
	(
		s_net_13_data[2],
		s_net_13_pre_pad[4],
		s_net_13_data[4]
	);

	nn2fpga::shift_op <
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		c_node_conv_9_ich,
		c_node_conv_9_och,
		c_node_conv_9_ih,
		c_node_conv_9_iw,
		c_node_conv_9_oh,
		c_node_conv_9_ow,
		c_node_conv_9_fh,
		c_node_conv_9_fw,
		c_node_conv_9_stride,
		c_node_conv_9_pad,
		1,
		2,
		c_node_conv_9_ow_ops,
		8,
		8>
	(
		s_net_13_data[3],
		s_net_13_pre_pad[5],
		s_net_13_data[5]
	);

	nn2fpga::shift_op <
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		c_node_conv_9_ich,
		c_node_conv_9_och,
		c_node_conv_9_ih,
		c_node_conv_9_iw,
		c_node_conv_9_oh,
		c_node_conv_9_ow,
		c_node_conv_9_fh,
		c_node_conv_9_fw,
		c_node_conv_9_stride,
		c_node_conv_9_pad,
		1,
		1,
		c_node_conv_9_ow_ops,
		8,
		8>
	(
		s_net_13_data[4],
		s_net_13_pre_pad[6],
		s_net_13_data[6]
	);

	nn2fpga::shift_op <
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		c_node_conv_9_ich,
		c_node_conv_9_och,
		c_node_conv_9_ih,
		c_node_conv_9_iw,
		c_node_conv_9_oh,
		c_node_conv_9_ow,
		c_node_conv_9_fh,
		c_node_conv_9_fw,
		c_node_conv_9_stride,
		c_node_conv_9_pad,
		1,
		0,
		c_node_conv_9_ow_ops,
		8,
		8>
	(
		s_net_13_data[5],
		s_net_13_pre_pad[7],
		s_net_13_data[7]
	);

	nn2fpga::shift_op <
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		c_node_conv_9_ich,
		c_node_conv_9_och,
		c_node_conv_9_ih,
		c_node_conv_9_iw,
		c_node_conv_9_oh,
		c_node_conv_9_ow,
		c_node_conv_9_fh,
		c_node_conv_9_fw,
		c_node_conv_9_stride,
		c_node_conv_9_pad,
		0,
		3,
		c_node_conv_9_ow_ops,
		8,
		8>
	(
		s_net_13_data[6],
		s_net_13_pre_pad[8],
		s_net_13_data[8]
	);

	nn2fpga::shift_op <
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		c_node_conv_9_ich,
		c_node_conv_9_och,
		c_node_conv_9_ih,
		c_node_conv_9_iw,
		c_node_conv_9_oh,
		c_node_conv_9_ow,
		c_node_conv_9_fh,
		c_node_conv_9_fw,
		c_node_conv_9_stride,
		c_node_conv_9_pad,
		0,
		2,
		c_node_conv_9_ow_ops,
		8,
		8>
	(
		s_net_13_data[7],
		s_net_13_pre_pad[9],
		s_net_13_data[9]
	);

	nn2fpga::shift_op <
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		std::nullptr_t,
		c_node_conv_9_ich,
		c_node_conv_9_och,
		c_node_conv_9_ih,
		c_node_conv_9_iw,
		c_node_conv_9_oh,
		c_node_conv_9_ow,
		c_node_conv_9_fh,
		c_node_conv_9_fw,
		c_node_conv_9_stride,
		c_node_conv_9_pad,
		0,
		1,
		c_node_conv_9_ow_ops,
		8,
		8>
	(
		s_net_13_data[8],
		s_net_13_pre_pad[10],
		s_net_13_null[1]
	);

	nn2fpga::shift_op <
		t_net_13_lb_struct,
		t_net_13_lb_struct,
		std::nullptr_t,
		c_node_conv_9_ich,
		c_node_conv_9_och,
		c_node_conv_9_ih,
		c_node_conv_9_iw,
		c_node_conv_9_oh,
		c_node_conv_9_ow,
		c_node_conv_9_fh,
		c_node_conv_9_fw,
		c_node_conv_9_stride,
		c_node_conv_9_pad,
		0,
		0,
		c_node_conv_9_ow_ops,
		8,
		8>
	(
		s_net_13_data[9],
		s_net_13_pre_pad[11],
		s_net_13_null[0]
	);

	nn2fpga::pad_input <
		t_net_13_lb_struct,
		t_net_13_window_struct,
		c_node_conv_9_ich,
		c_node_conv_9_ih,
		c_node_conv_9_iw,
		c_node_conv_9_fh,
		c_node_conv_9_fw,
		c_node_conv_9_stride,
		c_node_conv_9_pad,
		c_node_conv_9_ow_ops,
		8,
		8>
	(
		s_net_13_pre_pad,
		s_net_13_compute
	);

	nn2fpga::conv_comp_wrap <
		t_net_13_window_struct,
		t_net_13_window,
		t_net_13_reduce,
		t_net_13,
		t_node_conv_9_weight,
		t_node_conv_9_weight_mem,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_13_mod,
		t_net_13,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_14_acc_struct,
		t_net_14_acc,
		std::nullptr_t,
		std::nullptr_t,
		t_net_14_struct,
		t_net_14_vector,
		t_net_14,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_params_stream,
		c_node_conv_9_ich,
		c_node_conv_9_ich,
		c_node_conv_9_och,
		c_node_conv_9_och_1x1,
		c_node_conv_9_oh,
		c_node_conv_9_ow,
		c_node_conv_9_fh,
		c_node_conv_9_fw,
		c_node_conv_9_index,
		c_node_conv_9_stride,
		c_node_conv_9_ops,
		c_node_conv_9_ops_out,
		c_node_conv_9_ich_ops,
		1,		//add ops
		c_node_conv_9_ow_ops,
		c_node_conv_9_ow_ops_out,
		8,		//bias_ops
		c_node_conv_9_reuse,
		c_node_conv_9_ow_pack,
		c_node_conv_9_och_pack,
		8,		//act bits
		3,		//act integer bits
		8,		//weight bits
		0,		//weight integer bits
		2,		//guard bits
		16,		//accumulators in parallel
		15,		//mask bits
		0,		//act bits 1x1
		0,		//act integer bits 1x1
		0,		//weight bits 1x1
		0,		//weight integer bits 1x1
		2,		//guard bits 1x1
		1,		//accumulators in parallel 1x1
		0,		//mask bits 1x1
		8257536,		//shift cycles
		8,		//width stream
		1,		//weight reads per data
		0,		//bias reads per data
		0,		//weight reads per data 1x1
		0,		//bias reads per data 1x1
		c_node_conv_9_relu,
		0>		//depth
	(
		s_node_conv_7_out,
		s_node_conv_9_init_flag,
		s_node_conv_9_out,
		nullptr,
		c_node_conv_9_weight,
		nullptr,
		nullptr,
		s_net_13_compute,
		(hls::stream<std::nullptr_t>*)(nullptr),
		(hls::stream<std::nullptr_t>*)(nullptr),
		s_net_14,
		(hls::stream<std::nullptr_t>*)(nullptr)
	);

	nn2fpga::tensor_duplicator <
		t_net_14_struct,
		256,		//Tensor channels
		26,		//Tensor height
		26,		//Tensor width
		8,		//Number of channel packed per packet
		2>		//Number of streams in parallel
	(
		s_net_14,
		s_net_14_dup_0,
		s_net_14_dup_1
	);

	nn2fpga::bandwidth_adjust <
		t_net_14_dup_0_struct,
		t_net_14_dup_0_adj_struct,
		c_bandwidth_adjust_net_14_dup_0_ich,
		c_bandwidth_adjust_net_14_dup_0_iw,
		c_bandwidth_adjust_net_14_dup_0_ih,
		c_bandwidth_adjust_net_14_dup_0_ow_ops_in,
		c_bandwidth_adjust_net_14_dup_0_ow_ops,
		c_bandwidth_adjust_net_14_dup_0_old_in_ops,
		c_bandwidth_adjust_net_14_dup_0_in_ops,
		false>		//skip connection flag
	(
		s_net_14_dup_0,
		s_net_14_dup_0_adj
	);

	nn2fpga::shift_op <
		t_net_14_dup_0_adj_struct,
		t_net_14_dup_0_lb_struct,
		t_net_14_dup_0_lb_struct,
		c_node_pool_10_ich,
		c_node_pool_10_och,
		c_node_pool_10_ih,
		c_node_pool_10_iw,
		c_node_pool_10_oh,
		c_node_pool_10_ow,
		c_node_pool_10_fh,
		c_node_pool_10_fw,
		c_node_pool_10_stride,
		c_node_pool_10_pad,
		1,
		1,
		c_node_pool_10_ow_ops,
		8,
		8>
	(
		s_net_14_dup_0_adj[0],
		s_net_14_dup_0_pre_pad[0],
		s_net_14_dup_0_data[0]
	);

	nn2fpga::shift_op <
		t_net_14_dup_0_lb_struct,
		t_net_14_dup_0_lb_struct,
		t_net_14_dup_0_lb_struct,
		c_node_pool_10_ich,
		c_node_pool_10_och,
		c_node_pool_10_ih,
		c_node_pool_10_iw,
		c_node_pool_10_oh,
		c_node_pool_10_ow,
		c_node_pool_10_fh,
		c_node_pool_10_fw,
		c_node_pool_10_stride,
		c_node_pool_10_pad,
		1,
		0,
		c_node_pool_10_ow_ops,
		8,
		8>
	(
		s_net_14_dup_0_data[0],
		s_net_14_dup_0_pre_pad[1],
		s_net_14_dup_0_data[1]
	);

	nn2fpga::shift_op <
		t_net_14_dup_0_lb_struct,
		t_net_14_dup_0_lb_struct,
		t_net_14_dup_0_lb_struct,
		c_node_pool_10_ich,
		c_node_pool_10_och,
		c_node_pool_10_ih,
		c_node_pool_10_iw,
		c_node_pool_10_oh,
		c_node_pool_10_ow,
		c_node_pool_10_fh,
		c_node_pool_10_fw,
		c_node_pool_10_stride,
		c_node_pool_10_pad,
		0,
		1,
		c_node_pool_10_ow_ops,
		8,
		8>
	(
		s_net_14_dup_0_data[1],
		s_net_14_dup_0_pre_pad[2],
		s_net_14_dup_0_data[2]
	);

	nn2fpga::shift_op <
		t_net_14_dup_0_lb_struct,
		t_net_14_dup_0_lb_struct,
		std::nullptr_t,
		c_node_pool_10_ich,
		c_node_pool_10_och,
		c_node_pool_10_ih,
		c_node_pool_10_iw,
		c_node_pool_10_oh,
		c_node_pool_10_ow,
		c_node_pool_10_fh,
		c_node_pool_10_fw,
		c_node_pool_10_stride,
		c_node_pool_10_pad,
		0,
		0,
		c_node_pool_10_ow_ops,
		8,
		8>
	(
		s_net_14_dup_0_data[2],
		s_net_14_dup_0_pre_pad[3],
		s_net_14_dup_0_null[0]
	);

	nn2fpga::pad_input <
		t_net_14_dup_0_lb_struct,
		t_net_14_dup_0_window_struct,
		c_node_pool_10_ich,
		c_node_pool_10_ih,
		c_node_pool_10_iw,
		c_node_pool_10_fh,
		c_node_pool_10_fw,
		c_node_pool_10_stride,
		c_node_pool_10_pad,
		c_node_pool_10_ow_ops,
		8,
		8>
	(
		s_net_14_dup_0_pre_pad,
		s_net_14_dup_0_compute
	);

	nn2fpga::pool_op <
		t_net_14_dup_0_window_struct,
		t_net_14_dup_0_window,
		t_net_15_struct,
		t_net_15,
		t_node_pool_10_acc,
		t_node_pool_10_acc,
		c_node_pool_10_ich,
		c_node_pool_10_och,
		c_node_pool_10_ih,
		c_node_pool_10_iw,
		c_node_pool_10_oh,
		c_node_pool_10_ow,
		c_node_pool_10_fw,
		c_node_pool_10_fh,
		c_node_pool_10_stride,
		c_node_pool_10_pad,
		c_node_pool_10_pool,
		c_node_pool_10_ow_ops,
		c_node_pool_10_ops,
		c_node_pool_10_in_ops>
	(
		s_net_14_dup_0_compute,
		s_net_15
	);

	nn2fpga::shift_op <
		t_net_15_struct,
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		c_node_conv_11_ich,
		c_node_conv_11_och,
		c_node_conv_11_ih,
		c_node_conv_11_iw,
		c_node_conv_11_oh,
		c_node_conv_11_ow,
		c_node_conv_11_fh,
		c_node_conv_11_fw,
		c_node_conv_11_stride,
		c_node_conv_11_pad,
		2,
		2,
		c_node_conv_11_ow_ops,
		8,
		8>
	(
		s_net_15[0],
		s_net_15_pre_pad[0],
		s_net_15_data[0]
	);

	nn2fpga::shift_op <
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		c_node_conv_11_ich,
		c_node_conv_11_och,
		c_node_conv_11_ih,
		c_node_conv_11_iw,
		c_node_conv_11_oh,
		c_node_conv_11_ow,
		c_node_conv_11_fh,
		c_node_conv_11_fw,
		c_node_conv_11_stride,
		c_node_conv_11_pad,
		2,
		1,
		c_node_conv_11_ow_ops,
		8,
		8>
	(
		s_net_15_data[0],
		s_net_15_pre_pad[1],
		s_net_15_data[1]
	);

	nn2fpga::shift_op <
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		c_node_conv_11_ich,
		c_node_conv_11_och,
		c_node_conv_11_ih,
		c_node_conv_11_iw,
		c_node_conv_11_oh,
		c_node_conv_11_ow,
		c_node_conv_11_fh,
		c_node_conv_11_fw,
		c_node_conv_11_stride,
		c_node_conv_11_pad,
		2,
		0,
		c_node_conv_11_ow_ops,
		8,
		8>
	(
		s_net_15_data[1],
		s_net_15_pre_pad[2],
		s_net_15_data[2]
	);

	nn2fpga::shift_op <
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		c_node_conv_11_ich,
		c_node_conv_11_och,
		c_node_conv_11_ih,
		c_node_conv_11_iw,
		c_node_conv_11_oh,
		c_node_conv_11_ow,
		c_node_conv_11_fh,
		c_node_conv_11_fw,
		c_node_conv_11_stride,
		c_node_conv_11_pad,
		1,
		2,
		c_node_conv_11_ow_ops,
		8,
		8>
	(
		s_net_15_data[2],
		s_net_15_pre_pad[3],
		s_net_15_data[3]
	);

	nn2fpga::shift_op <
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		c_node_conv_11_ich,
		c_node_conv_11_och,
		c_node_conv_11_ih,
		c_node_conv_11_iw,
		c_node_conv_11_oh,
		c_node_conv_11_ow,
		c_node_conv_11_fh,
		c_node_conv_11_fw,
		c_node_conv_11_stride,
		c_node_conv_11_pad,
		1,
		1,
		c_node_conv_11_ow_ops,
		8,
		8>
	(
		s_net_15_data[3],
		s_net_15_pre_pad[4],
		s_net_15_data[4]
	);

	nn2fpga::shift_op <
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		c_node_conv_11_ich,
		c_node_conv_11_och,
		c_node_conv_11_ih,
		c_node_conv_11_iw,
		c_node_conv_11_oh,
		c_node_conv_11_ow,
		c_node_conv_11_fh,
		c_node_conv_11_fw,
		c_node_conv_11_stride,
		c_node_conv_11_pad,
		1,
		0,
		c_node_conv_11_ow_ops,
		8,
		8>
	(
		s_net_15_data[4],
		s_net_15_pre_pad[5],
		s_net_15_data[5]
	);

	nn2fpga::shift_op <
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		c_node_conv_11_ich,
		c_node_conv_11_och,
		c_node_conv_11_ih,
		c_node_conv_11_iw,
		c_node_conv_11_oh,
		c_node_conv_11_ow,
		c_node_conv_11_fh,
		c_node_conv_11_fw,
		c_node_conv_11_stride,
		c_node_conv_11_pad,
		0,
		2,
		c_node_conv_11_ow_ops,
		8,
		8>
	(
		s_net_15_data[5],
		s_net_15_pre_pad[6],
		s_net_15_data[6]
	);

	nn2fpga::shift_op <
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		c_node_conv_11_ich,
		c_node_conv_11_och,
		c_node_conv_11_ih,
		c_node_conv_11_iw,
		c_node_conv_11_oh,
		c_node_conv_11_ow,
		c_node_conv_11_fh,
		c_node_conv_11_fw,
		c_node_conv_11_stride,
		c_node_conv_11_pad,
		0,
		1,
		c_node_conv_11_ow_ops,
		8,
		8>
	(
		s_net_15_data[6],
		s_net_15_pre_pad[7],
		s_net_15_data[7]
	);

	nn2fpga::shift_op <
		t_net_15_lb_struct,
		t_net_15_lb_struct,
		std::nullptr_t,
		c_node_conv_11_ich,
		c_node_conv_11_och,
		c_node_conv_11_ih,
		c_node_conv_11_iw,
		c_node_conv_11_oh,
		c_node_conv_11_ow,
		c_node_conv_11_fh,
		c_node_conv_11_fw,
		c_node_conv_11_stride,
		c_node_conv_11_pad,
		0,
		0,
		c_node_conv_11_ow_ops,
		8,
		8>
	(
		s_net_15_data[7],
		s_net_15_pre_pad[8],
		s_net_15_null[0]
	);

	nn2fpga::pad_input <
		t_net_15_lb_struct,
		t_net_15_window_struct,
		c_node_conv_11_ich,
		c_node_conv_11_ih,
		c_node_conv_11_iw,
		c_node_conv_11_fh,
		c_node_conv_11_fw,
		c_node_conv_11_stride,
		c_node_conv_11_pad,
		c_node_conv_11_ow_ops,
		8,
		8>
	(
		s_net_15_pre_pad,
		s_net_15_compute
	);

	nn2fpga::conv_comp_wrap <
		t_net_15_window_struct,
		t_net_15_window,
		t_net_15_reduce,
		t_net_15,
		t_node_conv_11_weight,
		t_node_conv_11_weight_mem,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_15_mod,
		t_net_15,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_16_acc_struct,
		t_net_16_acc,
		std::nullptr_t,
		std::nullptr_t,
		t_net_16_struct,
		t_net_16_vector,
		t_net_16,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_params_stream,
		c_node_conv_11_ich,
		c_node_conv_11_ich,
		c_node_conv_11_och,
		c_node_conv_11_och_1x1,
		c_node_conv_11_oh,
		c_node_conv_11_ow,
		c_node_conv_11_fh,
		c_node_conv_11_fw,
		c_node_conv_11_index,
		c_node_conv_11_stride,
		c_node_conv_11_ops,
		c_node_conv_11_ops_out,
		c_node_conv_11_ich_ops,
		1,		//add ops
		c_node_conv_11_ow_ops,
		c_node_conv_11_ow_ops_out,
		16,		//bias_ops
		c_node_conv_11_reuse,
		c_node_conv_11_ow_pack,
		c_node_conv_11_och_pack,
		8,		//act bits
		3,		//act integer bits
		8,		//weight bits
		0,		//weight integer bits
		2,		//guard bits
		16,		//accumulators in parallel
		15,		//mask bits
		0,		//act bits 1x1
		0,		//act integer bits 1x1
		0,		//weight bits 1x1
		0,		//weight integer bits 1x1
		2,		//guard bits 1x1
		1,		//accumulators in parallel 1x1
		0,		//mask bits 1x1
		7077888,		//shift cycles
		8,		//width stream
		1,		//weight reads per data
		0,		//bias reads per data
		0,		//weight reads per data 1x1
		0,		//bias reads per data 1x1
		c_node_conv_11_relu,
		0>		//depth
	(
		s_node_conv_9_out,
		s_node_conv_11_init_flag,
		s_node_conv_11_out,
		nullptr,
		c_node_conv_11_weight,
		nullptr,
		nullptr,
		s_net_15_compute,
		(hls::stream<std::nullptr_t>*)(nullptr),
		(hls::stream<std::nullptr_t>*)(nullptr),
		s_net_16,
		(hls::stream<std::nullptr_t>*)(nullptr)
	);

	nn2fpga::shift_op <
		t_net_16_struct,
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		c_node_conv_12_ich,
		c_node_conv_12_och,
		c_node_conv_12_ih,
		c_node_conv_12_iw,
		c_node_conv_12_oh,
		c_node_conv_12_ow,
		c_node_conv_12_fh,
		c_node_conv_12_fw,
		c_node_conv_12_stride,
		c_node_conv_12_pad,
		2,
		2,
		c_node_conv_12_ow_ops,
		32,
		32>
	(
		s_net_16[0],
		s_net_16_pre_pad[0],
		s_net_16_data[0]
	);

	nn2fpga::shift_op <
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		c_node_conv_12_ich,
		c_node_conv_12_och,
		c_node_conv_12_ih,
		c_node_conv_12_iw,
		c_node_conv_12_oh,
		c_node_conv_12_ow,
		c_node_conv_12_fh,
		c_node_conv_12_fw,
		c_node_conv_12_stride,
		c_node_conv_12_pad,
		2,
		1,
		c_node_conv_12_ow_ops,
		32,
		32>
	(
		s_net_16_data[0],
		s_net_16_pre_pad[1],
		s_net_16_data[1]
	);

	nn2fpga::shift_op <
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		c_node_conv_12_ich,
		c_node_conv_12_och,
		c_node_conv_12_ih,
		c_node_conv_12_iw,
		c_node_conv_12_oh,
		c_node_conv_12_ow,
		c_node_conv_12_fh,
		c_node_conv_12_fw,
		c_node_conv_12_stride,
		c_node_conv_12_pad,
		2,
		0,
		c_node_conv_12_ow_ops,
		32,
		32>
	(
		s_net_16_data[1],
		s_net_16_pre_pad[2],
		s_net_16_data[2]
	);

	nn2fpga::shift_op <
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		c_node_conv_12_ich,
		c_node_conv_12_och,
		c_node_conv_12_ih,
		c_node_conv_12_iw,
		c_node_conv_12_oh,
		c_node_conv_12_ow,
		c_node_conv_12_fh,
		c_node_conv_12_fw,
		c_node_conv_12_stride,
		c_node_conv_12_pad,
		1,
		2,
		c_node_conv_12_ow_ops,
		32,
		32>
	(
		s_net_16_data[2],
		s_net_16_pre_pad[3],
		s_net_16_data[3]
	);

	nn2fpga::shift_op <
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		c_node_conv_12_ich,
		c_node_conv_12_och,
		c_node_conv_12_ih,
		c_node_conv_12_iw,
		c_node_conv_12_oh,
		c_node_conv_12_ow,
		c_node_conv_12_fh,
		c_node_conv_12_fw,
		c_node_conv_12_stride,
		c_node_conv_12_pad,
		1,
		1,
		c_node_conv_12_ow_ops,
		32,
		32>
	(
		s_net_16_data[3],
		s_net_16_pre_pad[4],
		s_net_16_data[4]
	);

	nn2fpga::shift_op <
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		c_node_conv_12_ich,
		c_node_conv_12_och,
		c_node_conv_12_ih,
		c_node_conv_12_iw,
		c_node_conv_12_oh,
		c_node_conv_12_ow,
		c_node_conv_12_fh,
		c_node_conv_12_fw,
		c_node_conv_12_stride,
		c_node_conv_12_pad,
		1,
		0,
		c_node_conv_12_ow_ops,
		32,
		32>
	(
		s_net_16_data[4],
		s_net_16_pre_pad[5],
		s_net_16_data[5]
	);

	nn2fpga::shift_op <
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		c_node_conv_12_ich,
		c_node_conv_12_och,
		c_node_conv_12_ih,
		c_node_conv_12_iw,
		c_node_conv_12_oh,
		c_node_conv_12_ow,
		c_node_conv_12_fh,
		c_node_conv_12_fw,
		c_node_conv_12_stride,
		c_node_conv_12_pad,
		0,
		2,
		c_node_conv_12_ow_ops,
		32,
		32>
	(
		s_net_16_data[5],
		s_net_16_pre_pad[6],
		s_net_16_data[6]
	);

	nn2fpga::shift_op <
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		c_node_conv_12_ich,
		c_node_conv_12_och,
		c_node_conv_12_ih,
		c_node_conv_12_iw,
		c_node_conv_12_oh,
		c_node_conv_12_ow,
		c_node_conv_12_fh,
		c_node_conv_12_fw,
		c_node_conv_12_stride,
		c_node_conv_12_pad,
		0,
		1,
		c_node_conv_12_ow_ops,
		32,
		32>
	(
		s_net_16_data[6],
		s_net_16_pre_pad[7],
		s_net_16_data[7]
	);

	nn2fpga::shift_op <
		t_net_16_lb_struct,
		t_net_16_lb_struct,
		std::nullptr_t,
		c_node_conv_12_ich,
		c_node_conv_12_och,
		c_node_conv_12_ih,
		c_node_conv_12_iw,
		c_node_conv_12_oh,
		c_node_conv_12_ow,
		c_node_conv_12_fh,
		c_node_conv_12_fw,
		c_node_conv_12_stride,
		c_node_conv_12_pad,
		0,
		0,
		c_node_conv_12_ow_ops,
		32,
		32>
	(
		s_net_16_data[7],
		s_net_16_pre_pad[8],
		s_net_16_null[0]
	);

	nn2fpga::pad_input <
		t_net_16_lb_struct,
		t_net_16_window_struct,
		c_node_conv_12_ich,
		c_node_conv_12_ih,
		c_node_conv_12_iw,
		c_node_conv_12_fh,
		c_node_conv_12_fw,
		c_node_conv_12_stride,
		c_node_conv_12_pad,
		c_node_conv_12_ow_ops,
		32,
		32>
	(
		s_net_16_pre_pad,
		s_net_16_compute
	);

	nn2fpga::conv_comp_wrap <
		t_net_16_window_struct,
		t_net_16_window,
		t_net_16_reduce,
		t_net_16,
		t_node_conv_12_weight,
		t_node_conv_12_weight_mem,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_16_mod,
		t_net_16,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_17_acc_struct,
		t_net_17_acc,
		std::nullptr_t,
		std::nullptr_t,
		t_net_17_struct,
		t_net_17_vector,
		t_net_17,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_params_stream,
		c_node_conv_12_ich,
		c_node_conv_12_ich,
		c_node_conv_12_och,
		c_node_conv_12_och_1x1,
		c_node_conv_12_oh,
		c_node_conv_12_ow,
		c_node_conv_12_fh,
		c_node_conv_12_fw,
		c_node_conv_12_index,
		c_node_conv_12_stride,
		c_node_conv_12_ops,
		c_node_conv_12_ops_out,
		c_node_conv_12_ich_ops,
		1,		//add ops
		c_node_conv_12_ow_ops,
		c_node_conv_12_ow_ops_out,
		16,		//bias_ops
		c_node_conv_12_reuse,
		c_node_conv_12_ow_pack,
		c_node_conv_12_och_pack,
		8,		//act bits
		3,		//act integer bits
		8,		//weight bits
		-1,		//weight integer bits
		2,		//guard bits
		64,		//accumulators in parallel
		63,		//mask bits
		0,		//act bits 1x1
		0,		//act integer bits 1x1
		0,		//weight bits 1x1
		0,		//weight integer bits 1x1
		2,		//guard bits 1x1
		1,		//accumulators in parallel 1x1
		0,		//mask bits 1x1
		2359296,		//shift cycles
		8,		//width stream
		1,		//weight reads per data
		0,		//bias reads per data
		0,		//weight reads per data 1x1
		0,		//bias reads per data 1x1
		c_node_conv_12_relu,
		0>		//depth
	(
		s_node_conv_11_out,
		s_node_conv_12_init_flag,
		s_node_conv_12_out,
		nullptr,
		c_node_conv_12_weight,
		nullptr,
		nullptr,
		s_net_16_compute,
		(hls::stream<std::nullptr_t>*)(nullptr),
		(hls::stream<std::nullptr_t>*)(nullptr),
		s_net_17,
		(hls::stream<std::nullptr_t>*)(nullptr)
	);

	// nn2fpga::shift_op <
	// 	t_net_17_struct,
	// 	t_net_17_lb_struct,
	// 	std::nullptr_t,
	// 	c_node_conv_13_ich,
	// 	c_node_conv_13_och,
	// 	c_node_conv_13_ih,
	// 	c_node_conv_13_iw,
	// 	c_node_conv_13_oh,
	// 	c_node_conv_13_ow,
	// 	c_node_conv_13_fh,
	// 	c_node_conv_13_fw,
	// 	c_node_conv_13_stride,
	// 	c_node_conv_13_pad,
	// 	0,
	// 	0,
	// 	c_node_conv_13_ow_ops,
	// 	16,
	// 	16>
	// (
	// 	s_net_17[0],
	// 	s_net_17_pre_pad[0],
	// 	s_net_17_null[0]
	// );

	nn2fpga::conv_comp_wrap <
		// t_net_17_lb_struct,
		t_net_17_struct,
		t_net_17_lb,
		t_net_17_reduce,
		t_net_17,
		t_node_conv_13_weight,
		t_node_conv_13_weight_mem,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_17_mod,
		t_net_17,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_18_acc_struct,
		t_net_18_acc,
		std::nullptr_t,
		std::nullptr_t,
		t_net_18_struct,
		t_net_18_vector,
		t_net_18,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_params_stream,
		c_node_conv_13_ich,
		c_node_conv_13_ich,
		c_node_conv_13_och,
		c_node_conv_13_och_1x1,
		c_node_conv_13_oh,
		c_node_conv_13_ow,
		c_node_conv_13_fh,
		c_node_conv_13_fw,
		c_node_conv_13_index,
		c_node_conv_13_stride,
		c_node_conv_13_ops,
		c_node_conv_13_ops_out,
		c_node_conv_13_ich_ops,
		1,		//add ops
		c_node_conv_13_ow_ops,
		c_node_conv_13_ow_ops_out,
		16,		//bias_ops
		c_node_conv_13_reuse,
		c_node_conv_13_ow_pack,
		c_node_conv_13_och_pack,
		8,		//act bits
		3,		//act integer bits
		8,		//weight bits
		0,		//weight integer bits
		2,		//guard bits
		4,		//accumulators in parallel
		3,		//mask bits
		0,		//act bits 1x1
		0,		//act integer bits 1x1
		0,		//weight bits 1x1
		0,		//weight integer bits 1x1
		2,		//guard bits 1x1
		1,		//accumulators in parallel 1x1
		0,		//mask bits 1x1
		2097152,		//shift cycles
		8,		//width stream
		1,		//weight reads per data
		0,		//bias reads per data
		0,		//weight reads per data 1x1
		0,		//bias reads per data 1x1
		c_node_conv_13_relu,
		0>		//depth
	(
		s_node_conv_12_out,
		s_node_conv_13_init_flag,
		s_node_conv_13_out,
		nullptr,
		c_node_conv_13_weight,
		nullptr,
		nullptr,
		// s_net_17_pre_pad,
		s_net_17,
		(hls::stream<std::nullptr_t>*)(nullptr),
		(hls::stream<std::nullptr_t>*)(nullptr),
		s_net_18,
		(hls::stream<std::nullptr_t>*)(nullptr)
	);

	nn2fpga::tensor_duplicator <
		t_net_18_struct,
		256,		//Tensor channels
		13,		//Tensor height
		13,		//Tensor width
		16,		//Number of channel packed per packet
		1>		//Number of streams in parallel
	(
		s_net_18,
		s_net_18_dup_0,
		s_net_18_dup_1
	);

	// nn2fpga::bandwidth_adjust <
	// 	t_net_18_dup_0_struct,
	// 	t_net_18_dup_0_adj_struct,
	// 	c_bandwidth_adjust_net_18_dup_0_ich,
	// 	c_bandwidth_adjust_net_18_dup_0_iw,
	// 	c_bandwidth_adjust_net_18_dup_0_ih,
	// 	c_bandwidth_adjust_net_18_dup_0_ow_ops_in,
	// 	c_bandwidth_adjust_net_18_dup_0_ow_ops,
	// 	c_bandwidth_adjust_net_18_dup_0_old_in_ops,
	// 	c_bandwidth_adjust_net_18_dup_0_in_ops,
	// 	false>		//skip connection flag
	// (
	// 	s_net_18_dup_0,
	// 	s_net_18_dup_0_adj
	// );

	nn2fpga::shift_op <
		t_net_18_dup_0_struct,
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		c_node_conv_14_ich,
		c_node_conv_14_och,
		c_node_conv_14_ih,
		c_node_conv_14_iw,
		c_node_conv_14_oh,
		c_node_conv_14_ow,
		c_node_conv_14_fh,
		c_node_conv_14_fw,
		c_node_conv_14_stride,
		c_node_conv_14_pad,
		2,
		2,
		c_node_conv_14_ow_ops,
		16,
		8>
	(
		s_net_18_dup_0[0],
		s_net_18_dup_0_pre_pad[0],
		s_net_18_dup_0_data[0]
	);

	nn2fpga::shift_op <
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		c_node_conv_14_ich,
		c_node_conv_14_och,
		c_node_conv_14_ih,
		c_node_conv_14_iw,
		c_node_conv_14_oh,
		c_node_conv_14_ow,
		c_node_conv_14_fh,
		c_node_conv_14_fw,
		c_node_conv_14_stride,
		c_node_conv_14_pad,
		2,
		1,
		c_node_conv_14_ow_ops,
		8,
		8>
	(
		s_net_18_dup_0_data[0],
		s_net_18_dup_0_pre_pad[1],
		s_net_18_dup_0_data[1]
	);

	nn2fpga::shift_op <
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		c_node_conv_14_ich,
		c_node_conv_14_och,
		c_node_conv_14_ih,
		c_node_conv_14_iw,
		c_node_conv_14_oh,
		c_node_conv_14_ow,
		c_node_conv_14_fh,
		c_node_conv_14_fw,
		c_node_conv_14_stride,
		c_node_conv_14_pad,
		2,
		0,
		c_node_conv_14_ow_ops,
		8,
		8>
	(
		s_net_18_dup_0_data[1],
		s_net_18_dup_0_pre_pad[2],
		s_net_18_dup_0_data[2]
	);

	nn2fpga::shift_op <
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		c_node_conv_14_ich,
		c_node_conv_14_och,
		c_node_conv_14_ih,
		c_node_conv_14_iw,
		c_node_conv_14_oh,
		c_node_conv_14_ow,
		c_node_conv_14_fh,
		c_node_conv_14_fw,
		c_node_conv_14_stride,
		c_node_conv_14_pad,
		1,
		2,
		c_node_conv_14_ow_ops,
		8,
		8>
	(
		s_net_18_dup_0_data[2],
		s_net_18_dup_0_pre_pad[3],
		s_net_18_dup_0_data[3]
	);

	nn2fpga::shift_op <
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		c_node_conv_14_ich,
		c_node_conv_14_och,
		c_node_conv_14_ih,
		c_node_conv_14_iw,
		c_node_conv_14_oh,
		c_node_conv_14_ow,
		c_node_conv_14_fh,
		c_node_conv_14_fw,
		c_node_conv_14_stride,
		c_node_conv_14_pad,
		1,
		1,
		c_node_conv_14_ow_ops,
		8,
		8>
	(
		s_net_18_dup_0_data[3],
		s_net_18_dup_0_pre_pad[4],
		s_net_18_dup_0_data[4]
	);

	nn2fpga::shift_op <
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		c_node_conv_14_ich,
		c_node_conv_14_och,
		c_node_conv_14_ih,
		c_node_conv_14_iw,
		c_node_conv_14_oh,
		c_node_conv_14_ow,
		c_node_conv_14_fh,
		c_node_conv_14_fw,
		c_node_conv_14_stride,
		c_node_conv_14_pad,
		1,
		0,
		c_node_conv_14_ow_ops,
		8,
		8>
	(
		s_net_18_dup_0_data[4],
		s_net_18_dup_0_pre_pad[5],
		s_net_18_dup_0_data[5]
	);

	nn2fpga::shift_op <
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		c_node_conv_14_ich,
		c_node_conv_14_och,
		c_node_conv_14_ih,
		c_node_conv_14_iw,
		c_node_conv_14_oh,
		c_node_conv_14_ow,
		c_node_conv_14_fh,
		c_node_conv_14_fw,
		c_node_conv_14_stride,
		c_node_conv_14_pad,
		0,
		2,
		c_node_conv_14_ow_ops,
		8,
		8>
	(
		s_net_18_dup_0_data[5],
		s_net_18_dup_0_pre_pad[6],
		s_net_18_dup_0_data[6]
	);

	nn2fpga::shift_op <
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		c_node_conv_14_ich,
		c_node_conv_14_och,
		c_node_conv_14_ih,
		c_node_conv_14_iw,
		c_node_conv_14_oh,
		c_node_conv_14_ow,
		c_node_conv_14_fh,
		c_node_conv_14_fw,
		c_node_conv_14_stride,
		c_node_conv_14_pad,
		0,
		1,
		c_node_conv_14_ow_ops,
		8,
		8>
	(
		s_net_18_dup_0_data[6],
		s_net_18_dup_0_pre_pad[7],
		s_net_18_dup_0_data[7]
	);

	nn2fpga::shift_op <
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_lb_struct,
		std::nullptr_t,
		c_node_conv_14_ich,
		c_node_conv_14_och,
		c_node_conv_14_ih,
		c_node_conv_14_iw,
		c_node_conv_14_oh,
		c_node_conv_14_ow,
		c_node_conv_14_fh,
		c_node_conv_14_fw,
		c_node_conv_14_stride,
		c_node_conv_14_pad,
		0,
		0,
		c_node_conv_14_ow_ops,
		8,
		8>
	(
		s_net_18_dup_0_data[7],
		s_net_18_dup_0_pre_pad[8],
		s_net_18_dup_0_null[0]
	);

	nn2fpga::pad_input <
		t_net_18_dup_0_lb_struct,
		t_net_18_dup_0_window_struct,
		c_node_conv_14_ich,
		c_node_conv_14_ih,
		c_node_conv_14_iw,
		c_node_conv_14_fh,
		c_node_conv_14_fw,
		c_node_conv_14_stride,
		c_node_conv_14_pad,
		c_node_conv_14_ow_ops,
		8,
		8>
	(
		s_net_18_dup_0_pre_pad,
		s_net_18_dup_0_compute
	);

	nn2fpga::conv_comp_wrap <
		t_net_18_dup_0_window_struct,
		t_net_18_dup_0_window,
		t_net_18_dup_0_reduce,
		t_net_18_dup_0,
		t_node_conv_14_weight,
		t_node_conv_14_weight_mem,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_18_dup_0_mod,
		t_net_18_dup_0,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_19_acc_struct,
		t_net_19_acc,
		std::nullptr_t,
		std::nullptr_t,
		t_net_19_struct,
		t_net_19_vector,
		t_net_19,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_params_stream,
		c_node_conv_14_ich,
		c_node_conv_14_ich,
		c_node_conv_14_och,
		c_node_conv_14_och_1x1,
		c_node_conv_14_oh,
		c_node_conv_14_ow,
		c_node_conv_14_fh,
		c_node_conv_14_fw,
		c_node_conv_14_index,
		c_node_conv_14_stride,
		c_node_conv_14_ops,
		c_node_conv_14_ops_out,
		c_node_conv_14_ich_ops,
		1,		//add ops
		c_node_conv_14_ow_ops,
		c_node_conv_14_ow_ops_out,
		16,		//bias_ops
		c_node_conv_14_reuse,
		c_node_conv_14_ow_pack,
		c_node_conv_14_och_pack,
		8,		//act bits
		3,		//act integer bits
		8,		//weight bits
		-1,		//weight integer bits
		2,		//guard bits
		16,		//accumulators in parallel
		15,		//mask bits
		0,		//act bits 1x1
		0,		//act integer bits 1x1
		0,		//weight bits 1x1
		0,		//weight integer bits 1x1
		2,		//guard bits 1x1
		1,		//accumulators in parallel 1x1
		0,		//mask bits 1x1
		917504,		//shift cycles
		8,		//width stream
		1,		//weight reads per data
		0,		//bias reads per data
		0,		//weight reads per data 1x1
		0,		//bias reads per data 1x1
		c_node_conv_14_relu,
		0>		//depth
	(
		s_node_conv_13_out,
		s_node_conv_14_init_flag,
		s_node_conv_14_out,
		nullptr,
		c_node_conv_14_weight,
		nullptr,
		nullptr,
		s_net_18_dup_0_compute,
		(hls::stream<std::nullptr_t>*)(nullptr),
		(hls::stream<std::nullptr_t>*)(nullptr),
		s_net_19,
		(hls::stream<std::nullptr_t>*)(nullptr)
	);

	nn2fpga::shift_op <
		t_net_18_dup_1_struct,
		t_net_18_dup_1_lb_struct,
		std::nullptr_t,
		c_node_conv_15_ich,
		c_node_conv_15_och,
		c_node_conv_15_ih,
		c_node_conv_15_iw,
		c_node_conv_15_oh,
		c_node_conv_15_ow,
		c_node_conv_15_fh,
		c_node_conv_15_fw,
		c_node_conv_15_stride,
		c_node_conv_15_pad,
		0,
		0,
		c_node_conv_15_ow_ops,
		// manually changed from 1 to 8
		16,
		8>
	(
		s_net_18_dup_1[0],
		s_net_18_dup_1_pre_pad[0],
		s_net_18_dup_1_null[0]
	);

	nn2fpga::conv_comp_wrap <
		t_net_18_dup_1_lb_struct,
		t_net_18_dup_1_lb,
		t_net_18_dup_1_reduce,
		t_net_18_dup_1,
		t_node_conv_15_weight,
		t_node_conv_15_weight_mem,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_18_dup_1_mod,
		t_net_18_dup_1,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_20_acc_struct,
		t_net_20_acc,
		std::nullptr_t,
		std::nullptr_t,
		t_net_20_struct,
		t_net_20_vector,
		t_net_20,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_params_stream,
		c_node_conv_15_ich,
		c_node_conv_15_ich,
		c_node_conv_15_och,
		c_node_conv_15_och_1x1,
		c_node_conv_15_oh,
		c_node_conv_15_ow,
		c_node_conv_15_fh,
		c_node_conv_15_fw,
		c_node_conv_15_index,
		c_node_conv_15_stride,
		c_node_conv_15_ops,
		c_node_conv_15_ops_out,
		c_node_conv_15_ich_ops,
		1,		//add ops
		c_node_conv_15_ow_ops,
		c_node_conv_15_ow_ops_out,
		4,		//bias_ops
		c_node_conv_15_reuse,
		c_node_conv_15_ow_pack,
		c_node_conv_15_och_pack,
		8,		//act bits
		3,		//act integer bits
		8,		//weight bits
		0,		//weight integer bits
		2,		//guard bits
		2,		//accumulators in parallel
		1,		//mask bits
		0,		//act bits 1x1
		0,		//act integer bits 1x1
		0,		//weight bits 1x1
		0,		//weight integer bits 1x1
		2,		//guard bits 1x1
		1,		//accumulators in parallel 1x1
		0,		//mask bits 1x1
		884736,		//shift cycles
		8,		//width stream
		1,		//weight reads per data
		0,		//bias reads per data
		0,		//weight reads per data 1x1
		0,		//bias reads per data 1x1
		c_node_conv_15_relu,
		0>		//depth
	(
		s_node_conv_14_out,
		s_node_conv_15_init_flag,
		s_node_conv_15_out,
		nullptr,
		c_node_conv_15_weight,
		nullptr,
		nullptr,
		s_net_18_dup_1_pre_pad,
		(hls::stream<std::nullptr_t>*)(nullptr),
		(hls::stream<std::nullptr_t>*)(nullptr),
		s_net_20,
		(hls::stream<std::nullptr_t>*)(nullptr)
	);

	nn2fpga::upsample_op <
		t_net_20_struct,
		c_node_upsample_16_ich,
		c_node_upsample_16_ih,
		c_node_upsample_16_iw,
		c_node_upsample_16_factor,
		c_node_upsample_16_ops,
		c_node_upsample_16_ow_ops>
	(
		s_net_20,
		s_net_23
	);

	nn2fpga::bandwidth_adjust <
		t_net_14_dup_1_struct,
		t_net_20_struct,
		256,
		26,	
		26,
		2,
		1,
		8,
		4,
		false>		//skip connection flag
	(
		s_net_14_dup_1,
		s_net_14_dup_1_adj
	);

	nn2fpga::concat_op <
		t_net_20_struct,
		c_node_concat_17_feature_map,
		128,
		256,
		384,
		4,	
		c_node_concat_17_ow_ops_in>
	(
		s_net_23,
		s_net_14_dup_1_adj,		
		s_net_24_pre_adj
	);

	nn2fpga::bandwidth_adjust <
		t_net_20_struct,
		t_net_24_struct,
		384,
		26,	
		26,
		1,	
		1,
		4,
		24,
		false>		//skip connection flag
	(
		s_net_24_pre_adj,
		s_net_24
	);

	nn2fpga::shift_op <
		t_net_24_struct,
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		c_node_conv_18_ich,
		c_node_conv_18_och,
		c_node_conv_18_ih,
		c_node_conv_18_iw,
		c_node_conv_18_oh,
		c_node_conv_18_ow,
		c_node_conv_18_fh,
		c_node_conv_18_fw,
		c_node_conv_18_stride,
		c_node_conv_18_pad,
		2,
		2,
		c_node_conv_18_ow_ops,
		24,
		24>
	(
		s_net_24[0],
		s_net_24_pre_pad[0],
		s_net_24_data[0]
	);

	nn2fpga::shift_op <
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		c_node_conv_18_ich,
		c_node_conv_18_och,
		c_node_conv_18_ih,
		c_node_conv_18_iw,
		c_node_conv_18_oh,
		c_node_conv_18_ow,
		c_node_conv_18_fh,
		c_node_conv_18_fw,
		c_node_conv_18_stride,
		c_node_conv_18_pad,
		2,
		1,
		c_node_conv_18_ow_ops,
		24,
		24>
	(
		s_net_24_data[0],
		s_net_24_pre_pad[1],
		s_net_24_data[1]
	);

	nn2fpga::shift_op <
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		c_node_conv_18_ich,
		c_node_conv_18_och,
		c_node_conv_18_ih,
		c_node_conv_18_iw,
		c_node_conv_18_oh,
		c_node_conv_18_ow,
		c_node_conv_18_fh,
		c_node_conv_18_fw,
		c_node_conv_18_stride,
		c_node_conv_18_pad,
		2,
		0,
		c_node_conv_18_ow_ops,
		24,
		24>
	(
		s_net_24_data[1],
		s_net_24_pre_pad[2],
		s_net_24_data[2]
	);

	nn2fpga::shift_op <
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		c_node_conv_18_ich,
		c_node_conv_18_och,
		c_node_conv_18_ih,
		c_node_conv_18_iw,
		c_node_conv_18_oh,
		c_node_conv_18_ow,
		c_node_conv_18_fh,
		c_node_conv_18_fw,
		c_node_conv_18_stride,
		c_node_conv_18_pad,
		1,
		2,
		c_node_conv_18_ow_ops,
		24,
		24>
	(
		s_net_24_data[2],
		s_net_24_pre_pad[3],
		s_net_24_data[3]
	);

	nn2fpga::shift_op <
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		c_node_conv_18_ich,
		c_node_conv_18_och,
		c_node_conv_18_ih,
		c_node_conv_18_iw,
		c_node_conv_18_oh,
		c_node_conv_18_ow,
		c_node_conv_18_fh,
		c_node_conv_18_fw,
		c_node_conv_18_stride,
		c_node_conv_18_pad,
		1,
		1,
		c_node_conv_18_ow_ops,
		24,
		24>
	(
		s_net_24_data[3],
		s_net_24_pre_pad[4],
		s_net_24_data[4]
	);

	nn2fpga::shift_op <
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		c_node_conv_18_ich,
		c_node_conv_18_och,
		c_node_conv_18_ih,
		c_node_conv_18_iw,
		c_node_conv_18_oh,
		c_node_conv_18_ow,
		c_node_conv_18_fh,
		c_node_conv_18_fw,
		c_node_conv_18_stride,
		c_node_conv_18_pad,
		1,
		0,
		c_node_conv_18_ow_ops,
		24,
		24>
	(
		s_net_24_data[4],
		s_net_24_pre_pad[5],
		s_net_24_data[5]
	);

	nn2fpga::shift_op <
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		c_node_conv_18_ich,
		c_node_conv_18_och,
		c_node_conv_18_ih,
		c_node_conv_18_iw,
		c_node_conv_18_oh,
		c_node_conv_18_ow,
		c_node_conv_18_fh,
		c_node_conv_18_fw,
		c_node_conv_18_stride,
		c_node_conv_18_pad,
		0,
		2,
		c_node_conv_18_ow_ops,
		24,
		24>
	(
		s_net_24_data[5],
		s_net_24_pre_pad[6],
		s_net_24_data[6]
	);

	nn2fpga::shift_op <
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		c_node_conv_18_ich,
		c_node_conv_18_och,
		c_node_conv_18_ih,
		c_node_conv_18_iw,
		c_node_conv_18_oh,
		c_node_conv_18_ow,
		c_node_conv_18_fh,
		c_node_conv_18_fw,
		c_node_conv_18_stride,
		c_node_conv_18_pad,
		0,
		1,
		c_node_conv_18_ow_ops,
		24,
		24>
	(
		s_net_24_data[6],
		s_net_24_pre_pad[7],
		s_net_24_data[7]
	);

	nn2fpga::shift_op <
		t_net_24_lb_struct,
		t_net_24_lb_struct,
		std::nullptr_t,
		c_node_conv_18_ich,
		c_node_conv_18_och,
		c_node_conv_18_ih,
		c_node_conv_18_iw,
		c_node_conv_18_oh,
		c_node_conv_18_ow,
		c_node_conv_18_fh,
		c_node_conv_18_fw,
		c_node_conv_18_stride,
		c_node_conv_18_pad,
		0,
		0,
		c_node_conv_18_ow_ops,
		24,
		24>
	(
		s_net_24_data[7],
		s_net_24_pre_pad[8],
		s_net_24_null[0]
	);

	nn2fpga::pad_input <
		t_net_24_lb_struct,
		t_net_24_window_struct,
		c_node_conv_18_ich,
		c_node_conv_18_ih,
		c_node_conv_18_iw,
		c_node_conv_18_fh,
		c_node_conv_18_fw,
		c_node_conv_18_stride,
		c_node_conv_18_pad,
		c_node_conv_18_ow_ops,
		24,
		24>
	(
		s_net_24_pre_pad,
		s_net_24_compute
	);

	nn2fpga::conv_comp_wrap <
		t_net_24_window_struct,
		t_net_24_window,
		t_net_24_reduce,
		t_net_24,
		t_node_conv_18_weight,
		t_node_conv_18_weight_mem,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_24_mod,
		t_net_24,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_net_25_acc_struct,
		t_net_25_acc,
		std::nullptr_t,
		std::nullptr_t,
		t_net_25_struct,
		t_net_25_vector,
		t_net_25,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		std::nullptr_t,
		t_params_stream,
		c_node_conv_18_ich,
		c_node_conv_18_ich,
		c_node_conv_18_och,
		c_node_conv_18_och_1x1,
		c_node_conv_18_oh,
		c_node_conv_18_ow,
		c_node_conv_18_fh,
		c_node_conv_18_fw,
		c_node_conv_18_index,
		c_node_conv_18_stride,
		c_node_conv_18_ops,
		c_node_conv_18_ops_out,
		c_node_conv_18_ich_ops,
		1,		//add ops
		c_node_conv_18_ow_ops,
		c_node_conv_18_ow_ops_out,
		16,		//bias_ops
		c_node_conv_18_reuse,
		c_node_conv_18_ow_pack,
		c_node_conv_18_och_pack,
		8,		//act bits
		3,		//act integer bits
		8,		//weight bits
		-1,		//weight integer bits
		2,		//guard bits
		32,		//accumulators in parallel
		31,		//mask bits
		0,		//act bits 1x1
		0,		//act integer bits 1x1
		0,		//weight bits 1x1
		0,		//weight integer bits 1x1
		2,		//guard bits 1x1
		1,		//accumulators in parallel 1x1
		0,		//mask bits 1x1
		0,		//shift cycles
		8,		//width stream
		1,		//weight reads per data
		0,		//bias reads per data
		0,		//weight reads per data 1x1
		0,		//bias reads per data 1x1
		c_node_conv_18_relu,
		0>		//depth
	(
		s_node_conv_15_out,
		s_node_conv_18_init_flag,
		s_node_conv_18_out,
		nullptr,
		c_node_conv_18_weight,
		nullptr,
		nullptr,
		s_net_24_compute,
		(hls::stream<std::nullptr_t>*)(nullptr),
		(hls::stream<std::nullptr_t>*)(nullptr),
		s_net_25,
		(hls::stream<std::nullptr_t>*)(nullptr)
	);
	
	// hls::stream<t_net_25_struct> s_net_25_2[1];

	// nn2fpga::act_tensor_hook <
	// 	t_net_25_struct,
	// 	c_node_conv_18_och,
	// 	c_node_conv_18_oh,
	// 	c_node_conv_18_ow,
	// 	c_node_conv_18_ops_out,
	// 	c_node_conv_18_ow_ops_out>
	// (
	// 	s_net_25,
	// 	s_net_25_2,
	// 	"out_25"
	// );

	// hls::stream<t_net_19_struct> s_net_19_2[1];

	// nn2fpga::act_tensor_hook <
	// 	t_net_19_struct,
	// 	c_consume_stream_node_consume_19_och,
	// 	c_consume_stream_node_consume_19_oh,
	// 	c_consume_stream_node_consume_19_ow,
	// 	16,
	// 	1>
	// (
	// 	s_net_19,
	// 	s_net_19_2,
	// 	"out_19"
	// );
	nn2fpga::consume_stream <
		t_net_19_struct,
		t_net_19,
		c_consume_stream_node_consume_19_och,
		c_consume_stream_node_consume_19_ow,
		c_consume_stream_node_consume_19_oh,
		c_consume_stream_node_consume_19_ow_ops,
		c_consume_stream_node_consume_19_ops>
	(
		s_net_19,
		c_outp1_stream
	);


	nn2fpga::consume_stream <
		t_net_25_struct,
		t_net_25,
		c_consume_stream_node_consume_25_och,
		c_consume_stream_node_consume_25_ow,
		c_consume_stream_node_consume_25_oh,
		c_consume_stream_node_consume_25_ow_ops,
		c_consume_stream_node_consume_25_ops>
	(
		s_net_25,
		c_outp2_stream
	);

	nn2fpga::s2mm<
		t_out_mem1,
		t_out_mem1>(
		o_outp1,
		n_out1,
		c_outp1_stream);

	nn2fpga::s2mm<
		t_out_mem2,
		t_out_mem2>(
		o_outp2,
		n_out2,
		c_outp2_stream);
}

}
#endif