#ifndef __NN2FPGA_NETWORK_PARAMS_H__
#define __NN2FPGA_NETWORK_PARAMS_H__

#include "ap_int.h"
#include "hls_stream.h"
#include "hls_vector.h"
#include "stdint.h"
#include "ap_axi_sdata.h"


/************************* axi_to_stream *************************/
typedef ap_uint<8> t_params_stream;
// typedef ap_axiu<8, 0, 0, 0> t_params_axi_stream;
typedef uint8_t t_params_st;

/************************* produce_stream *************************/
const int c_act_width = 8;
const int c_data_per_packet = 8;
const int c_width_act_stream = 64;
const int c_inp_1 = 64;
typedef ap_uint<64> t_in_mem;
// typedef ap_axiu<64, 0, 0, 0> t_inp_1;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_inp_1_part;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_5;
typedef std::array<t_net_5, 3> t_net_5_vector;
typedef struct {
	std::array<t_net_5_vector, 1> data;
	bool last;
} t_net_5_struct;
const int c_produce_stream_ich = 3;
const int c_produce_stream_iw = 416;
const int c_produce_stream_ih = 416;
const int c_node_produce_0_ich = 3;
const int c_node_produce_0_iw = 416;
const int c_node_produce_0_ih = 416;
const int c_node_produce_0_ops = 3;

/************************* conv_comp_wrap *************************/
typedef ap_fixed<21, 10, AP_RND_ZERO, AP_WRAP> t_net_6_acc;
typedef struct {
	t_net_6_acc data;
	bool last;
} t_net_6_acc_struct;
typedef std::array<t_net_5, 3> t_net_5_reduce;
typedef std::array<t_net_5_reduce, 9> t_net_5_window;
typedef struct {
	t_net_5_window data;
	bool last;
} t_net_5_window_struct;
typedef std::array<t_net_5_reduce, 1> t_net_5_lb;
typedef struct {
	t_net_5_lb data;
	bool last;
} t_net_5_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_6;
typedef std::array<t_net_6, 16> t_net_6_vector;
typedef struct {
	std::array<t_net_6_vector, 1> data;
	bool last;
} t_net_6_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_5_mod;
const int c_node_conv_1_ich = 3;
const int c_node_conv_1_och = 16;
const int c_node_conv_1_och_1x1 = 16;
const int c_node_conv_1_iw = 416;
const int c_node_conv_1_ih = 416;
const int c_node_conv_1_fw = 3;
const int c_node_conv_1_fh = 3;
const int c_node_conv_1_ow = 416;
const int c_node_conv_1_oh = 416;
const int c_node_conv_1_relu = 1;
const int c_node_conv_1_stride = 1;
const int c_node_conv_1_pad = 1;
const int c_node_conv_1_ops = 16;
const int c_node_conv_1_ops_out = 16;
const int c_node_conv_1_in_ops = 3;
const int c_node_conv_1_ich_ops = 3;
const int c_node_conv_1_index = 9;
const int c_node_conv_1_reuse = 1;
const int c_node_conv_1_ow_ops = 1;
const int c_node_conv_1_ow_ops_out = 1;
const int c_node_conv_1_ow_pack = 1;
const int c_node_conv_1_och_pack = 2;
const int c_net_6_add_ops = 16;

/************************* pool_op *************************/
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_7;
typedef std::array<t_net_7, 16> t_net_7_vector;
typedef struct {
	std::array<t_net_7_vector, 1> data;
	bool last;
} t_net_7_struct;
typedef std::array<t_net_6, 16> t_net_6_reduce;
typedef std::array<t_net_6_reduce, 4> t_net_6_window;
typedef struct {
	t_net_6_window data;
	bool last;
} t_net_6_window_struct;
typedef std::array<t_net_6_reduce, 1> t_net_6_lb;
typedef struct {
	t_net_6_lb data;
	bool last;
} t_net_6_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_node_pool_2_acc;
const int c_node_pool_2_ich = 16;
const int c_node_pool_2_och = 16;
const int c_node_pool_2_iw = 416;
const int c_node_pool_2_ih = 416;
const int c_node_pool_2_ow = 208;
const int c_node_pool_2_oh = 208;
const int c_node_pool_2_fw = 2;
const int c_node_pool_2_fh = 2;
const int c_node_pool_2_stride = 2;
const int c_node_pool_2_pad = 0;
const int c_node_pool_2_pool = 1;
const int c_node_pool_2_ow_ops = 1;
const int c_node_pool_2_ops = 16;
const int c_node_pool_2_in_ops = 16;

/************************* bandwidth_adjust *************************/
typedef t_net_7 t_net_7_adj;
typedef std::array<t_net_7, 16> t_net_7_adj_vector;
typedef struct {
	std::array<t_net_7_adj_vector, 1> data;
	bool last;
} t_net_7_adj_struct;
const int c_bandwidth_adjust_net_7_ich = 16;
const int c_bandwidth_adjust_net_7_ih = 208;
const int c_bandwidth_adjust_net_7_iw = 208;
const int c_bandwidth_adjust_net_7_ow_ops_in = 1;
const int c_bandwidth_adjust_net_7_ow_ops = 2;
const int c_bandwidth_adjust_net_7_old_in_ops = 16;
const int c_bandwidth_adjust_net_7_in_ops = 16;

/************************* conv_comp_wrap *************************/
typedef ap_fixed<24, 13, AP_RND_ZERO, AP_WRAP> t_net_8_acc;
typedef struct {
	t_net_8_acc data;
	bool last;
} t_net_8_acc_struct;
typedef std::array<t_net_7, 8> t_net_7_reduce;
typedef std::array<t_net_7_reduce, 12> t_net_7_window;
typedef struct {
	t_net_7_window data;
	bool last;
} t_net_7_window_struct;
typedef std::array<t_net_7_reduce, 1> t_net_7_lb;
typedef struct {
	t_net_7_lb data;
	bool last;
} t_net_7_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_8;
typedef std::array<t_net_8, 16> t_net_8_vector;
typedef struct {
	std::array<t_net_8_vector, 1> data;
	bool last;
} t_net_8_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_7_mod;
const int c_node_conv_3_ich = 16;
const int c_node_conv_3_och = 32;
const int c_node_conv_3_och_1x1 = 32;
const int c_node_conv_3_iw = 208;
const int c_node_conv_3_ih = 208;
const int c_node_conv_3_fw = 3;
const int c_node_conv_3_fh = 3;
const int c_node_conv_3_ow = 208;
const int c_node_conv_3_oh = 208;
const int c_node_conv_3_relu = 1;
const int c_node_conv_3_stride = 1;
const int c_node_conv_3_pad = 1;
const int c_node_conv_3_ops = 8;
const int c_node_conv_3_ops_out = 16;
const int c_node_conv_3_in_ops = 16;
const int c_node_conv_3_ich_ops = 8;
const int c_node_conv_3_index = 9;
const int c_node_conv_3_reuse = 2;
const int c_node_conv_3_ow_ops = 2;
const int c_node_conv_3_ow_ops_out = 2;
const int c_node_conv_3_ow_pack = 2;
const int c_node_conv_3_och_pack = 1;
const int c_net_8_add_ops = 16;

/************************* bandwidth_adjust *************************/
typedef t_net_8 t_net_8_adj;
typedef std::array<t_net_8, 16> t_net_8_adj_vector;
typedef struct {
	std::array<t_net_8_adj_vector, 1> data;
	bool last;
} t_net_8_adj_struct;
const int c_bandwidth_adjust_net_8_ich = 32;
const int c_bandwidth_adjust_net_8_ih = 208;
const int c_bandwidth_adjust_net_8_iw = 208;
const int c_bandwidth_adjust_net_8_ow_ops_in = 2;
const int c_bandwidth_adjust_net_8_ow_ops = 1;
const int c_bandwidth_adjust_net_8_old_in_ops = 16;
const int c_bandwidth_adjust_net_8_in_ops = 16;

/************************* pool_op *************************/
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_9;
typedef std::array<t_net_9, 16> t_net_9_vector;
typedef struct {
	std::array<t_net_9_vector, 1> data;
	bool last;
} t_net_9_struct;
typedef std::array<t_net_8, 16> t_net_8_reduce;
typedef std::array<t_net_8_reduce, 4> t_net_8_window;
typedef struct {
	t_net_8_window data;
	bool last;
} t_net_8_window_struct;
typedef std::array<t_net_8_reduce, 1> t_net_8_lb;
typedef struct {
	t_net_8_lb data;
	bool last;
} t_net_8_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_node_pool_4_acc;
const int c_node_pool_4_ich = 32;
const int c_node_pool_4_och = 32;
const int c_node_pool_4_iw = 208;
const int c_node_pool_4_ih = 208;
const int c_node_pool_4_ow = 104;
const int c_node_pool_4_oh = 104;
const int c_node_pool_4_fw = 2;
const int c_node_pool_4_fh = 2;
const int c_node_pool_4_stride = 2;
const int c_node_pool_4_pad = 0;
const int c_node_pool_4_pool = 1;
const int c_node_pool_4_ow_ops = 1;
const int c_node_pool_4_ops = 16;
const int c_node_pool_4_in_ops = 16;

/************************* bandwidth_adjust *************************/
typedef t_net_9 t_net_9_adj;
typedef std::array<t_net_9, 16> t_net_9_adj_vector;
typedef struct {
	std::array<t_net_9_adj_vector, 1> data;
	bool last;
} t_net_9_adj_struct;
const int c_bandwidth_adjust_net_9_ich = 32;
const int c_bandwidth_adjust_net_9_ih = 104;
const int c_bandwidth_adjust_net_9_iw = 104;
const int c_bandwidth_adjust_net_9_ow_ops_in = 1;
const int c_bandwidth_adjust_net_9_ow_ops = 2;
const int c_bandwidth_adjust_net_9_old_in_ops = 16;
const int c_bandwidth_adjust_net_9_in_ops = 16;

/************************* conv_comp_wrap *************************/
typedef ap_fixed<25, 13, AP_RND_ZERO, AP_WRAP> t_net_10_acc;
typedef struct {
	t_net_10_acc data;
	bool last;
} t_net_10_acc_struct;
typedef std::array<t_net_9, 8> t_net_9_reduce;
typedef std::array<t_net_9_reduce, 12> t_net_9_window;
typedef struct {
	t_net_9_window data;
	bool last;
} t_net_9_window_struct;
typedef std::array<t_net_9_reduce, 1> t_net_9_lb;
typedef struct {
	t_net_9_lb data;
	bool last;
} t_net_9_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_10;
typedef std::array<t_net_10, 8> t_net_10_vector;
typedef struct {
	std::array<t_net_10_vector, 1> data;
	bool last;
} t_net_10_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_9_mod;
const int c_node_conv_5_ich = 32;
const int c_node_conv_5_och = 64;
const int c_node_conv_5_och_1x1 = 64;
const int c_node_conv_5_iw = 104;
const int c_node_conv_5_ih = 104;
const int c_node_conv_5_fw = 3;
const int c_node_conv_5_fh = 3;
const int c_node_conv_5_ow = 104;
const int c_node_conv_5_oh = 104;
const int c_node_conv_5_relu = 1;
const int c_node_conv_5_stride = 1;
const int c_node_conv_5_pad = 1;
const int c_node_conv_5_ops = 8;
const int c_node_conv_5_ops_out = 8;
const int c_node_conv_5_in_ops = 16;
const int c_node_conv_5_ich_ops = 8;
const int c_node_conv_5_index = 9;
const int c_node_conv_5_reuse = 2;
const int c_node_conv_5_ow_ops = 2;
const int c_node_conv_5_ow_ops_out = 2;
const int c_node_conv_5_ow_pack = 2;
const int c_node_conv_5_och_pack = 1;
const int c_net_10_add_ops = 8;

/************************* bandwidth_adjust *************************/
typedef t_net_10 t_net_10_adj;
typedef std::array<t_net_10, 8> t_net_10_adj_vector;
typedef struct {
	std::array<t_net_10_adj_vector, 1> data;
	bool last;
} t_net_10_adj_struct;
const int c_bandwidth_adjust_net_10_ich = 64;
const int c_bandwidth_adjust_net_10_ih = 104;
const int c_bandwidth_adjust_net_10_iw = 104;
const int c_bandwidth_adjust_net_10_ow_ops_in = 2;
const int c_bandwidth_adjust_net_10_ow_ops = 1;
const int c_bandwidth_adjust_net_10_old_in_ops = 8;
const int c_bandwidth_adjust_net_10_in_ops = 8;

/************************* pool_op *************************/
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_11;
typedef std::array<t_net_11, 8> t_net_11_vector;
typedef struct {
	std::array<t_net_11_vector, 1> data;
	bool last;
} t_net_11_struct;
typedef std::array<t_net_10, 8> t_net_10_reduce;
typedef std::array<t_net_10_reduce, 4> t_net_10_window;
typedef struct {
	t_net_10_window data;
	bool last;
} t_net_10_window_struct;
typedef std::array<t_net_10_reduce, 1> t_net_10_lb;
typedef struct {
	t_net_10_lb data;
	bool last;
} t_net_10_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_node_pool_6_acc;
const int c_node_pool_6_ich = 64;
const int c_node_pool_6_och = 64;
const int c_node_pool_6_iw = 104;
const int c_node_pool_6_ih = 104;
const int c_node_pool_6_ow = 52;
const int c_node_pool_6_oh = 52;
const int c_node_pool_6_fw = 2;
const int c_node_pool_6_fh = 2;
const int c_node_pool_6_stride = 2;
const int c_node_pool_6_pad = 0;
const int c_node_pool_6_pool = 1;
const int c_node_pool_6_ow_ops = 1;
const int c_node_pool_6_ops = 8;
const int c_node_pool_6_in_ops = 8;

/************************* bandwidth_adjust *************************/
typedef t_net_11 t_net_11_adj;
typedef std::array<t_net_11, 8> t_net_11_adj_vector;
typedef struct {
	std::array<t_net_11_adj_vector, 1> data;
	bool last;
} t_net_11_adj_struct;
const int c_bandwidth_adjust_net_11_ich = 64;
const int c_bandwidth_adjust_net_11_ih = 52;
const int c_bandwidth_adjust_net_11_iw = 52;
const int c_bandwidth_adjust_net_11_ow_ops_in = 1;
const int c_bandwidth_adjust_net_11_ow_ops = 2;
const int c_bandwidth_adjust_net_11_old_in_ops = 8;
const int c_bandwidth_adjust_net_11_in_ops = 8;

/************************* conv_comp_wrap *************************/
typedef ap_fixed<26, 14, AP_RND_ZERO, AP_WRAP> t_net_12_acc;
typedef struct {
	t_net_12_acc data;
	bool last;
} t_net_12_acc_struct;
typedef std::array<t_net_11, 8> t_net_11_reduce;
typedef std::array<t_net_11_reduce, 12> t_net_11_window;
typedef struct {
	t_net_11_window data;
	bool last;
} t_net_11_window_struct;
typedef std::array<t_net_11_reduce, 1> t_net_11_lb;
typedef struct {
	t_net_11_lb data;
	bool last;
} t_net_11_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_12;
typedef std::array<t_net_12, 8> t_net_12_vector;
typedef struct {
	std::array<t_net_12_vector, 1> data;
	bool last;
} t_net_12_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_11_mod;
const int c_node_conv_7_ich = 64;
const int c_node_conv_7_och = 128;
const int c_node_conv_7_och_1x1 = 128;
const int c_node_conv_7_iw = 52;
const int c_node_conv_7_ih = 52;
const int c_node_conv_7_fw = 3;
const int c_node_conv_7_fh = 3;
const int c_node_conv_7_ow = 52;
const int c_node_conv_7_oh = 52;
const int c_node_conv_7_relu = 1;
const int c_node_conv_7_stride = 1;
const int c_node_conv_7_pad = 1;
const int c_node_conv_7_ops = 8;
const int c_node_conv_7_ops_out = 8;
const int c_node_conv_7_in_ops = 8;
const int c_node_conv_7_ich_ops = 8;
const int c_node_conv_7_index = 9;
const int c_node_conv_7_reuse = 2;
const int c_node_conv_7_ow_ops = 2;
const int c_node_conv_7_ow_ops_out = 2;
const int c_node_conv_7_ow_pack = 2;
const int c_node_conv_7_och_pack = 1;
const int c_net_12_add_ops = 8;

/************************* bandwidth_adjust *************************/
typedef t_net_12 t_net_12_adj;
typedef std::array<t_net_12, 8> t_net_12_adj_vector;
typedef struct {
	std::array<t_net_12_adj_vector, 1> data;
	bool last;
} t_net_12_adj_struct;
const int c_bandwidth_adjust_net_12_ich = 128;
const int c_bandwidth_adjust_net_12_ih = 52;
const int c_bandwidth_adjust_net_12_iw = 52;
const int c_bandwidth_adjust_net_12_ow_ops_in = 2;
const int c_bandwidth_adjust_net_12_ow_ops = 1;
const int c_bandwidth_adjust_net_12_old_in_ops = 8;
const int c_bandwidth_adjust_net_12_in_ops = 8;

/************************* pool_op *************************/
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_13;
typedef std::array<t_net_13, 8> t_net_13_vector;
typedef struct {
	std::array<t_net_13_vector, 1> data;
	bool last;
} t_net_13_struct;
typedef std::array<t_net_12, 8> t_net_12_reduce;
typedef std::array<t_net_12_reduce, 4> t_net_12_window;
typedef struct {
	t_net_12_window data;
	bool last;
} t_net_12_window_struct;
typedef std::array<t_net_12_reduce, 1> t_net_12_lb;
typedef struct {
	t_net_12_lb data;
	bool last;
} t_net_12_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_node_pool_8_acc;
const int c_node_pool_8_ich = 128;
const int c_node_pool_8_och = 128;
const int c_node_pool_8_iw = 52;
const int c_node_pool_8_ih = 52;
const int c_node_pool_8_ow = 26;
const int c_node_pool_8_oh = 26;
const int c_node_pool_8_fw = 2;
const int c_node_pool_8_fh = 2;
const int c_node_pool_8_stride = 2;
const int c_node_pool_8_pad = 0;
const int c_node_pool_8_pool = 1;
const int c_node_pool_8_ow_ops = 1;
const int c_node_pool_8_ops = 8;
const int c_node_pool_8_in_ops = 8;

/************************* bandwidth_adjust *************************/
typedef t_net_13 t_net_13_adj;
typedef std::array<t_net_13, 8> t_net_13_adj_vector;
typedef struct {
	std::array<t_net_13_adj_vector, 1> data;
	bool last;
} t_net_13_adj_struct;
const int c_bandwidth_adjust_net_13_ich = 128;
const int c_bandwidth_adjust_net_13_ih = 26;
const int c_bandwidth_adjust_net_13_iw = 26;
const int c_bandwidth_adjust_net_13_ow_ops_in = 1;
const int c_bandwidth_adjust_net_13_ow_ops = 2;
const int c_bandwidth_adjust_net_13_old_in_ops = 8;
const int c_bandwidth_adjust_net_13_in_ops = 8;

/************************* conv_comp_wrap *************************/
typedef ap_fixed<27, 14, AP_RND_ZERO, AP_WRAP> t_net_14_acc;
typedef struct {
	t_net_14_acc data;
	bool last;
} t_net_14_acc_struct;
typedef std::array<t_net_13, 8> t_net_13_reduce;
typedef std::array<t_net_13_reduce, 12> t_net_13_window;
typedef struct {
	t_net_13_window data;
	bool last;
} t_net_13_window_struct;
typedef std::array<t_net_13_reduce, 1> t_net_13_lb;
typedef struct {
	t_net_13_lb data;
	bool last;
} t_net_13_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_14;
typedef std::array<t_net_14, 8> t_net_14_vector;
typedef struct {
	std::array<t_net_14_vector, 1> data;
	bool last;
} t_net_14_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_13_mod;
const int c_node_conv_9_ich = 128;
const int c_node_conv_9_och = 256;
const int c_node_conv_9_och_1x1 = 256;
const int c_node_conv_9_iw = 26;
const int c_node_conv_9_ih = 26;
const int c_node_conv_9_fw = 3;
const int c_node_conv_9_fh = 3;
const int c_node_conv_9_ow = 26;
const int c_node_conv_9_oh = 26;
const int c_node_conv_9_relu = 1;
const int c_node_conv_9_stride = 1;
const int c_node_conv_9_pad = 1;
const int c_node_conv_9_ops = 8;
const int c_node_conv_9_ops_out = 8;
const int c_node_conv_9_in_ops = 8;
const int c_node_conv_9_ich_ops = 8;
const int c_node_conv_9_index = 9;
const int c_node_conv_9_reuse = 2;
const int c_node_conv_9_ow_ops = 2;
const int c_node_conv_9_ow_ops_out = 2;
const int c_node_conv_9_ow_pack = 2;
const int c_node_conv_9_och_pack = 1;
const int c_net_14_add_ops = 8;

/************************* tensor_duplicator *************************/
using t_net_14_dup_0=t_net_14;
using t_net_14_dup_0_struct=t_net_14_struct;
using t_net_14_dup_0_vector=t_net_14_vector;
const int c_net_14_dup_0_add_ops = c_net_14_add_ops;
using t_net_14_dup_1=t_net_14;
using t_net_14_dup_1_struct=t_net_14_struct;
using t_net_14_dup_1_vector=t_net_14_vector;
const int c_net_14_dup_1_add_ops = c_net_14_add_ops;

/************************* bandwidth_adjust *************************/
typedef t_net_14_dup_0 t_net_14_dup_0_adj;
typedef std::array<t_net_14_dup_0, 8> t_net_14_dup_0_adj_vector;
typedef struct {
	std::array<t_net_14_dup_0_adj_vector, 1> data;
	bool last;
} t_net_14_dup_0_adj_struct;
const int c_bandwidth_adjust_net_14_dup_0_ich = 256;
const int c_bandwidth_adjust_net_14_dup_0_ih = 26;
const int c_bandwidth_adjust_net_14_dup_0_iw = 26;
const int c_bandwidth_adjust_net_14_dup_0_ow_ops_in = 2;
const int c_bandwidth_adjust_net_14_dup_0_ow_ops = 1;
const int c_bandwidth_adjust_net_14_dup_0_old_in_ops = 8;
const int c_bandwidth_adjust_net_14_dup_0_in_ops = 8;

/************************* pool_op *************************/
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_15;
typedef std::array<t_net_15, 8> t_net_15_vector;
typedef struct {
	std::array<t_net_15_vector, 1> data;
	bool last;
} t_net_15_struct;
typedef std::array<t_net_14_dup_0, 8> t_net_14_dup_0_reduce;
typedef std::array<t_net_14_dup_0_reduce, 4> t_net_14_dup_0_window;
typedef struct {
	t_net_14_dup_0_window data;
	bool last;
} t_net_14_dup_0_window_struct;
typedef std::array<t_net_14_dup_0_reduce, 1> t_net_14_dup_0_lb;
typedef struct {
	t_net_14_dup_0_lb data;
	bool last;
} t_net_14_dup_0_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_node_pool_10_acc;
const int c_node_pool_10_ich = 256;
const int c_node_pool_10_och = 256;
const int c_node_pool_10_iw = 26;
const int c_node_pool_10_ih = 26;
const int c_node_pool_10_ow = 13;
const int c_node_pool_10_oh = 13;
const int c_node_pool_10_fw = 2;
const int c_node_pool_10_fh = 2;
const int c_node_pool_10_stride = 2;
const int c_node_pool_10_pad = 0;
const int c_node_pool_10_pool = 1;
const int c_node_pool_10_ow_ops = 1;
const int c_node_pool_10_ops = 8;
const int c_node_pool_10_in_ops = 8;

/************************* conv_comp_wrap *************************/
typedef ap_fixed<28, 15, AP_RND_ZERO, AP_WRAP> t_net_16_acc;
typedef struct {
	t_net_16_acc data;
	bool last;
} t_net_16_acc_struct;
typedef std::array<t_net_15, 8> t_net_15_reduce;
typedef std::array<t_net_15_reduce, 9> t_net_15_window;
typedef struct {
	t_net_15_window data;
	bool last;
} t_net_15_window_struct;
typedef std::array<t_net_15_reduce, 1> t_net_15_lb;
typedef struct {
	t_net_15_lb data;
	bool last;
} t_net_15_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_16;
typedef std::array<t_net_16, 32> t_net_16_vector;
typedef struct {
	std::array<t_net_16_vector, 1> data;
	bool last;
} t_net_16_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_15_mod;
const int c_node_conv_11_ich = 256;
const int c_node_conv_11_och = 512;
const int c_node_conv_11_och_1x1 = 512;
const int c_node_conv_11_iw = 13;
const int c_node_conv_11_ih = 13;
const int c_node_conv_11_fw = 3;
const int c_node_conv_11_fh = 3;
const int c_node_conv_11_ow = 13;
const int c_node_conv_11_oh = 13;
const int c_node_conv_11_relu = 1;
const int c_node_conv_11_stride = 1;
const int c_node_conv_11_pad = 1;
const int c_node_conv_11_ops = 16;
const int c_node_conv_11_ops_out = 32;
const int c_node_conv_11_in_ops = 8;
const int c_node_conv_11_ich_ops = 8;
const int c_node_conv_11_index = 9;
const int c_node_conv_11_reuse = 1;
const int c_node_conv_11_ow_ops = 1;
const int c_node_conv_11_ow_ops_out = 1;
const int c_node_conv_11_ow_pack = 1;
const int c_node_conv_11_och_pack = 2;
const int c_net_16_add_ops = 32;

/************************* conv_comp_wrap *************************/
typedef ap_fixed<29, 15, AP_RND_ZERO, AP_WRAP> t_net_17_acc;
typedef struct {
	t_net_17_acc data;
	bool last;
} t_net_17_acc_struct;
typedef std::array<t_net_16, 32> t_net_16_reduce;
typedef std::array<t_net_16_reduce, 9> t_net_16_window;
typedef struct {
	t_net_16_window data;
	bool last;
} t_net_16_window_struct;
typedef std::array<t_net_16_reduce, 1> t_net_16_lb;
typedef struct {
	t_net_16_lb data;
	bool last;
} t_net_16_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_17;
typedef std::array<t_net_17, 16> t_net_17_vector;
typedef struct {
	std::array<t_net_17_vector, 1> data;
	bool last;
} t_net_17_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_16_mod;
const int c_node_conv_12_ich = 512;
const int c_node_conv_12_och = 1024;
const int c_node_conv_12_och_1x1 = 1024;
const int c_node_conv_12_iw = 13;
const int c_node_conv_12_ih = 13;
const int c_node_conv_12_fw = 3;
const int c_node_conv_12_fh = 3;
const int c_node_conv_12_ow = 13;
const int c_node_conv_12_oh = 13;
const int c_node_conv_12_relu = 1;
const int c_node_conv_12_stride = 1;
const int c_node_conv_12_pad = 1;
const int c_node_conv_12_ops = 16;
const int c_node_conv_12_ops_out = 16;
const int c_node_conv_12_in_ops = 32;
const int c_node_conv_12_ich_ops = 32;
const int c_node_conv_12_index = 9;
const int c_node_conv_12_reuse = 1;
const int c_node_conv_12_ow_ops = 1;
const int c_node_conv_12_ow_ops_out = 1;
const int c_node_conv_12_ow_pack = 1;
const int c_node_conv_12_och_pack = 2;
const int c_net_17_add_ops = 16;

/************************* conv_comp_wrap *************************/
typedef ap_fixed<26, 13, AP_RND_ZERO, AP_WRAP> t_net_18_acc;
typedef struct {
	t_net_18_acc data;
	bool last;
} t_net_18_acc_struct;
typedef std::array<t_net_17, 16> t_net_17_reduce;
typedef std::array<t_net_17_reduce, 1> t_net_17_window;
typedef struct {
	t_net_17_window data;
	bool last;
} t_net_17_window_struct;
typedef std::array<t_net_17_reduce, 1> t_net_17_lb;
typedef struct {
	t_net_17_lb data;
	bool last;
} t_net_17_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_18;
typedef std::array<t_net_18, 16> t_net_18_vector;
typedef struct {
	std::array<t_net_18_vector, 1> data;
	bool last;
} t_net_18_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_17_mod;
const int c_node_conv_13_ich = 1024;
const int c_node_conv_13_och = 256;
const int c_node_conv_13_och_1x1 = 256;
const int c_node_conv_13_iw = 13;
const int c_node_conv_13_ih = 13;
const int c_node_conv_13_fw = 1;
const int c_node_conv_13_fh = 1;
const int c_node_conv_13_ow = 13;
const int c_node_conv_13_oh = 13;
const int c_node_conv_13_relu = 1;
const int c_node_conv_13_stride = 1;
const int c_node_conv_13_pad = 0;
const int c_node_conv_13_ops = 16;
const int c_node_conv_13_ops_out = 16;
const int c_node_conv_13_in_ops = 16;
const int c_node_conv_13_ich_ops = 16;
const int c_node_conv_13_index = 1;
const int c_node_conv_13_reuse = 1;
const int c_node_conv_13_ow_ops = 1;
const int c_node_conv_13_ow_ops_out = 1;
const int c_node_conv_13_ow_pack = 1;
const int c_node_conv_13_och_pack = 2;
const int c_net_18_add_ops = 16;

/************************* tensor_duplicator *************************/
using t_net_18_dup_0=t_net_18;
using t_net_18_dup_0_struct=t_net_18_struct;
using t_net_18_dup_0_vector=t_net_18_vector;
const int c_net_18_dup_0_add_ops = c_net_18_add_ops;
using t_net_18_dup_1=t_net_18;
using t_net_18_dup_1_struct=t_net_18_struct;
using t_net_18_dup_1_vector=t_net_18_vector;
const int c_net_18_dup_1_add_ops = c_net_18_add_ops;

/************************* conv_comp_wrap *************************/
typedef ap_fixed<28, 14, AP_RND_ZERO, AP_WRAP> t_net_19_acc;
typedef struct {
	t_net_19_acc data;
	bool last;
} t_net_19_acc_struct;
typedef std::array<t_net_18_dup_0, 8> t_net_18_dup_0_reduce;
typedef std::array<t_net_18_dup_0_reduce, 9> t_net_18_dup_0_window;
typedef struct {
	t_net_18_dup_0_window data;
	bool last;
} t_net_18_dup_0_window_struct;
typedef std::array<t_net_18_dup_0_reduce, 1> t_net_18_dup_0_lb;
typedef struct {
	t_net_18_dup_0_lb data;
	bool last;
} t_net_18_dup_0_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_19;
typedef std::array<t_net_19, 16> t_net_19_vector;
typedef struct {
	std::array<t_net_19_vector, 1> data;
	bool last;
} t_net_19_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_18_dup_0_mod;
const int c_node_conv_14_ich = 256;
const int c_node_conv_14_och = 512;
const int c_node_conv_14_och_1x1 = 512;
const int c_node_conv_14_iw = 13;
const int c_node_conv_14_ih = 13;
const int c_node_conv_14_fw = 3;
const int c_node_conv_14_fh = 3;
const int c_node_conv_14_ow = 13;
const int c_node_conv_14_oh = 13;
const int c_node_conv_14_relu = 1;
const int c_node_conv_14_stride = 1;
const int c_node_conv_14_pad = 1;
const int c_node_conv_14_ops = 16;
const int c_node_conv_14_ops_out = 16;
const int c_node_conv_14_in_ops = 16;
const int c_node_conv_14_ich_ops = 8;
const int c_node_conv_14_index = 9;
const int c_node_conv_14_reuse = 1;
const int c_node_conv_14_ow_ops = 1;
const int c_node_conv_14_ow_ops_out = 1;
const int c_node_conv_14_ow_pack = 1;
const int c_node_conv_14_och_pack = 2;
const int c_net_19_add_ops = 16;

/************************* conv_comp_wrap *************************/
typedef ap_fixed<24, 11, AP_RND_ZERO, AP_WRAP> t_net_20_acc;
typedef struct {
	t_net_20_acc data;
	bool last;
} t_net_20_acc_struct;
typedef std::array<t_net_18_dup_1, 8> t_net_18_dup_1_reduce;
typedef std::array<t_net_18_dup_1_reduce, 1> t_net_18_dup_1_window;
typedef struct {
	t_net_18_dup_1_window data;
	bool last;
} t_net_18_dup_1_window_struct;
typedef std::array<t_net_18_dup_1_reduce, 1> t_net_18_dup_1_lb;
typedef struct {
	t_net_18_dup_1_lb data;
	bool last;
} t_net_18_dup_1_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_20;
typedef std::array<t_net_20, 4> t_net_20_vector;
typedef struct {
	std::array<t_net_20_vector, 1> data;
	bool last;
} t_net_20_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_18_dup_1_mod;
const int c_node_conv_15_ich = 256;
const int c_node_conv_15_och = 128;
const int c_node_conv_15_och_1x1 = 128;
const int c_node_conv_15_iw = 13;
const int c_node_conv_15_ih = 13;
const int c_node_conv_15_fw = 1;
const int c_node_conv_15_fh = 1;
const int c_node_conv_15_ow = 13;
const int c_node_conv_15_oh = 13;
const int c_node_conv_15_relu = 1;
const int c_node_conv_15_stride = 1;
const int c_node_conv_15_pad = 0;
const int c_node_conv_15_ops = 4;
const int c_node_conv_15_ops_out = 4;
const int c_node_conv_15_in_ops = 1;
const int c_node_conv_15_ich_ops = 8;
const int c_node_conv_15_index = 1;
const int c_node_conv_15_reuse = 1;
const int c_node_conv_15_ow_ops = 1;
const int c_node_conv_15_ow_ops_out = 1;
const int c_node_conv_15_ow_pack = 1;
const int c_node_conv_15_och_pack = 2;
const int c_net_20_add_ops = 4;

/************************* upsample_op *************************/
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_23;
typedef std::array<t_net_23, 4> t_net_23_vector;
typedef struct {
	std::array<t_net_23_vector, 1> data;
	bool last;
} t_net_23_struct;
const int c_node_upsample_16_ich = 128;
const int c_node_upsample_16_ih = 13;
const int c_node_upsample_16_iw = 13;
const int c_node_upsample_16_factor = 2;
const int c_node_upsample_16_ow_ops = 1;
const int c_node_upsample_16_scale_factor = 0;

/************************* concat_op *************************/
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_24;
typedef std::array<t_net_24, 24> t_net_24_vector;
typedef struct {
	std::array<t_net_24_vector, 1> data;
	bool last;
} t_net_24_struct;
const int c_node_concat_17_feature_map = 676;
const int c_node_concat_17_scale_factor = 0;
const int c_node_concat_17_ow_ops_in = 1;
const int c_node_concat_17_ow_ops_out = 1;

/************************* conv_comp_wrap *************************/
typedef ap_fixed<28, 14, AP_RND_ZERO, AP_WRAP> t_net_25_acc;
typedef struct {
	t_net_25_acc data;
	bool last;
} t_net_25_acc_struct;
typedef std::array<t_net_24, 24> t_net_24_reduce;
typedef std::array<t_net_24_reduce, 9> t_net_24_window;
typedef struct {
	t_net_24_window data;
	bool last;
} t_net_24_window_struct;
typedef std::array<t_net_24_reduce, 1> t_net_24_lb;
typedef struct {
	t_net_24_lb data;
	bool last;
} t_net_24_lb_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_25;
typedef std::array<t_net_25, 16> t_net_25_vector;
typedef struct {
	std::array<t_net_25_vector, 1> data;
	bool last;
} t_net_25_struct;
typedef ap_ufixed<8, 3, AP_RND_CONV, AP_SAT> t_net_24_mod;
const int c_node_conv_18_ich = 384;
const int c_node_conv_18_och = 256;
const int c_node_conv_18_och_1x1 = 256;
const int c_node_conv_18_iw = 26;
const int c_node_conv_18_ih = 26;
const int c_node_conv_18_fw = 3;
const int c_node_conv_18_fh = 3;
const int c_node_conv_18_ow = 26;
const int c_node_conv_18_oh = 26;
const int c_node_conv_18_relu = 1;
const int c_node_conv_18_stride = 1;
const int c_node_conv_18_pad = 1;
const int c_node_conv_18_ops = 16;
const int c_node_conv_18_ops_out = 16;
const int c_node_conv_18_in_ops = 24;
const int c_node_conv_18_ich_ops = 24;
const int c_node_conv_18_index = 9;
const int c_node_conv_18_reuse = 1;
const int c_node_conv_18_ow_ops = 1;
const int c_node_conv_18_ow_ops_out = 1;
const int c_node_conv_18_ow_pack = 1;
const int c_node_conv_18_och_pack = 2;
const int c_net_25_add_ops = 16;

/************************* consume_stream *************************/
// typedef hls::axis<t_net_25, 0, 0, 0> t_o_net_25;
using t_out_mem2=t_net_25;
// using t_o_outp2=t_o_net_25;
// using t_o_data=t_o_net_25;
const int c_consume_stream_node_consume_25_och = 512;
const int c_consume_stream_node_consume_25_ow = 13;
const int c_consume_stream_node_consume_25_oh = 13;
const int c_consume_stream_node_consume_25_ow_ops = 1;
const int c_consume_stream_node_consume_25_ops = 16;

// typedef hls::axis<t_net_19, 0, 0, 0> t_o_net_19;
using t_out_mem1=t_net_19;
// using t_o_outp1=t_o_net_19;
// using t_o_data=t_o_net_19;
const int c_consume_stream_node_consume_19_och = 256;
const int c_consume_stream_node_consume_19_ow = 26;
const int c_consume_stream_node_consume_19_oh = 26;
const int c_consume_stream_node_consume_19_ow_ops = 1;
const int c_consume_stream_node_consume_19_ops = 16;

/************************* fake_func_params *************************/
typedef ap_fixed<8, 2, AP_RND_ZERO, AP_SAT_SYM> t_node_conv_1_weight_mem;
typedef std::array<std::array<t_node_conv_1_weight_mem, 16>, 3> t_node_conv_1_weight;

/************************* fake_func_params *************************/
typedef ap_fixed<8, 2, AP_RND_ZERO, AP_SAT_SYM> t_node_conv_3_weight_mem;
typedef std::array<std::array<t_node_conv_3_weight_mem, 8>, 8> t_node_conv_3_weight;

/************************* fake_func_params *************************/
typedef ap_fixed<8, 1, AP_RND_ZERO, AP_SAT_SYM> t_node_conv_5_weight_mem;
typedef std::array<std::array<t_node_conv_5_weight_mem, 8>, 8> t_node_conv_5_weight;

/************************* fake_func_params *************************/
typedef ap_fixed<8, 1, AP_RND_ZERO, AP_SAT_SYM> t_node_conv_7_weight_mem;
typedef std::array<std::array<t_node_conv_7_weight_mem, 8>, 8> t_node_conv_7_weight;

/************************* fake_func_params *************************/
typedef ap_fixed<8, 0, AP_RND_ZERO, AP_SAT_SYM> t_node_conv_9_weight_mem;
typedef std::array<std::array<t_node_conv_9_weight_mem, 8>, 8> t_node_conv_9_weight;

/************************* fake_func_params *************************/
typedef ap_fixed<8, 0, AP_RND_ZERO, AP_SAT_SYM> t_node_conv_11_weight_mem;
typedef std::array<std::array<t_node_conv_11_weight_mem, 16>, 8> t_node_conv_11_weight;

/************************* fake_func_params *************************/
typedef ap_fixed<8, -1, AP_RND_ZERO, AP_SAT_SYM> t_node_conv_12_weight_mem;
typedef std::array<std::array<t_node_conv_12_weight_mem, 16>, 32> t_node_conv_12_weight;

/************************* fake_func_params *************************/
typedef ap_fixed<8, 0, AP_RND_ZERO, AP_SAT_SYM> t_node_conv_13_weight_mem;
typedef std::array<std::array<t_node_conv_13_weight_mem, 16>, 16> t_node_conv_13_weight;

/************************* fake_func_params *************************/
typedef ap_fixed<8, -1, AP_RND_ZERO, AP_SAT_SYM> t_node_conv_14_weight_mem;
typedef std::array<std::array<t_node_conv_14_weight_mem, 16>, 8> t_node_conv_14_weight;

/************************* fake_func_params *************************/
typedef ap_fixed<8, 0, AP_RND_ZERO, AP_SAT_SYM> t_node_conv_15_weight_mem;
typedef std::array<std::array<t_node_conv_15_weight_mem, 4>, 8> t_node_conv_15_weight;

/************************* fake_func_params *************************/
typedef ap_fixed<8, -1, AP_RND_ZERO, AP_SAT_SYM> t_node_conv_18_weight_mem;
typedef std::array<std::array<t_node_conv_18_weight_mem, 16>, 24> t_node_conv_18_weight;

#endif /*__NN2FPGA_NETWORK_PARAMS_H__ */