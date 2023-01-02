#ifndef __NETWORK__
#define __NETWORK__
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"
#include <stdint.h>
#define c_i_data 64
typedef ap_axiu<c_i_data, 0, 0, 0> t_i_data;
#define c_o_data 32
typedef ap_axis<c_o_data, 0, 0, 0> t_o_data;
typedef ap_uint<1> t_last;
typedef uint8_t t_input;

const int c_input_ich    = 3;
const int c_input_ih     = 32;
const int c_input_iw     = 32;


const int c_output_och = 10;
const int c_output_oh  = 1;
const int c_output_ow  = 1;


const int c_conv_0_ops  = 8;

typedef uint64_t t_conv0_weight_st;
typedef ap_uint<8*c_conv_0_ops> t_conv0_weight;
const int c_conv0_weight_och = 16;
const int c_conv0_weight_ich = 3;
const int c_conv0_weight_ih  = 3;
const int c_conv0_weight_iw  = 3;
const int c_conv0_weight_ops = 8;
const int c_conv0_weight_index = 9;
const int c_conv0_weight_iter  = 7;

typedef uint8_t t_conv_5;
typedef ap_int<32> t_conv_0_acc;
const int c_conv_0_ich    = 3;
const int c_conv_0_och    = 16;
const int c_conv_0_ih     = 32;
const int c_conv_0_iw     = 32;
const int c_conv_0_ow     = 16;
const int c_conv_0_oh     = 16;
const int c_conv_0_fw     = 3;
const int c_conv_0_fh     = 3;
const int c_conv_0_relu   = 1;
const int c_conv_5_a_split  = 0;
const int c_conv_0_stride = 2;
const int c_conv_0_pad    = 1;
const int c_conv_0_split  = 2;


typedef uint8_t t_conv_5;
const int c_relu_1_ich    = 16;
const int c_relu_1_ih     = 16;
const int c_relu_1_iw     = 16;


const int c_conv_2_ops  = 8;

typedef uint64_t t_conv1_weight_st;
typedef ap_uint<8*c_conv_2_ops> t_conv1_weight;
const int c_conv1_weight_och = 16;
const int c_conv1_weight_ich = 16;
const int c_conv1_weight_ih  = 3;
const int c_conv1_weight_iw  = 3;
const int c_conv1_weight_ops = 8;
const int c_conv1_weight_index = 9;
const int c_conv1_weight_iter  = 33;

typedef uint8_t t_pad_7;
typedef ap_int<32> t_conv_2_acc;
const int c_conv_2_ich    = 16;
const int c_conv_2_och    = 16;
const int c_conv_2_ih     = 16;
const int c_conv_2_iw     = 16;
const int c_conv_2_ow     = 8;
const int c_conv_2_oh     = 8;
const int c_conv_2_fw     = 3;
const int c_conv_2_fh     = 3;
const int c_conv_2_relu   = 1;
const int c_pad_7_a_split  = 0;
const int c_conv_2_stride = 2;
const int c_conv_2_pad    = 1;
const int c_conv_2_split  = 2;


typedef uint8_t t_pad_7;
const int c_relu_3_ich    = 16;
const int c_relu_3_ih     = 8;
const int c_relu_3_iw     = 8;


typedef uint8_t t_averagepool_9;
typedef int8_t t_pad_5_acc;
const int c_pad_5_ich    = 16;
const int c_pad_5_och    = 16;
const int c_pad_5_ih     = 8;
const int c_pad_5_iw     = 8;
const int c_pad_5_oh     = 8;
const int c_pad_5_ow     = 8;
const int c_pad_5_pad    = 0;


typedef uint8_t t_conv_10;
typedef int32_t t_averagepool_6_acc;
const int c_averagepool_6_ich    = 16;
const int c_averagepool_6_och    = 16;
const int c_averagepool_6_ih     = 8;
const int c_averagepool_6_iw     = 8;
const int c_averagepool_6_oh     = 1;
const int c_averagepool_6_ow     = 1;
const int c_averagepool_6_fh     = 8;
const int c_averagepool_6_fw     = 8;
const int c_averagepool_6_stride = 1;
const int c_averagepool_6_pad    = 0;
const int c_averagepool_6_pool   = 0;


const int c_conv_7_ops  = 1;

typedef uint8_t t_fc_weight_st;
typedef ap_uint<8*c_conv_7_ops> t_fc_weight;
const int c_fc_weight_och = 10;
const int c_fc_weight_ich = 16;
const int c_fc_weight_ih  = 1;
const int c_fc_weight_iw  = 1;
const int c_fc_weight_ops = 1;
const int c_fc_weight_index = 1;
const int c_fc_weight_iter  = 161;

typedef int32_t t_output;
typedef ap_int<32> t_conv_7_acc;
const int c_conv_7_ich    = 16;
const int c_conv_7_och    = 10;
const int c_conv_7_ih     = 1;
const int c_conv_7_iw     = 1;
const int c_conv_7_ow     = 1;
const int c_conv_7_oh     = 1;
const int c_conv_7_fw     = 1;
const int c_conv_7_fh     = 1;
const int c_conv_7_relu   = 0;
const int c_output_a_split  = 0;
const int c_conv_7_stride = 1;
const int c_conv_7_pad    = 0;
const int c_conv_7_split  = 2;

void Network(
	hls::stream<t_i_data> &i_data,
	hls::stream<t_o_data> &o_data
);
#endif