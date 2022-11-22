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


const int c_conv_0_ops  = 1;

typedef int8_t t_conv0_weight_st;
typedef ap_int<8*c_conv_0_ops> t_conv0_weight;
const int c_conv0_weight_och = 16;
const int c_conv0_weight_ich = 3;
const int c_conv0_weight_ih  = 3;
const int c_conv0_weight_iw  = 3;

typedef uint8_t t_conv_129;
typedef ap_int<32> t_conv_0_acc;
const int c_conv_0_ich    = 3;
const int c_conv_0_och    = 16;
const int c_conv_0_ih     = 32;
const int c_conv_0_iw     = 32;
const int c_conv_0_ow     = 32;
const int c_conv_0_oh     = 32;
const int c_conv_0_fw     = 3;
const int c_conv_0_fh     = 3;
const int c_conv_0_relu   = 1;
const int c_conv_129_split  = 0;
const int c_conv_0_stride = 1;
const int c_conv_0_pad    = 1;
const int c_conv_0_split  = 2;


typedef uint8_t t_conv_129;
const int c_relu_1_ich    = 16;
const int c_relu_1_ih     = 32;
const int c_relu_1_iw     = 32;

typedef uint8_t t_conv_129_skip;


const int c_conv_2_ops  = 2;

typedef int16_t t_layers_0_conv0_weight_st;
typedef ap_int<8*c_conv_2_ops> t_layers_0_conv0_weight;
const int c_layers_0_conv0_weight_och = 16;
const int c_layers_0_conv0_weight_ich = 16;
const int c_layers_0_conv0_weight_ih  = 3;
const int c_layers_0_conv0_weight_iw  = 3;

typedef uint8_t t_conv_131;
typedef ap_int<32> t_conv_2_acc;
const int c_conv_2_ich    = 16;
const int c_conv_2_och    = 16;
const int c_conv_2_ih     = 32;
const int c_conv_2_iw     = 32;
const int c_conv_2_ow     = 32;
const int c_conv_2_oh     = 32;
const int c_conv_2_fw     = 3;
const int c_conv_2_fh     = 3;
const int c_conv_2_relu   = 1;
const int c_conv_131_split  = 0;
const int c_conv_2_stride = 1;
const int c_conv_2_pad    = 1;
const int c_conv_2_split  = 2;


typedef uint8_t t_conv_131;
const int c_relu_3_ich    = 16;
const int c_relu_3_ih     = 32;
const int c_relu_3_iw     = 32;


const int c_conv_4_ops  = 2;

typedef int16_t t_layers_0_conv1_weight_st;
typedef ap_int<8*c_conv_4_ops> t_layers_0_conv1_weight;
const int c_layers_0_conv1_weight_och = 16;
const int c_layers_0_conv1_weight_ich = 16;
const int c_layers_0_conv1_weight_ih  = 3;
const int c_layers_0_conv1_weight_iw  = 3;

typedef uint8_t t_conv_134;
typedef ap_int<32> t_conv_4_acc;
const int c_conv_4_ich    = 16;
const int c_conv_4_och    = 16;
const int c_conv_4_ih     = 32;
const int c_conv_4_iw     = 32;
const int c_conv_4_ow     = 32;
const int c_conv_4_oh     = 32;
const int c_conv_4_fw     = 3;
const int c_conv_4_fh     = 3;
const int c_conv_4_relu   = 1;
const int c_conv_134_split  = 0;
const int c_conv_4_stride = 1;
const int c_conv_4_pad    = 1;
const int c_conv_4_split  = 2;


typedef uint8_t t_relu_133;
const int c_add_5_ich    = 16;
const int c_add_5_ih     = 32;
const int c_add_5_iw     = 32;


typedef uint8_t t_conv_134;
const int c_relu_6_ich    = 16;
const int c_relu_6_ih     = 32;
const int c_relu_6_iw     = 32;

typedef uint8_t t_conv_134_skip;


const int c_conv_7_ops  = 2;

typedef int16_t t_layers_1_conv0_weight_st;
typedef ap_int<8*c_conv_7_ops> t_layers_1_conv0_weight;
const int c_layers_1_conv0_weight_och = 16;
const int c_layers_1_conv0_weight_ich = 16;
const int c_layers_1_conv0_weight_ih  = 3;
const int c_layers_1_conv0_weight_iw  = 3;

typedef uint8_t t_conv_136;
typedef ap_int<32> t_conv_7_acc;
const int c_conv_7_ich    = 16;
const int c_conv_7_och    = 16;
const int c_conv_7_ih     = 32;
const int c_conv_7_iw     = 32;
const int c_conv_7_ow     = 32;
const int c_conv_7_oh     = 32;
const int c_conv_7_fw     = 3;
const int c_conv_7_fh     = 3;
const int c_conv_7_relu   = 1;
const int c_conv_136_split  = 0;
const int c_conv_7_stride = 1;
const int c_conv_7_pad    = 1;
const int c_conv_7_split  = 2;


typedef uint8_t t_conv_136;
const int c_relu_8_ich    = 16;
const int c_relu_8_ih     = 32;
const int c_relu_8_iw     = 32;


const int c_conv_9_ops  = 2;

typedef int16_t t_layers_1_conv1_weight_st;
typedef ap_int<8*c_conv_9_ops> t_layers_1_conv1_weight;
const int c_layers_1_conv1_weight_och = 16;
const int c_layers_1_conv1_weight_ich = 16;
const int c_layers_1_conv1_weight_ih  = 3;
const int c_layers_1_conv1_weight_iw  = 3;

typedef uint8_t t_conv_139;
typedef ap_int<32> t_conv_9_acc;
const int c_conv_9_ich    = 16;
const int c_conv_9_och    = 16;
const int c_conv_9_ih     = 32;
const int c_conv_9_iw     = 32;
const int c_conv_9_ow     = 32;
const int c_conv_9_oh     = 32;
const int c_conv_9_fw     = 3;
const int c_conv_9_fh     = 3;
const int c_conv_9_relu   = 1;
const int c_conv_139_split  = 0;
const int c_conv_9_stride = 1;
const int c_conv_9_pad    = 1;
const int c_conv_9_split  = 2;


typedef uint8_t t_relu_138;
const int c_add_10_ich    = 16;
const int c_add_10_ih     = 32;
const int c_add_10_iw     = 32;


typedef uint8_t t_conv_139;
const int c_relu_11_ich    = 16;
const int c_relu_11_ih     = 32;
const int c_relu_11_iw     = 32;

typedef uint8_t t_conv_139_skip;


const int c_conv_12_ops  = 2;

typedef int16_t t_layers_2_conv0_weight_st;
typedef ap_int<8*c_conv_12_ops> t_layers_2_conv0_weight;
const int c_layers_2_conv0_weight_och = 16;
const int c_layers_2_conv0_weight_ich = 16;
const int c_layers_2_conv0_weight_ih  = 3;
const int c_layers_2_conv0_weight_iw  = 3;

typedef uint8_t t_conv_141;
typedef ap_int<32> t_conv_12_acc;
const int c_conv_12_ich    = 16;
const int c_conv_12_och    = 16;
const int c_conv_12_ih     = 32;
const int c_conv_12_iw     = 32;
const int c_conv_12_ow     = 32;
const int c_conv_12_oh     = 32;
const int c_conv_12_fw     = 3;
const int c_conv_12_fh     = 3;
const int c_conv_12_relu   = 1;
const int c_conv_141_split  = 0;
const int c_conv_12_stride = 1;
const int c_conv_12_pad    = 1;
const int c_conv_12_split  = 2;


typedef uint8_t t_conv_141;
const int c_relu_13_ich    = 16;
const int c_relu_13_ih     = 32;
const int c_relu_13_iw     = 32;


const int c_conv_14_ops  = 2;

typedef int16_t t_layers_2_conv1_weight_st;
typedef ap_int<8*c_conv_14_ops> t_layers_2_conv1_weight;
const int c_layers_2_conv1_weight_och = 16;
const int c_layers_2_conv1_weight_ich = 16;
const int c_layers_2_conv1_weight_ih  = 3;
const int c_layers_2_conv1_weight_iw  = 3;

typedef uint8_t t_conv_144;
typedef ap_int<32> t_conv_14_acc;
const int c_conv_14_ich    = 16;
const int c_conv_14_och    = 16;
const int c_conv_14_ih     = 32;
const int c_conv_14_iw     = 32;
const int c_conv_14_ow     = 32;
const int c_conv_14_oh     = 32;
const int c_conv_14_fw     = 3;
const int c_conv_14_fh     = 3;
const int c_conv_14_relu   = 1;
const int c_conv_144_split  = 2;
const int c_conv_14_stride = 1;
const int c_conv_14_pad    = 1;
const int c_conv_14_split  = 2;


typedef uint8_t t_relu_143;
const int c_add_15_ich    = 16;
const int c_add_15_ih     = 32;
const int c_add_15_iw     = 32;


typedef uint8_t t_conv_144;
const int c_relu_16_ich    = 16;
const int c_relu_16_ih     = 32;
const int c_relu_16_iw     = 32;


const int c_conv_17_ops  = 1;

typedef int8_t t_layers_3_skip_conv_weight_st;
typedef ap_int<8*c_conv_17_ops> t_layers_3_skip_conv_weight;
const int c_layers_3_skip_conv_weight_och = 32;
const int c_layers_3_skip_conv_weight_ich = 16;
const int c_layers_3_skip_conv_weight_ih  = 1;
const int c_layers_3_skip_conv_weight_iw  = 1;

typedef uint8_t t_add_146;
typedef ap_int<32> t_conv_17_acc;
const int c_conv_17_ich    = 16;
const int c_conv_17_och    = 32;
const int c_conv_17_ih     = 32;
const int c_conv_17_iw     = 32;
const int c_conv_17_ow     = 16;
const int c_conv_17_oh     = 16;
const int c_conv_17_fw     = 1;
const int c_conv_17_fh     = 1;
const int c_conv_17_relu   = 1;
const int c_add_146_split  = 0;
const int c_conv_17_stride = 2;
const int c_conv_17_pad    = 0;
const int c_conv_17_split  = 2;


typedef uint8_t t_add_146;
const int c_relu_18_ich    = 32;
const int c_relu_18_ih     = 16;
const int c_relu_18_iw     = 16;


const int c_conv_19_ops  = 1;

typedef int8_t t_layers_3_conv0_weight_st;
typedef ap_int<8*c_conv_19_ops> t_layers_3_conv0_weight;
const int c_layers_3_conv0_weight_och = 32;
const int c_layers_3_conv0_weight_ich = 16;
const int c_layers_3_conv0_weight_ih  = 3;
const int c_layers_3_conv0_weight_iw  = 3;

typedef uint8_t t_conv_148;
typedef ap_int<32> t_conv_19_acc;
const int c_conv_19_ich    = 16;
const int c_conv_19_och    = 32;
const int c_conv_19_ih     = 32;
const int c_conv_19_iw     = 32;
const int c_conv_19_ow     = 16;
const int c_conv_19_oh     = 16;
const int c_conv_19_fw     = 3;
const int c_conv_19_fh     = 3;
const int c_conv_19_relu   = 1;
const int c_conv_148_split  = 0;
const int c_conv_19_stride = 2;
const int c_conv_19_pad    = 1;
const int c_conv_19_split  = 2;


typedef uint8_t t_conv_148;
const int c_relu_20_ich    = 32;
const int c_relu_20_ih     = 16;
const int c_relu_20_iw     = 16;


const int c_conv_21_ops  = 1;

typedef int8_t t_layers_3_conv1_weight_st;
typedef ap_int<8*c_conv_21_ops> t_layers_3_conv1_weight;
const int c_layers_3_conv1_weight_och = 32;
const int c_layers_3_conv1_weight_ich = 32;
const int c_layers_3_conv1_weight_ih  = 3;
const int c_layers_3_conv1_weight_iw  = 3;

typedef uint8_t t_conv_151;
typedef ap_int<32> t_conv_21_acc;
const int c_conv_21_ich    = 32;
const int c_conv_21_och    = 32;
const int c_conv_21_ih     = 16;
const int c_conv_21_iw     = 16;
const int c_conv_21_ow     = 16;
const int c_conv_21_oh     = 16;
const int c_conv_21_fw     = 3;
const int c_conv_21_fh     = 3;
const int c_conv_21_relu   = 1;
const int c_conv_151_split  = 0;
const int c_conv_21_stride = 1;
const int c_conv_21_pad    = 1;
const int c_conv_21_split  = 2;


typedef uint8_t t_relu_150;
const int c_add_22_ich    = 32;
const int c_add_22_ih     = 16;
const int c_add_22_iw     = 16;


typedef uint8_t t_conv_151;
const int c_relu_23_ich    = 32;
const int c_relu_23_ih     = 16;
const int c_relu_23_iw     = 16;

typedef uint8_t t_conv_151_skip;


const int c_conv_24_ops  = 1;

typedef int8_t t_layers_4_conv0_weight_st;
typedef ap_int<8*c_conv_24_ops> t_layers_4_conv0_weight;
const int c_layers_4_conv0_weight_och = 32;
const int c_layers_4_conv0_weight_ich = 32;
const int c_layers_4_conv0_weight_ih  = 3;
const int c_layers_4_conv0_weight_iw  = 3;

typedef uint8_t t_conv_153;
typedef ap_int<32> t_conv_24_acc;
const int c_conv_24_ich    = 32;
const int c_conv_24_och    = 32;
const int c_conv_24_ih     = 16;
const int c_conv_24_iw     = 16;
const int c_conv_24_ow     = 16;
const int c_conv_24_oh     = 16;
const int c_conv_24_fw     = 3;
const int c_conv_24_fh     = 3;
const int c_conv_24_relu   = 1;
const int c_conv_153_split  = 0;
const int c_conv_24_stride = 1;
const int c_conv_24_pad    = 1;
const int c_conv_24_split  = 2;


typedef uint8_t t_conv_153;
const int c_relu_25_ich    = 32;
const int c_relu_25_ih     = 16;
const int c_relu_25_iw     = 16;


const int c_conv_26_ops  = 1;

typedef int8_t t_layers_4_conv1_weight_st;
typedef ap_int<8*c_conv_26_ops> t_layers_4_conv1_weight;
const int c_layers_4_conv1_weight_och = 32;
const int c_layers_4_conv1_weight_ich = 32;
const int c_layers_4_conv1_weight_ih  = 3;
const int c_layers_4_conv1_weight_iw  = 3;

typedef uint8_t t_conv_156;
typedef ap_int<32> t_conv_26_acc;
const int c_conv_26_ich    = 32;
const int c_conv_26_och    = 32;
const int c_conv_26_ih     = 16;
const int c_conv_26_iw     = 16;
const int c_conv_26_ow     = 16;
const int c_conv_26_oh     = 16;
const int c_conv_26_fw     = 3;
const int c_conv_26_fh     = 3;
const int c_conv_26_relu   = 1;
const int c_conv_156_split  = 0;
const int c_conv_26_stride = 1;
const int c_conv_26_pad    = 1;
const int c_conv_26_split  = 2;


typedef uint8_t t_relu_155;
const int c_add_27_ich    = 32;
const int c_add_27_ih     = 16;
const int c_add_27_iw     = 16;


typedef uint8_t t_conv_156;
const int c_relu_28_ich    = 32;
const int c_relu_28_ih     = 16;
const int c_relu_28_iw     = 16;

typedef uint8_t t_conv_156_skip;


const int c_conv_29_ops  = 1;

typedef int8_t t_layers_5_conv0_weight_st;
typedef ap_int<8*c_conv_29_ops> t_layers_5_conv0_weight;
const int c_layers_5_conv0_weight_och = 32;
const int c_layers_5_conv0_weight_ich = 32;
const int c_layers_5_conv0_weight_ih  = 3;
const int c_layers_5_conv0_weight_iw  = 3;

typedef uint8_t t_conv_158;
typedef ap_int<32> t_conv_29_acc;
const int c_conv_29_ich    = 32;
const int c_conv_29_och    = 32;
const int c_conv_29_ih     = 16;
const int c_conv_29_iw     = 16;
const int c_conv_29_ow     = 16;
const int c_conv_29_oh     = 16;
const int c_conv_29_fw     = 3;
const int c_conv_29_fh     = 3;
const int c_conv_29_relu   = 1;
const int c_conv_158_split  = 0;
const int c_conv_29_stride = 1;
const int c_conv_29_pad    = 1;
const int c_conv_29_split  = 2;


typedef uint8_t t_conv_158;
const int c_relu_30_ich    = 32;
const int c_relu_30_ih     = 16;
const int c_relu_30_iw     = 16;


const int c_conv_31_ops  = 1;

typedef int8_t t_layers_5_conv1_weight_st;
typedef ap_int<8*c_conv_31_ops> t_layers_5_conv1_weight;
const int c_layers_5_conv1_weight_och = 32;
const int c_layers_5_conv1_weight_ich = 32;
const int c_layers_5_conv1_weight_ih  = 3;
const int c_layers_5_conv1_weight_iw  = 3;

typedef uint8_t t_conv_161;
typedef ap_int<32> t_conv_31_acc;
const int c_conv_31_ich    = 32;
const int c_conv_31_och    = 32;
const int c_conv_31_ih     = 16;
const int c_conv_31_iw     = 16;
const int c_conv_31_ow     = 16;
const int c_conv_31_oh     = 16;
const int c_conv_31_fw     = 3;
const int c_conv_31_fh     = 3;
const int c_conv_31_relu   = 1;
const int c_conv_161_split  = 2;
const int c_conv_31_stride = 1;
const int c_conv_31_pad    = 1;
const int c_conv_31_split  = 2;


typedef uint8_t t_relu_160;
const int c_add_32_ich    = 32;
const int c_add_32_ih     = 16;
const int c_add_32_iw     = 16;


typedef uint8_t t_conv_161;
const int c_relu_33_ich    = 32;
const int c_relu_33_ih     = 16;
const int c_relu_33_iw     = 16;


const int c_conv_34_ops  = 1;

typedef int8_t t_layers_6_skip_conv_weight_st;
typedef ap_int<8*c_conv_34_ops> t_layers_6_skip_conv_weight;
const int c_layers_6_skip_conv_weight_och = 64;
const int c_layers_6_skip_conv_weight_ich = 32;
const int c_layers_6_skip_conv_weight_ih  = 1;
const int c_layers_6_skip_conv_weight_iw  = 1;

typedef uint8_t t_add_163;
typedef ap_int<32> t_conv_34_acc;
const int c_conv_34_ich    = 32;
const int c_conv_34_och    = 64;
const int c_conv_34_ih     = 16;
const int c_conv_34_iw     = 16;
const int c_conv_34_ow     = 8;
const int c_conv_34_oh     = 8;
const int c_conv_34_fw     = 1;
const int c_conv_34_fh     = 1;
const int c_conv_34_relu   = 1;
const int c_add_163_split  = 0;
const int c_conv_34_stride = 2;
const int c_conv_34_pad    = 0;
const int c_conv_34_split  = 2;


typedef uint8_t t_add_163;
const int c_relu_35_ich    = 64;
const int c_relu_35_ih     = 8;
const int c_relu_35_iw     = 8;


const int c_conv_36_ops  = 1;

typedef int8_t t_layers_6_conv0_weight_st;
typedef ap_int<8*c_conv_36_ops> t_layers_6_conv0_weight;
const int c_layers_6_conv0_weight_och = 64;
const int c_layers_6_conv0_weight_ich = 32;
const int c_layers_6_conv0_weight_ih  = 3;
const int c_layers_6_conv0_weight_iw  = 3;

typedef uint8_t t_conv_165;
typedef ap_int<32> t_conv_36_acc;
const int c_conv_36_ich    = 32;
const int c_conv_36_och    = 64;
const int c_conv_36_ih     = 16;
const int c_conv_36_iw     = 16;
const int c_conv_36_ow     = 8;
const int c_conv_36_oh     = 8;
const int c_conv_36_fw     = 3;
const int c_conv_36_fh     = 3;
const int c_conv_36_relu   = 1;
const int c_conv_165_split  = 0;
const int c_conv_36_stride = 2;
const int c_conv_36_pad    = 1;
const int c_conv_36_split  = 2;


typedef uint8_t t_conv_165;
const int c_relu_37_ich    = 64;
const int c_relu_37_ih     = 8;
const int c_relu_37_iw     = 8;


const int c_conv_38_ops  = 1;

typedef int8_t t_layers_6_conv1_weight_st;
typedef ap_int<8*c_conv_38_ops> t_layers_6_conv1_weight;
const int c_layers_6_conv1_weight_och = 64;
const int c_layers_6_conv1_weight_ich = 64;
const int c_layers_6_conv1_weight_ih  = 3;
const int c_layers_6_conv1_weight_iw  = 3;

typedef uint8_t t_conv_168;
typedef ap_int<32> t_conv_38_acc;
const int c_conv_38_ich    = 64;
const int c_conv_38_och    = 64;
const int c_conv_38_ih     = 8;
const int c_conv_38_iw     = 8;
const int c_conv_38_ow     = 8;
const int c_conv_38_oh     = 8;
const int c_conv_38_fw     = 3;
const int c_conv_38_fh     = 3;
const int c_conv_38_relu   = 1;
const int c_conv_168_split  = 0;
const int c_conv_38_stride = 1;
const int c_conv_38_pad    = 1;
const int c_conv_38_split  = 2;


typedef uint8_t t_relu_167;
const int c_add_39_ich    = 64;
const int c_add_39_ih     = 8;
const int c_add_39_iw     = 8;


typedef uint8_t t_conv_168;
const int c_relu_40_ich    = 64;
const int c_relu_40_ih     = 8;
const int c_relu_40_iw     = 8;

typedef uint8_t t_conv_168_skip;


const int c_conv_41_ops  = 1;

typedef int8_t t_layers_7_conv0_weight_st;
typedef ap_int<8*c_conv_41_ops> t_layers_7_conv0_weight;
const int c_layers_7_conv0_weight_och = 64;
const int c_layers_7_conv0_weight_ich = 64;
const int c_layers_7_conv0_weight_ih  = 3;
const int c_layers_7_conv0_weight_iw  = 3;

typedef uint8_t t_conv_170;
typedef ap_int<32> t_conv_41_acc;
const int c_conv_41_ich    = 64;
const int c_conv_41_och    = 64;
const int c_conv_41_ih     = 8;
const int c_conv_41_iw     = 8;
const int c_conv_41_ow     = 8;
const int c_conv_41_oh     = 8;
const int c_conv_41_fw     = 3;
const int c_conv_41_fh     = 3;
const int c_conv_41_relu   = 1;
const int c_conv_170_split  = 0;
const int c_conv_41_stride = 1;
const int c_conv_41_pad    = 1;
const int c_conv_41_split  = 2;


typedef uint8_t t_conv_170;
const int c_relu_42_ich    = 64;
const int c_relu_42_ih     = 8;
const int c_relu_42_iw     = 8;


const int c_conv_43_ops  = 1;

typedef int8_t t_layers_7_conv1_weight_st;
typedef ap_int<8*c_conv_43_ops> t_layers_7_conv1_weight;
const int c_layers_7_conv1_weight_och = 64;
const int c_layers_7_conv1_weight_ich = 64;
const int c_layers_7_conv1_weight_ih  = 3;
const int c_layers_7_conv1_weight_iw  = 3;

typedef uint8_t t_conv_173;
typedef ap_int<32> t_conv_43_acc;
const int c_conv_43_ich    = 64;
const int c_conv_43_och    = 64;
const int c_conv_43_ih     = 8;
const int c_conv_43_iw     = 8;
const int c_conv_43_ow     = 8;
const int c_conv_43_oh     = 8;
const int c_conv_43_fw     = 3;
const int c_conv_43_fh     = 3;
const int c_conv_43_relu   = 1;
const int c_conv_173_split  = 0;
const int c_conv_43_stride = 1;
const int c_conv_43_pad    = 1;
const int c_conv_43_split  = 2;


typedef uint8_t t_relu_172;
const int c_add_44_ich    = 64;
const int c_add_44_ih     = 8;
const int c_add_44_iw     = 8;


typedef uint8_t t_conv_173;
const int c_relu_45_ich    = 64;
const int c_relu_45_ih     = 8;
const int c_relu_45_iw     = 8;

typedef uint8_t t_conv_173_skip;


const int c_conv_46_ops  = 1;

typedef int8_t t_layers_8_conv0_weight_st;
typedef ap_int<8*c_conv_46_ops> t_layers_8_conv0_weight;
const int c_layers_8_conv0_weight_och = 64;
const int c_layers_8_conv0_weight_ich = 64;
const int c_layers_8_conv0_weight_ih  = 3;
const int c_layers_8_conv0_weight_iw  = 3;

typedef uint8_t t_conv_175;
typedef ap_int<32> t_conv_46_acc;
const int c_conv_46_ich    = 64;
const int c_conv_46_och    = 64;
const int c_conv_46_ih     = 8;
const int c_conv_46_iw     = 8;
const int c_conv_46_ow     = 8;
const int c_conv_46_oh     = 8;
const int c_conv_46_fw     = 3;
const int c_conv_46_fh     = 3;
const int c_conv_46_relu   = 1;
const int c_conv_175_split  = 0;
const int c_conv_46_stride = 1;
const int c_conv_46_pad    = 1;
const int c_conv_46_split  = 2;


typedef uint8_t t_conv_175;
const int c_relu_47_ich    = 64;
const int c_relu_47_ih     = 8;
const int c_relu_47_iw     = 8;


const int c_conv_48_ops  = 1;

typedef int8_t t_layers_8_conv1_weight_st;
typedef ap_int<8*c_conv_48_ops> t_layers_8_conv1_weight;
const int c_layers_8_conv1_weight_och = 64;
const int c_layers_8_conv1_weight_ich = 64;
const int c_layers_8_conv1_weight_ih  = 3;
const int c_layers_8_conv1_weight_iw  = 3;

typedef uint8_t t_pad_178;
typedef ap_int<32> t_conv_48_acc;
const int c_conv_48_ich    = 64;
const int c_conv_48_och    = 64;
const int c_conv_48_ih     = 8;
const int c_conv_48_iw     = 8;
const int c_conv_48_ow     = 8;
const int c_conv_48_oh     = 8;
const int c_conv_48_fw     = 3;
const int c_conv_48_fh     = 3;
const int c_conv_48_relu   = 1;
const int c_pad_178_split  = 0;
const int c_conv_48_stride = 1;
const int c_conv_48_pad    = 1;
const int c_conv_48_split  = 2;


typedef uint8_t t_relu_177;
const int c_add_49_ich    = 64;
const int c_add_49_ih     = 8;
const int c_add_49_iw     = 8;


typedef uint8_t t_pad_178;
const int c_relu_50_ich    = 64;
const int c_relu_50_ih     = 8;
const int c_relu_50_iw     = 8;


typedef uint8_t t_averagepool_180;
typedef int8_t t_pad_52_acc;
const int c_pad_52_ich    = 64;
const int c_pad_52_och    = 64;
const int c_pad_52_ih     = 8;
const int c_pad_52_iw     = 8;
const int c_pad_52_oh     = 8;
const int c_pad_52_ow     = 8;
const int c_pad_52_pad    = 0;


typedef uint8_t t_input_1;
typedef int32_t t_averagepool_53_acc;
const int c_averagepool_53_ich    = 64;
const int c_averagepool_53_och    = 64;
const int c_averagepool_53_ih     = 8;
const int c_averagepool_53_iw     = 8;
const int c_averagepool_53_oh     = 1;
const int c_averagepool_53_ow     = 1;
const int c_averagepool_53_fh     = 8;
const int c_averagepool_53_fw     = 8;
const int c_averagepool_53_stride = 1;
const int c_averagepool_53_pad    = 0;
const int c_averagepool_53_pool   = 0;


const int c_conv_54_ops  = 1;

typedef int8_t t_fc_weight_st;
typedef ap_int<8*c_conv_54_ops> t_fc_weight;
const int c_fc_weight_och = 10;
const int c_fc_weight_ich = 64;
const int c_fc_weight_ih  = 1;
const int c_fc_weight_iw  = 1;

typedef int32_t t_output;
typedef ap_int<32> t_conv_54_acc;
const int c_conv_54_ich    = 64;
const int c_conv_54_och    = 10;
const int c_conv_54_ih     = 1;
const int c_conv_54_iw     = 1;
const int c_conv_54_ow     = 1;
const int c_conv_54_oh     = 1;
const int c_conv_54_fw     = 1;
const int c_conv_54_fh     = 1;
const int c_conv_54_relu   = 0;
const int c_output_split  = 0;
const int c_conv_54_stride = 1;
const int c_conv_54_pad    = 0;
const int c_conv_54_split  = 2;

void Network(
	hls::stream<t_i_data> &i_data,
	hls::stream<t_o_data> &o_data
);
#endif