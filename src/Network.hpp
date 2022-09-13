#ifndef __NETWORK__
#define __NETWORK__
#include "hls_stream.h"
#include "ap_int.h"
#include <stdint.h>
typedef int8_t t_i_data;
typedef int8_t t_weight;
typedef int8_t t_o_data;
typedef uint8_t t_input;

const int c_input_ich    = 3;
const int c_input_ih     = 32;
const int c_input_iw     = 32;


const int c_output_och = 10;
const int c_output_oh  = 1;
const int c_output_ow  = 1;


typedef ap_uint<8> t_conv_208_st;
typedef ap_uint<8> t_conv_208;
const int c_conv_208_och = 16;
const int c_conv_208_ich = 3;
const int c_conv_208_ih  = 3;
const int c_conv_208_iw  = 3;

typedef ap_uint<8> t_conv_135;
typedef ap_uint<32> t_conv_0_acc;
const int c_conv_0_ich    = 3;
const int c_conv_0_och    = 16;
const int c_conv_0_ih     = 32;
const int c_conv_0_iw     = 32;
const int c_conv_0_ow     = 32;
const int c_conv_0_oh     = 32;
const int c_conv_0_fw     = 3;
const int c_conv_0_fh     = 3;
const int c_conv_0_relu   = 1;
const int c_conv_135_split  = 0;
const int c_conv_0_stride = 1;
const int c_conv_0_pad    = 1;


typedef ap_uint<8> t_conv_135;
const int c_relu_1_ich    = 16;
const int c_relu_1_ih     = 32;
const int c_relu_1_iw     = 32;

typedef ap_uint<8> t_conv_135_skip;


typedef ap_uint<8> t_conv_211_st;
typedef ap_uint<8> t_conv_211;
const int c_conv_211_och = 16;
const int c_conv_211_ich = 16;
const int c_conv_211_ih  = 3;
const int c_conv_211_iw  = 3;

typedef ap_uint<8> t_conv_138;
typedef ap_uint<32> t_conv_2_acc;
const int c_conv_2_ich    = 16;
const int c_conv_2_och    = 16;
const int c_conv_2_ih     = 32;
const int c_conv_2_iw     = 32;
const int c_conv_2_ow     = 32;
const int c_conv_2_oh     = 32;
const int c_conv_2_fw     = 3;
const int c_conv_2_fh     = 3;
const int c_conv_2_relu   = 1;
const int c_conv_138_split  = 0;
const int c_conv_2_stride = 1;
const int c_conv_2_pad    = 1;


typedef ap_uint<8> t_conv_138;
const int c_relu_3_ich    = 16;
const int c_relu_3_ih     = 32;
const int c_relu_3_iw     = 32;


typedef ap_uint<8> t_conv_214_st;
typedef ap_uint<8> t_conv_214;
const int c_conv_214_och = 16;
const int c_conv_214_ich = 16;
const int c_conv_214_ih  = 3;
const int c_conv_214_iw  = 3;

typedef ap_uint<8> t_input_20;
typedef ap_uint<32> t_conv_4_acc;
const int c_conv_4_ich    = 16;
const int c_conv_4_och    = 16;
const int c_conv_4_ih     = 32;
const int c_conv_4_iw     = 32;
const int c_conv_4_ow     = 32;
const int c_conv_4_oh     = 32;
const int c_conv_4_fw     = 3;
const int c_conv_4_fh     = 3;
const int c_conv_4_relu   = 1;
const int c_input_20_split  = 0;
const int c_conv_4_stride = 1;
const int c_conv_4_pad    = 1;


typedef ap_uint<8> t_relu_141;
const int c_add_5_ich    = 16;
const int c_add_5_ih     = 32;
const int c_add_5_iw     = 32;


typedef ap_uint<8> t_input_20;
const int c_relu_6_ich    = 16;
const int c_relu_6_ih     = 32;
const int c_relu_6_iw     = 32;

typedef ap_uint<8> t_input_20_skip;


typedef ap_uint<8> t_conv_217_st;
typedef ap_uint<8> t_conv_217;
const int c_conv_217_och = 16;
const int c_conv_217_ich = 16;
const int c_conv_217_ih  = 3;
const int c_conv_217_iw  = 3;

typedef ap_uint<8> t_conv_145;
typedef ap_uint<32> t_conv_7_acc;
const int c_conv_7_ich    = 16;
const int c_conv_7_och    = 16;
const int c_conv_7_ih     = 32;
const int c_conv_7_iw     = 32;
const int c_conv_7_ow     = 32;
const int c_conv_7_oh     = 32;
const int c_conv_7_fw     = 3;
const int c_conv_7_fh     = 3;
const int c_conv_7_relu   = 1;
const int c_conv_145_split  = 0;
const int c_conv_7_stride = 1;
const int c_conv_7_pad    = 1;


typedef ap_uint<8> t_conv_145;
const int c_relu_8_ich    = 16;
const int c_relu_8_ih     = 32;
const int c_relu_8_iw     = 32;


typedef ap_uint<8> t_conv_220_st;
typedef ap_uint<8> t_conv_220;
const int c_conv_220_och = 16;
const int c_conv_220_ich = 16;
const int c_conv_220_ih  = 3;
const int c_conv_220_iw  = 3;

typedef ap_uint<8> t_input_36;
typedef ap_uint<32> t_conv_9_acc;
const int c_conv_9_ich    = 16;
const int c_conv_9_och    = 16;
const int c_conv_9_ih     = 32;
const int c_conv_9_iw     = 32;
const int c_conv_9_ow     = 32;
const int c_conv_9_oh     = 32;
const int c_conv_9_fw     = 3;
const int c_conv_9_fh     = 3;
const int c_conv_9_relu   = 1;
const int c_input_36_split  = 0;
const int c_conv_9_stride = 1;
const int c_conv_9_pad    = 1;


typedef ap_uint<8> t_relu_148;
const int c_add_10_ich    = 16;
const int c_add_10_ih     = 32;
const int c_add_10_iw     = 32;


typedef ap_uint<8> t_input_36;
const int c_relu_11_ich    = 16;
const int c_relu_11_ih     = 32;
const int c_relu_11_iw     = 32;

typedef ap_uint<8> t_input_36_skip;


typedef ap_uint<8> t_conv_223_st;
typedef ap_uint<8> t_conv_223;
const int c_conv_223_och = 16;
const int c_conv_223_ich = 16;
const int c_conv_223_ih  = 3;
const int c_conv_223_iw  = 3;

typedef ap_uint<8> t_conv_152;
typedef ap_uint<32> t_conv_12_acc;
const int c_conv_12_ich    = 16;
const int c_conv_12_och    = 16;
const int c_conv_12_ih     = 32;
const int c_conv_12_iw     = 32;
const int c_conv_12_ow     = 32;
const int c_conv_12_oh     = 32;
const int c_conv_12_fw     = 3;
const int c_conv_12_fh     = 3;
const int c_conv_12_relu   = 1;
const int c_conv_152_split  = 0;
const int c_conv_12_stride = 1;
const int c_conv_12_pad    = 1;


typedef ap_uint<8> t_conv_152;
const int c_relu_13_ich    = 16;
const int c_relu_13_ih     = 32;
const int c_relu_13_iw     = 32;


typedef ap_uint<8> t_conv_226_st;
typedef ap_uint<8> t_conv_226;
const int c_conv_226_och = 16;
const int c_conv_226_ich = 16;
const int c_conv_226_ih  = 3;
const int c_conv_226_iw  = 3;

typedef ap_uint<8> t_input_52;
typedef ap_uint<32> t_conv_14_acc;
const int c_conv_14_ich    = 16;
const int c_conv_14_och    = 16;
const int c_conv_14_ih     = 32;
const int c_conv_14_iw     = 32;
const int c_conv_14_ow     = 32;
const int c_conv_14_oh     = 32;
const int c_conv_14_fw     = 3;
const int c_conv_14_fh     = 3;
const int c_conv_14_relu   = 1;
const int c_input_52_split  = 2;
const int c_conv_14_stride = 1;
const int c_conv_14_pad    = 1;


typedef ap_uint<8> t_relu_155;
const int c_add_15_ich    = 16;
const int c_add_15_ih     = 32;
const int c_add_15_iw     = 32;


typedef ap_uint<8> t_input_52;
const int c_relu_16_ich    = 16;
const int c_relu_16_ih     = 32;
const int c_relu_16_iw     = 32;


typedef ap_uint<8> t_conv_229_st;
typedef ap_uint<8> t_conv_229;
const int c_conv_229_och = 32;
const int c_conv_229_ich = 16;
const int c_conv_229_ih  = 3;
const int c_conv_229_iw  = 3;

typedef ap_uint<8> t_conv_159;
typedef ap_uint<32> t_conv_17_acc;
const int c_conv_17_ich    = 16;
const int c_conv_17_och    = 32;
const int c_conv_17_ih     = 32;
const int c_conv_17_iw     = 32;
const int c_conv_17_ow     = 16;
const int c_conv_17_oh     = 16;
const int c_conv_17_fw     = 3;
const int c_conv_17_fh     = 3;
const int c_conv_17_relu   = 1;
const int c_conv_159_split  = 0;
const int c_conv_17_stride = 2;
const int c_conv_17_pad    = 1;


typedef ap_uint<8> t_conv_159;
const int c_relu_18_ich    = 32;
const int c_relu_18_ih     = 16;
const int c_relu_18_iw     = 16;


typedef ap_uint<8> t_conv_232_st;
typedef ap_uint<8> t_conv_232;
const int c_conv_232_och = 32;
const int c_conv_232_ich = 32;
const int c_conv_232_ih  = 3;
const int c_conv_232_iw  = 3;

typedef ap_uint<8> t_input_72;
typedef ap_uint<32> t_conv_19_acc;
const int c_conv_19_ich    = 32;
const int c_conv_19_och    = 32;
const int c_conv_19_ih     = 16;
const int c_conv_19_iw     = 16;
const int c_conv_19_ow     = 16;
const int c_conv_19_oh     = 16;
const int c_conv_19_fw     = 3;
const int c_conv_19_fh     = 3;
const int c_conv_19_relu   = 1;
const int c_input_72_split  = 0;
const int c_conv_19_stride = 1;
const int c_conv_19_pad    = 1;


typedef ap_uint<8> t_conv_235_st;
typedef ap_uint<8> t_conv_235;
const int c_conv_235_och = 32;
const int c_conv_235_ich = 16;
const int c_conv_235_ih  = 1;
const int c_conv_235_iw  = 1;

typedef ap_uint<8> t_add_234;
typedef ap_uint<32> t_conv_20_acc;
const int c_conv_20_ich    = 16;
const int c_conv_20_och    = 32;
const int c_conv_20_ih     = 32;
const int c_conv_20_iw     = 32;
const int c_conv_20_ow     = 16;
const int c_conv_20_oh     = 16;
const int c_conv_20_fw     = 1;
const int c_conv_20_fh     = 1;
const int c_conv_20_relu   = 0;
const int c_add_234_split  = 0;
const int c_conv_20_stride = 2;
const int c_conv_20_pad    = 0;


typedef ap_uint<8> t_relu_164;
const int c_add_21_ich    = 32;
const int c_add_21_ih     = 16;
const int c_add_21_iw     = 16;


typedef ap_uint<8> t_input_72;
const int c_relu_22_ich    = 32;
const int c_relu_22_ih     = 16;
const int c_relu_22_iw     = 16;

typedef ap_uint<8> t_input_72_skip;


typedef ap_uint<8> t_conv_238_st;
typedef ap_uint<8> t_conv_238;
const int c_conv_238_och = 32;
const int c_conv_238_ich = 32;
const int c_conv_238_ih  = 3;
const int c_conv_238_iw  = 3;

typedef ap_uint<8> t_conv_168;
typedef ap_uint<32> t_conv_23_acc;
const int c_conv_23_ich    = 32;
const int c_conv_23_och    = 32;
const int c_conv_23_ih     = 16;
const int c_conv_23_iw     = 16;
const int c_conv_23_ow     = 16;
const int c_conv_23_oh     = 16;
const int c_conv_23_fw     = 3;
const int c_conv_23_fh     = 3;
const int c_conv_23_relu   = 1;
const int c_conv_168_split  = 0;
const int c_conv_23_stride = 1;
const int c_conv_23_pad    = 1;


typedef ap_uint<8> t_conv_168;
const int c_relu_24_ich    = 32;
const int c_relu_24_ih     = 16;
const int c_relu_24_iw     = 16;


typedef ap_uint<8> t_conv_241_st;
typedef ap_uint<8> t_conv_241;
const int c_conv_241_och = 32;
const int c_conv_241_ich = 32;
const int c_conv_241_ih  = 3;
const int c_conv_241_iw  = 3;

typedef ap_uint<8> t_input_88;
typedef ap_uint<32> t_conv_25_acc;
const int c_conv_25_ich    = 32;
const int c_conv_25_och    = 32;
const int c_conv_25_ih     = 16;
const int c_conv_25_iw     = 16;
const int c_conv_25_ow     = 16;
const int c_conv_25_oh     = 16;
const int c_conv_25_fw     = 3;
const int c_conv_25_fh     = 3;
const int c_conv_25_relu   = 1;
const int c_input_88_split  = 0;
const int c_conv_25_stride = 1;
const int c_conv_25_pad    = 1;


typedef ap_uint<8> t_relu_171;
const int c_add_26_ich    = 32;
const int c_add_26_ih     = 16;
const int c_add_26_iw     = 16;


typedef ap_uint<8> t_input_88;
const int c_relu_27_ich    = 32;
const int c_relu_27_ih     = 16;
const int c_relu_27_iw     = 16;

typedef ap_uint<8> t_input_88_skip;


typedef ap_uint<8> t_conv_244_st;
typedef ap_uint<8> t_conv_244;
const int c_conv_244_och = 32;
const int c_conv_244_ich = 32;
const int c_conv_244_ih  = 3;
const int c_conv_244_iw  = 3;

typedef ap_uint<8> t_conv_175;
typedef ap_uint<32> t_conv_28_acc;
const int c_conv_28_ich    = 32;
const int c_conv_28_och    = 32;
const int c_conv_28_ih     = 16;
const int c_conv_28_iw     = 16;
const int c_conv_28_ow     = 16;
const int c_conv_28_oh     = 16;
const int c_conv_28_fw     = 3;
const int c_conv_28_fh     = 3;
const int c_conv_28_relu   = 1;
const int c_conv_175_split  = 0;
const int c_conv_28_stride = 1;
const int c_conv_28_pad    = 1;


typedef ap_uint<8> t_conv_175;
const int c_relu_29_ich    = 32;
const int c_relu_29_ih     = 16;
const int c_relu_29_iw     = 16;


typedef ap_uint<8> t_conv_247_st;
typedef ap_uint<8> t_conv_247;
const int c_conv_247_och = 32;
const int c_conv_247_ich = 32;
const int c_conv_247_ih  = 3;
const int c_conv_247_iw  = 3;

typedef ap_uint<8> t_input_104;
typedef ap_uint<32> t_conv_30_acc;
const int c_conv_30_ich    = 32;
const int c_conv_30_och    = 32;
const int c_conv_30_ih     = 16;
const int c_conv_30_iw     = 16;
const int c_conv_30_ow     = 16;
const int c_conv_30_oh     = 16;
const int c_conv_30_fw     = 3;
const int c_conv_30_fh     = 3;
const int c_conv_30_relu   = 1;
const int c_input_104_split  = 2;
const int c_conv_30_stride = 1;
const int c_conv_30_pad    = 1;


typedef ap_uint<8> t_relu_178;
const int c_add_31_ich    = 32;
const int c_add_31_ih     = 16;
const int c_add_31_iw     = 16;


typedef ap_uint<8> t_input_104;
const int c_relu_32_ich    = 32;
const int c_relu_32_ih     = 16;
const int c_relu_32_iw     = 16;


typedef ap_uint<8> t_conv_250_st;
typedef ap_uint<8> t_conv_250;
const int c_conv_250_och = 64;
const int c_conv_250_ich = 32;
const int c_conv_250_ih  = 3;
const int c_conv_250_iw  = 3;

typedef ap_uint<8> t_conv_182;
typedef ap_uint<32> t_conv_33_acc;
const int c_conv_33_ich    = 32;
const int c_conv_33_och    = 64;
const int c_conv_33_ih     = 16;
const int c_conv_33_iw     = 16;
const int c_conv_33_ow     = 8;
const int c_conv_33_oh     = 8;
const int c_conv_33_fw     = 3;
const int c_conv_33_fh     = 3;
const int c_conv_33_relu   = 1;
const int c_conv_182_split  = 0;
const int c_conv_33_stride = 2;
const int c_conv_33_pad    = 1;


typedef ap_uint<8> t_conv_182;
const int c_relu_34_ich    = 64;
const int c_relu_34_ih     = 8;
const int c_relu_34_iw     = 8;


typedef ap_uint<8> t_conv_253_st;
typedef ap_uint<8> t_conv_253;
const int c_conv_253_och = 64;
const int c_conv_253_ich = 64;
const int c_conv_253_ih  = 3;
const int c_conv_253_iw  = 3;

typedef ap_uint<8> t_input_124;
typedef ap_uint<32> t_conv_35_acc;
const int c_conv_35_ich    = 64;
const int c_conv_35_och    = 64;
const int c_conv_35_ih     = 8;
const int c_conv_35_iw     = 8;
const int c_conv_35_ow     = 8;
const int c_conv_35_oh     = 8;
const int c_conv_35_fw     = 3;
const int c_conv_35_fh     = 3;
const int c_conv_35_relu   = 1;
const int c_input_124_split  = 0;
const int c_conv_35_stride = 1;
const int c_conv_35_pad    = 1;


typedef ap_uint<8> t_conv_256_st;
typedef ap_uint<8> t_conv_256;
const int c_conv_256_och = 64;
const int c_conv_256_ich = 32;
const int c_conv_256_ih  = 1;
const int c_conv_256_iw  = 1;

typedef ap_uint<8> t_add_255;
typedef ap_uint<32> t_conv_36_acc;
const int c_conv_36_ich    = 32;
const int c_conv_36_och    = 64;
const int c_conv_36_ih     = 16;
const int c_conv_36_iw     = 16;
const int c_conv_36_ow     = 8;
const int c_conv_36_oh     = 8;
const int c_conv_36_fw     = 1;
const int c_conv_36_fh     = 1;
const int c_conv_36_relu   = 0;
const int c_add_255_split  = 0;
const int c_conv_36_stride = 2;
const int c_conv_36_pad    = 0;


typedef ap_uint<8> t_relu_187;
const int c_add_37_ich    = 64;
const int c_add_37_ih     = 8;
const int c_add_37_iw     = 8;


typedef ap_uint<8> t_input_124;
const int c_relu_38_ich    = 64;
const int c_relu_38_ih     = 8;
const int c_relu_38_iw     = 8;

typedef ap_uint<8> t_input_124_skip;


typedef ap_uint<8> t_conv_259_st;
typedef ap_uint<8> t_conv_259;
const int c_conv_259_och = 64;
const int c_conv_259_ich = 64;
const int c_conv_259_ih  = 3;
const int c_conv_259_iw  = 3;

typedef ap_uint<8> t_conv_191;
typedef ap_uint<32> t_conv_39_acc;
const int c_conv_39_ich    = 64;
const int c_conv_39_och    = 64;
const int c_conv_39_ih     = 8;
const int c_conv_39_iw     = 8;
const int c_conv_39_ow     = 8;
const int c_conv_39_oh     = 8;
const int c_conv_39_fw     = 3;
const int c_conv_39_fh     = 3;
const int c_conv_39_relu   = 1;
const int c_conv_191_split  = 0;
const int c_conv_39_stride = 1;
const int c_conv_39_pad    = 1;


typedef ap_uint<8> t_conv_191;
const int c_relu_40_ich    = 64;
const int c_relu_40_ih     = 8;
const int c_relu_40_iw     = 8;


typedef ap_uint<8> t_conv_262_st;
typedef ap_uint<8> t_conv_262;
const int c_conv_262_och = 64;
const int c_conv_262_ich = 64;
const int c_conv_262_ih  = 3;
const int c_conv_262_iw  = 3;

typedef ap_uint<8> t_input_140;
typedef ap_uint<32> t_conv_41_acc;
const int c_conv_41_ich    = 64;
const int c_conv_41_och    = 64;
const int c_conv_41_ih     = 8;
const int c_conv_41_iw     = 8;
const int c_conv_41_ow     = 8;
const int c_conv_41_oh     = 8;
const int c_conv_41_fw     = 3;
const int c_conv_41_fh     = 3;
const int c_conv_41_relu   = 1;
const int c_input_140_split  = 0;
const int c_conv_41_stride = 1;
const int c_conv_41_pad    = 1;


typedef ap_uint<8> t_relu_194;
const int c_add_42_ich    = 64;
const int c_add_42_ih     = 8;
const int c_add_42_iw     = 8;


typedef ap_uint<8> t_input_140;
const int c_relu_43_ich    = 64;
const int c_relu_43_ih     = 8;
const int c_relu_43_iw     = 8;

typedef ap_uint<8> t_input_140_skip;


typedef ap_uint<8> t_conv_265_st;
typedef ap_uint<8> t_conv_265;
const int c_conv_265_och = 64;
const int c_conv_265_ich = 64;
const int c_conv_265_ih  = 3;
const int c_conv_265_iw  = 3;

typedef ap_uint<8> t_conv_198;
typedef ap_uint<32> t_conv_44_acc;
const int c_conv_44_ich    = 64;
const int c_conv_44_och    = 64;
const int c_conv_44_ih     = 8;
const int c_conv_44_iw     = 8;
const int c_conv_44_ow     = 8;
const int c_conv_44_oh     = 8;
const int c_conv_44_fw     = 3;
const int c_conv_44_fh     = 3;
const int c_conv_44_relu   = 1;
const int c_conv_198_split  = 0;
const int c_conv_44_stride = 1;
const int c_conv_44_pad    = 1;


typedef ap_uint<8> t_conv_198;
const int c_relu_45_ich    = 64;
const int c_relu_45_ih     = 8;
const int c_relu_45_iw     = 8;


typedef ap_uint<8> t_conv_268_st;
typedef ap_uint<8> t_conv_268;
const int c_conv_268_och = 64;
const int c_conv_268_ich = 64;
const int c_conv_268_ih  = 3;
const int c_conv_268_iw  = 3;

typedef ap_uint<8> t_pad_202;
typedef ap_uint<32> t_conv_46_acc;
const int c_conv_46_ich    = 64;
const int c_conv_46_och    = 64;
const int c_conv_46_ih     = 8;
const int c_conv_46_iw     = 8;
const int c_conv_46_ow     = 8;
const int c_conv_46_oh     = 8;
const int c_conv_46_fw     = 3;
const int c_conv_46_fh     = 3;
const int c_conv_46_relu   = 1;
const int c_pad_202_split  = 0;
const int c_conv_46_stride = 1;
const int c_conv_46_pad    = 1;


typedef ap_uint<8> t_relu_201;
const int c_add_47_ich    = 64;
const int c_add_47_ih     = 8;
const int c_add_47_iw     = 8;


typedef ap_uint<8> t_pad_202;
const int c_relu_48_ich    = 64;
const int c_relu_48_ih     = 8;
const int c_relu_48_iw     = 8;


typedef ap_uint<8> t_averagepool_203;
typedef ap_uint<8> t_pad_49_acc;
const int c_pad_49_ich    = 64;
const int c_pad_49_och    = 64;
const int c_pad_49_ih     = 8;
const int c_pad_49_iw     = 8;
const int c_pad_49_oh     = 8;
const int c_pad_49_ow     = 8;
const int c_pad_49_pad    = 0;


typedef ap_uint<8> t_input_156;
typedef ap_uint<8> t_averagepool_50_acc;
const int c_averagepool_50_ich    = 64;
const int c_averagepool_50_och    = 64;
const int c_averagepool_50_ih     = 8;
const int c_averagepool_50_iw     = 8;
const int c_averagepool_50_oh     = 1;
const int c_averagepool_50_ow     = 1;
const int c_averagepool_50_fh     = 8;
const int c_averagepool_50_fw     = 8;
const int c_averagepool_50_stride = 1;
const int c_averagepool_50_pad    = 0;


typedef ap_uint<8> t_conv_271_st;
typedef ap_uint<8> t_conv_271;
const int c_conv_271_och = 10;
const int c_conv_271_ich = 64;
const int c_conv_271_ih  = 1;
const int c_conv_271_iw  = 1;

typedef ap_uint<8> t_output;
typedef ap_uint<32> t_conv_51_acc;
const int c_conv_51_ich    = 64;
const int c_conv_51_och    = 10;
const int c_conv_51_ih     = 1;
const int c_conv_51_iw     = 1;
const int c_conv_51_ow     = 1;
const int c_conv_51_oh     = 1;
const int c_conv_51_fw     = 1;
const int c_conv_51_fh     = 1;
const int c_conv_51_relu   = 0;
const int c_output_split  = 0;
const int c_conv_51_stride = 1;
const int c_conv_51_pad    = 0;

void Network(
	t_i_data* i_data,
	t_weight* i_weight,
	t_o_data* o_data
);
#endif