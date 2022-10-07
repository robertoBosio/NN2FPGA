#ifndef __NETWORK__
#define __NETWORK__
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"
#include <stdint.h>
#define c_i_data 32
typedef ap_axiu<c_i_data, 0, 0, 0> t_i_data;
#define c_o_data 32
typedef ap_axiu<c_o_data, 0, 0, 0> t_o_data;
#define c_last_depth 256
typedef ap_uint<1> t_last;
typedef uint8_t t_input;

const int c_input_ich    = 3;
const int c_input_ih     = 32;
const int c_input_iw     = 32;


const int c_output_och = 10;
const int c_output_oh  = 1;
const int c_output_ow  = 1;


typedef uint8_t t_conv_209_st;
typedef ap_uint<8> t_conv_209;
const int c_conv_209_och = 16;
const int c_conv_209_ich = 3;
const int c_conv_209_ih  = 3;
const int c_conv_209_iw  = 3;

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


typedef uint8_t t_conv_212_st;
typedef ap_uint<8> t_conv_212;
const int c_conv_212_och = 16;
const int c_conv_212_ich = 16;
const int c_conv_212_ih  = 3;
const int c_conv_212_iw  = 3;

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


typedef uint8_t t_conv_215_st;
typedef ap_uint<8> t_conv_215;
const int c_conv_215_och = 16;
const int c_conv_215_ich = 16;
const int c_conv_215_ih  = 3;
const int c_conv_215_iw  = 3;

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


typedef uint8_t t_conv_218_st;
typedef ap_uint<8> t_conv_218;
const int c_conv_218_och = 16;
const int c_conv_218_ich = 16;
const int c_conv_218_ih  = 3;
const int c_conv_218_iw  = 3;

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


typedef uint8_t t_conv_221_st;
typedef ap_uint<8> t_conv_221;
const int c_conv_221_och = 16;
const int c_conv_221_ich = 16;
const int c_conv_221_ih  = 3;
const int c_conv_221_iw  = 3;

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


typedef uint8_t t_conv_224_st;
typedef ap_uint<8> t_conv_224;
const int c_conv_224_och = 16;
const int c_conv_224_ich = 16;
const int c_conv_224_ih  = 3;
const int c_conv_224_iw  = 3;

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


typedef uint8_t t_conv_227_st;
typedef ap_uint<8> t_conv_227;
const int c_conv_227_och = 16;
const int c_conv_227_ich = 16;
const int c_conv_227_ih  = 3;
const int c_conv_227_iw  = 3;

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


typedef uint8_t t_conv_230_st;
typedef ap_uint<8> t_conv_230;
const int c_conv_230_och = 32;
const int c_conv_230_ich = 16;
const int c_conv_230_ih  = 3;
const int c_conv_230_iw  = 3;

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


typedef uint8_t t_conv_233_st;
typedef ap_uint<8> t_conv_233;
const int c_conv_233_och = 32;
const int c_conv_233_ich = 32;
const int c_conv_233_ih  = 3;
const int c_conv_233_iw  = 3;

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


typedef uint8_t t_conv_236_st;
typedef ap_uint<8> t_conv_236;
const int c_conv_236_och = 32;
const int c_conv_236_ich = 16;
const int c_conv_236_ih  = 1;
const int c_conv_236_iw  = 1;

typedef ap_uint<8> t_add_235;
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
const int c_add_235_split  = 0;
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


typedef uint8_t t_conv_239_st;
typedef ap_uint<8> t_conv_239;
const int c_conv_239_och = 32;
const int c_conv_239_ich = 32;
const int c_conv_239_ih  = 3;
const int c_conv_239_iw  = 3;

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


typedef uint8_t t_conv_242_st;
typedef ap_uint<8> t_conv_242;
const int c_conv_242_och = 32;
const int c_conv_242_ich = 32;
const int c_conv_242_ih  = 3;
const int c_conv_242_iw  = 3;

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


typedef uint8_t t_conv_245_st;
typedef ap_uint<8> t_conv_245;
const int c_conv_245_och = 32;
const int c_conv_245_ich = 32;
const int c_conv_245_ih  = 3;
const int c_conv_245_iw  = 3;

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


typedef uint8_t t_conv_248_st;
typedef ap_uint<8> t_conv_248;
const int c_conv_248_och = 32;
const int c_conv_248_ich = 32;
const int c_conv_248_ih  = 3;
const int c_conv_248_iw  = 3;

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


typedef uint8_t t_conv_251_st;
typedef ap_uint<8> t_conv_251;
const int c_conv_251_och = 64;
const int c_conv_251_ich = 32;
const int c_conv_251_ih  = 3;
const int c_conv_251_iw  = 3;

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


typedef uint8_t t_conv_254_st;
typedef ap_uint<8> t_conv_254;
const int c_conv_254_och = 64;
const int c_conv_254_ich = 64;
const int c_conv_254_ih  = 3;
const int c_conv_254_iw  = 3;

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


typedef uint8_t t_conv_257_st;
typedef ap_uint<8> t_conv_257;
const int c_conv_257_och = 64;
const int c_conv_257_ich = 32;
const int c_conv_257_ih  = 1;
const int c_conv_257_iw  = 1;

typedef ap_uint<8> t_add_256;
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
const int c_add_256_split  = 0;
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


typedef uint8_t t_conv_260_st;
typedef ap_uint<8> t_conv_260;
const int c_conv_260_och = 64;
const int c_conv_260_ich = 64;
const int c_conv_260_ih  = 3;
const int c_conv_260_iw  = 3;

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


typedef uint8_t t_conv_263_st;
typedef ap_uint<8> t_conv_263;
const int c_conv_263_och = 64;
const int c_conv_263_ich = 64;
const int c_conv_263_ih  = 3;
const int c_conv_263_iw  = 3;

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


typedef uint8_t t_conv_266_st;
typedef ap_uint<8> t_conv_266;
const int c_conv_266_och = 64;
const int c_conv_266_ich = 64;
const int c_conv_266_ih  = 3;
const int c_conv_266_iw  = 3;

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


typedef uint8_t t_conv_269_st;
typedef ap_uint<8> t_conv_269;
const int c_conv_269_och = 64;
const int c_conv_269_ich = 64;
const int c_conv_269_ih  = 3;
const int c_conv_269_iw  = 3;

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


typedef ap_uint<8> t_averagepool_204;
typedef ap_uint<8> t_pad_50_acc;
const int c_pad_50_ich    = 64;
const int c_pad_50_och    = 64;
const int c_pad_50_ih     = 8;
const int c_pad_50_iw     = 8;
const int c_pad_50_oh     = 8;
const int c_pad_50_ow     = 8;
const int c_pad_50_pad    = 0;


typedef ap_uint<8> t_input_156;
typedef ap_uint<8> t_averagepool_51_acc;
const int c_averagepool_51_ich    = 64;
const int c_averagepool_51_och    = 64;
const int c_averagepool_51_ih     = 8;
const int c_averagepool_51_iw     = 8;
const int c_averagepool_51_oh     = 1;
const int c_averagepool_51_ow     = 1;
const int c_averagepool_51_fh     = 8;
const int c_averagepool_51_fw     = 8;
const int c_averagepool_51_stride = 1;
const int c_averagepool_51_pad    = 0;


typedef uint8_t t_conv_272_st;
typedef ap_uint<8> t_conv_272;
const int c_conv_272_och = 10;
const int c_conv_272_ich = 64;
const int c_conv_272_ih  = 1;
const int c_conv_272_iw  = 1;

typedef ap_uint<8> t_output;
typedef ap_uint<32> t_conv_52_acc;
const int c_conv_52_ich    = 64;
const int c_conv_52_och    = 10;
const int c_conv_52_ih     = 1;
const int c_conv_52_iw     = 1;
const int c_conv_52_ow     = 1;
const int c_conv_52_oh     = 1;
const int c_conv_52_fw     = 1;
const int c_conv_52_fh     = 1;
const int c_conv_52_relu   = 0;
const int c_output_split  = 0;
const int c_conv_52_stride = 1;
const int c_conv_52_pad    = 0;

void Network(
	t_i_data* i_data,
	t_o_data* o_data
);
#endif