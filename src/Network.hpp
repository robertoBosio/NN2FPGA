#ifndef __NETWORK__
#define __NETWORK__
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"
#include <stdint.h>
#define c_i_data 64
typedef ap_axiu<c_i_data, 0, 0, 0> t_i_data;
#define c_o_data 8
typedef ap_axiu<c_o_data, 0, 0, 0> t_o_data;
typedef ap_uint<1> t_last;
typedef uint8_t t_input;

const int c_input_ich    = 3;
const int c_input_ih     = 224;
const int c_input_iw     = 224;


const int c_output_och = 1000;
const int c_output_oh  = 1;
const int c_output_ow  = 1;


typedef uint8_t t_conv_497_st;
typedef ap_uint<8> t_conv_497;
const int c_conv_497_och = 64;
const int c_conv_497_ich = 3;
const int c_conv_497_ih  = 7;
const int c_conv_497_iw  = 7;

typedef ap_uint<8> t_maxpool_323;
typedef ap_uint<32> t_conv_0_acc;
const int c_conv_0_ich    = 3;
const int c_conv_0_och    = 64;
const int c_conv_0_ih     = 224;
const int c_conv_0_iw     = 224;
const int c_conv_0_ow     = 112;
const int c_conv_0_oh     = 112;
const int c_conv_0_fw     = 7;
const int c_conv_0_fh     = 7;
const int c_conv_0_relu   = 1;
const int c_maxpool_323_split  = 0;
const int c_conv_0_stride = 2;
const int c_conv_0_pad    = 3;
const int c_conv_0_split  = 50;


typedef ap_uint<8> t_maxpool_323;
const int c_relu_1_ich    = 64;
const int c_relu_1_ih     = 112;
const int c_relu_1_iw     = 112;


typedef ap_uint<8> t_input_8;
typedef ap_uint<8> t_maxpool_2_acc;
const int c_maxpool_2_ich    = 64;
const int c_maxpool_2_och    = 64;
const int c_maxpool_2_ih     = 112;
const int c_maxpool_2_iw     = 112;
const int c_maxpool_2_oh     = 56;
const int c_maxpool_2_ow     = 56;
const int c_maxpool_2_fh     = 56;
const int c_maxpool_2_fw     = 56;
const int c_maxpool_2_stride = 1;
const int c_maxpool_2_pad    = 0;
const int c_maxpool_2_pool   = 1;


typedef uint8_t t_conv_500_st;
typedef ap_uint<8> t_conv_500;
const int c_conv_500_och = 64;
const int c_conv_500_ich = 64;
const int c_conv_500_ih  = 1;
const int c_conv_500_iw  = 1;

typedef ap_uint<8> t_conv_327;
typedef ap_uint<32> t_conv_3_acc;
const int c_conv_3_ich    = 64;
const int c_conv_3_och    = 64;
const int c_conv_3_ih     = 56;
const int c_conv_3_iw     = 56;
const int c_conv_3_ow     = 56;
const int c_conv_3_oh     = 56;
const int c_conv_3_fw     = 1;
const int c_conv_3_fh     = 1;
const int c_conv_3_relu   = 1;
const int c_conv_327_split  = 0;
const int c_conv_3_stride = 1;
const int c_conv_3_pad    = 0;
const int c_conv_3_split  = 2;


typedef ap_uint<8> t_conv_327;
const int c_relu_4_ich    = 64;
const int c_relu_4_ih     = 56;
const int c_relu_4_iw     = 56;


typedef uint8_t t_conv_503_st;
typedef ap_uint<8> t_conv_503;
const int c_conv_503_och = 64;
const int c_conv_503_ich = 64;
const int c_conv_503_ih  = 3;
const int c_conv_503_iw  = 3;

typedef ap_uint<8> t_conv_330;
typedef ap_uint<32> t_conv_5_acc;
const int c_conv_5_ich    = 64;
const int c_conv_5_och    = 64;
const int c_conv_5_ih     = 56;
const int c_conv_5_iw     = 56;
const int c_conv_5_ow     = 56;
const int c_conv_5_oh     = 56;
const int c_conv_5_fw     = 3;
const int c_conv_5_fh     = 3;
const int c_conv_5_relu   = 1;
const int c_conv_330_split  = 0;
const int c_conv_5_stride = 1;
const int c_conv_5_pad    = 1;
const int c_conv_5_split  = 10;


typedef ap_uint<8> t_conv_330;
const int c_relu_6_ich    = 64;
const int c_relu_6_ih     = 56;
const int c_relu_6_iw     = 56;


typedef uint8_t t_conv_506_st;
typedef ap_uint<8> t_conv_506;
const int c_conv_506_och = 256;
const int c_conv_506_ich = 64;
const int c_conv_506_ih  = 1;
const int c_conv_506_iw  = 1;

typedef ap_uint<8> t_input_36;
typedef ap_uint<32> t_conv_7_acc;
const int c_conv_7_ich    = 64;
const int c_conv_7_och    = 256;
const int c_conv_7_ih     = 56;
const int c_conv_7_iw     = 56;
const int c_conv_7_ow     = 56;
const int c_conv_7_oh     = 56;
const int c_conv_7_fw     = 1;
const int c_conv_7_fh     = 1;
const int c_conv_7_relu   = 1;
const int c_input_36_split  = 0;
const int c_conv_7_stride = 1;
const int c_conv_7_pad    = 0;
const int c_conv_7_split  = 2;


typedef uint8_t t_conv_509_st;
typedef ap_uint<8> t_conv_509;
const int c_conv_509_och = 256;
const int c_conv_509_ich = 64;
const int c_conv_509_ih  = 1;
const int c_conv_509_iw  = 1;

typedef ap_uint<8> t_add_508;
typedef ap_uint<32> t_conv_8_acc;
const int c_conv_8_ich    = 64;
const int c_conv_8_och    = 256;
const int c_conv_8_ih     = 56;
const int c_conv_8_iw     = 56;
const int c_conv_8_ow     = 56;
const int c_conv_8_oh     = 56;
const int c_conv_8_fw     = 1;
const int c_conv_8_fh     = 1;
const int c_conv_8_relu   = 0;
const int c_add_508_split  = 0;
const int c_conv_8_stride = 1;
const int c_conv_8_pad    = 0;
const int c_conv_8_split  = 2;


typedef ap_uint<8> t_relu_335;
const int c_add_9_ich    = 256;
const int c_add_9_ih     = 56;
const int c_add_9_iw     = 56;


typedef ap_uint<8> t_input_36;
const int c_relu_10_ich    = 256;
const int c_relu_10_ih     = 56;
const int c_relu_10_iw     = 56;

typedef ap_uint<8> t_input_36_skip;


typedef uint8_t t_conv_512_st;
typedef ap_uint<8> t_conv_512;
const int c_conv_512_och = 64;
const int c_conv_512_ich = 256;
const int c_conv_512_ih  = 1;
const int c_conv_512_iw  = 1;

typedef ap_uint<8> t_conv_339;
typedef ap_uint<32> t_conv_11_acc;
const int c_conv_11_ich    = 256;
const int c_conv_11_och    = 64;
const int c_conv_11_ih     = 56;
const int c_conv_11_iw     = 56;
const int c_conv_11_ow     = 56;
const int c_conv_11_oh     = 56;
const int c_conv_11_fw     = 1;
const int c_conv_11_fh     = 1;
const int c_conv_11_relu   = 1;
const int c_conv_339_split  = 0;
const int c_conv_11_stride = 1;
const int c_conv_11_pad    = 0;
const int c_conv_11_split  = 2;


typedef ap_uint<8> t_conv_339;
const int c_relu_12_ich    = 64;
const int c_relu_12_ih     = 56;
const int c_relu_12_iw     = 56;


typedef uint8_t t_conv_515_st;
typedef ap_uint<8> t_conv_515;
const int c_conv_515_och = 64;
const int c_conv_515_ich = 64;
const int c_conv_515_ih  = 3;
const int c_conv_515_iw  = 3;

typedef ap_uint<8> t_conv_342;
typedef ap_uint<32> t_conv_13_acc;
const int c_conv_13_ich    = 64;
const int c_conv_13_och    = 64;
const int c_conv_13_ih     = 56;
const int c_conv_13_iw     = 56;
const int c_conv_13_ow     = 56;
const int c_conv_13_oh     = 56;
const int c_conv_13_fw     = 3;
const int c_conv_13_fh     = 3;
const int c_conv_13_relu   = 1;
const int c_conv_342_split  = 0;
const int c_conv_13_stride = 1;
const int c_conv_13_pad    = 1;
const int c_conv_13_split  = 10;


typedef ap_uint<8> t_conv_342;
const int c_relu_14_ich    = 64;
const int c_relu_14_ih     = 56;
const int c_relu_14_iw     = 56;


typedef uint8_t t_conv_518_st;
typedef ap_uint<8> t_conv_518;
const int c_conv_518_och = 256;
const int c_conv_518_ich = 64;
const int c_conv_518_ih  = 1;
const int c_conv_518_iw  = 1;

typedef ap_uint<8> t_input_60;
typedef ap_uint<32> t_conv_15_acc;
const int c_conv_15_ich    = 64;
const int c_conv_15_och    = 256;
const int c_conv_15_ih     = 56;
const int c_conv_15_iw     = 56;
const int c_conv_15_ow     = 56;
const int c_conv_15_oh     = 56;
const int c_conv_15_fw     = 1;
const int c_conv_15_fh     = 1;
const int c_conv_15_relu   = 1;
const int c_input_60_split  = 0;
const int c_conv_15_stride = 1;
const int c_conv_15_pad    = 0;
const int c_conv_15_split  = 2;


typedef ap_uint<8> t_relu_345;
const int c_add_16_ich    = 256;
const int c_add_16_ih     = 56;
const int c_add_16_iw     = 56;


typedef ap_uint<8> t_input_60;
const int c_relu_17_ich    = 256;
const int c_relu_17_ih     = 56;
const int c_relu_17_iw     = 56;

typedef ap_uint<8> t_input_60_skip;


typedef uint8_t t_conv_521_st;
typedef ap_uint<8> t_conv_521;
const int c_conv_521_och = 64;
const int c_conv_521_ich = 256;
const int c_conv_521_ih  = 1;
const int c_conv_521_iw  = 1;

typedef ap_uint<8> t_conv_349;
typedef ap_uint<32> t_conv_18_acc;
const int c_conv_18_ich    = 256;
const int c_conv_18_och    = 64;
const int c_conv_18_ih     = 56;
const int c_conv_18_iw     = 56;
const int c_conv_18_ow     = 56;
const int c_conv_18_oh     = 56;
const int c_conv_18_fw     = 1;
const int c_conv_18_fh     = 1;
const int c_conv_18_relu   = 1;
const int c_conv_349_split  = 0;
const int c_conv_18_stride = 1;
const int c_conv_18_pad    = 0;
const int c_conv_18_split  = 2;


typedef ap_uint<8> t_conv_349;
const int c_relu_19_ich    = 64;
const int c_relu_19_ih     = 56;
const int c_relu_19_iw     = 56;


typedef uint8_t t_conv_524_st;
typedef ap_uint<8> t_conv_524;
const int c_conv_524_och = 64;
const int c_conv_524_ich = 64;
const int c_conv_524_ih  = 3;
const int c_conv_524_iw  = 3;

typedef ap_uint<8> t_conv_352;
typedef ap_uint<32> t_conv_20_acc;
const int c_conv_20_ich    = 64;
const int c_conv_20_och    = 64;
const int c_conv_20_ih     = 56;
const int c_conv_20_iw     = 56;
const int c_conv_20_ow     = 56;
const int c_conv_20_oh     = 56;
const int c_conv_20_fw     = 3;
const int c_conv_20_fh     = 3;
const int c_conv_20_relu   = 1;
const int c_conv_352_split  = 0;
const int c_conv_20_stride = 1;
const int c_conv_20_pad    = 1;
const int c_conv_20_split  = 10;


typedef ap_uint<8> t_conv_352;
const int c_relu_21_ich    = 64;
const int c_relu_21_ih     = 56;
const int c_relu_21_iw     = 56;


typedef uint8_t t_conv_527_st;
typedef ap_uint<8> t_conv_527;
const int c_conv_527_och = 256;
const int c_conv_527_ich = 64;
const int c_conv_527_ih  = 1;
const int c_conv_527_iw  = 1;

typedef ap_uint<8> t_input_84;
typedef ap_uint<32> t_conv_22_acc;
const int c_conv_22_ich    = 64;
const int c_conv_22_och    = 256;
const int c_conv_22_ih     = 56;
const int c_conv_22_iw     = 56;
const int c_conv_22_ow     = 56;
const int c_conv_22_oh     = 56;
const int c_conv_22_fw     = 1;
const int c_conv_22_fh     = 1;
const int c_conv_22_relu   = 1;
const int c_input_84_split  = 2;
const int c_conv_22_stride = 1;
const int c_conv_22_pad    = 0;
const int c_conv_22_split  = 2;


typedef ap_uint<8> t_relu_355;
const int c_add_23_ich    = 256;
const int c_add_23_ih     = 56;
const int c_add_23_iw     = 56;


typedef ap_uint<8> t_input_84;
const int c_relu_24_ich    = 256;
const int c_relu_24_ih     = 56;
const int c_relu_24_iw     = 56;


typedef uint8_t t_conv_530_st;
typedef ap_uint<8> t_conv_530;
const int c_conv_530_och = 128;
const int c_conv_530_ich = 256;
const int c_conv_530_ih  = 1;
const int c_conv_530_iw  = 1;

typedef ap_uint<8> t_conv_359;
typedef ap_uint<32> t_conv_25_acc;
const int c_conv_25_ich    = 256;
const int c_conv_25_och    = 128;
const int c_conv_25_ih     = 56;
const int c_conv_25_iw     = 56;
const int c_conv_25_ow     = 56;
const int c_conv_25_oh     = 56;
const int c_conv_25_fw     = 1;
const int c_conv_25_fh     = 1;
const int c_conv_25_relu   = 1;
const int c_conv_359_split  = 0;
const int c_conv_25_stride = 1;
const int c_conv_25_pad    = 0;
const int c_conv_25_split  = 2;


typedef ap_uint<8> t_conv_359;
const int c_relu_26_ich    = 128;
const int c_relu_26_ih     = 56;
const int c_relu_26_iw     = 56;


typedef uint8_t t_conv_533_st;
typedef ap_uint<8> t_conv_533;
const int c_conv_533_och = 128;
const int c_conv_533_ich = 128;
const int c_conv_533_ih  = 3;
const int c_conv_533_iw  = 3;

typedef ap_uint<8> t_conv_362;
typedef ap_uint<32> t_conv_27_acc;
const int c_conv_27_ich    = 128;
const int c_conv_27_och    = 128;
const int c_conv_27_ih     = 56;
const int c_conv_27_iw     = 56;
const int c_conv_27_ow     = 28;
const int c_conv_27_oh     = 28;
const int c_conv_27_fw     = 3;
const int c_conv_27_fh     = 3;
const int c_conv_27_relu   = 1;
const int c_conv_362_split  = 0;
const int c_conv_27_stride = 2;
const int c_conv_27_pad    = 1;
const int c_conv_27_split  = 10;


typedef ap_uint<8> t_conv_362;
const int c_relu_28_ich    = 128;
const int c_relu_28_ih     = 28;
const int c_relu_28_iw     = 28;


typedef uint8_t t_conv_536_st;
typedef ap_uint<8> t_conv_536;
const int c_conv_536_och = 512;
const int c_conv_536_ich = 128;
const int c_conv_536_ih  = 1;
const int c_conv_536_iw  = 1;

typedef ap_uint<8> t_input_112;
typedef ap_uint<32> t_conv_29_acc;
const int c_conv_29_ich    = 128;
const int c_conv_29_och    = 512;
const int c_conv_29_ih     = 28;
const int c_conv_29_iw     = 28;
const int c_conv_29_ow     = 28;
const int c_conv_29_oh     = 28;
const int c_conv_29_fw     = 1;
const int c_conv_29_fh     = 1;
const int c_conv_29_relu   = 1;
const int c_input_112_split  = 0;
const int c_conv_29_stride = 1;
const int c_conv_29_pad    = 0;
const int c_conv_29_split  = 2;


typedef uint8_t t_conv_539_st;
typedef ap_uint<8> t_conv_539;
const int c_conv_539_och = 512;
const int c_conv_539_ich = 256;
const int c_conv_539_ih  = 1;
const int c_conv_539_iw  = 1;

typedef ap_uint<8> t_add_538;
typedef ap_uint<32> t_conv_30_acc;
const int c_conv_30_ich    = 256;
const int c_conv_30_och    = 512;
const int c_conv_30_ih     = 56;
const int c_conv_30_iw     = 56;
const int c_conv_30_ow     = 28;
const int c_conv_30_oh     = 28;
const int c_conv_30_fw     = 1;
const int c_conv_30_fh     = 1;
const int c_conv_30_relu   = 0;
const int c_add_538_split  = 0;
const int c_conv_30_stride = 2;
const int c_conv_30_pad    = 0;
const int c_conv_30_split  = 2;


typedef ap_uint<8> t_relu_367;
const int c_add_31_ich    = 512;
const int c_add_31_ih     = 28;
const int c_add_31_iw     = 28;


typedef ap_uint<8> t_input_112;
const int c_relu_32_ich    = 512;
const int c_relu_32_ih     = 28;
const int c_relu_32_iw     = 28;

typedef ap_uint<8> t_input_112_skip;


typedef uint8_t t_conv_542_st;
typedef ap_uint<8> t_conv_542;
const int c_conv_542_och = 128;
const int c_conv_542_ich = 512;
const int c_conv_542_ih  = 1;
const int c_conv_542_iw  = 1;

typedef ap_uint<8> t_conv_371;
typedef ap_uint<32> t_conv_33_acc;
const int c_conv_33_ich    = 512;
const int c_conv_33_och    = 128;
const int c_conv_33_ih     = 28;
const int c_conv_33_iw     = 28;
const int c_conv_33_ow     = 28;
const int c_conv_33_oh     = 28;
const int c_conv_33_fw     = 1;
const int c_conv_33_fh     = 1;
const int c_conv_33_relu   = 1;
const int c_conv_371_split  = 0;
const int c_conv_33_stride = 1;
const int c_conv_33_pad    = 0;
const int c_conv_33_split  = 2;


typedef ap_uint<8> t_conv_371;
const int c_relu_34_ich    = 128;
const int c_relu_34_ih     = 28;
const int c_relu_34_iw     = 28;


typedef uint8_t t_conv_545_st;
typedef ap_uint<8> t_conv_545;
const int c_conv_545_och = 128;
const int c_conv_545_ich = 128;
const int c_conv_545_ih  = 3;
const int c_conv_545_iw  = 3;

typedef ap_uint<8> t_conv_374;
typedef ap_uint<32> t_conv_35_acc;
const int c_conv_35_ich    = 128;
const int c_conv_35_och    = 128;
const int c_conv_35_ih     = 28;
const int c_conv_35_iw     = 28;
const int c_conv_35_ow     = 28;
const int c_conv_35_oh     = 28;
const int c_conv_35_fw     = 3;
const int c_conv_35_fh     = 3;
const int c_conv_35_relu   = 1;
const int c_conv_374_split  = 0;
const int c_conv_35_stride = 1;
const int c_conv_35_pad    = 1;
const int c_conv_35_split  = 10;


typedef ap_uint<8> t_conv_374;
const int c_relu_36_ich    = 128;
const int c_relu_36_ih     = 28;
const int c_relu_36_iw     = 28;


typedef uint8_t t_conv_548_st;
typedef ap_uint<8> t_conv_548;
const int c_conv_548_och = 512;
const int c_conv_548_ich = 128;
const int c_conv_548_ih  = 1;
const int c_conv_548_iw  = 1;

typedef ap_uint<8> t_input_136;
typedef ap_uint<32> t_conv_37_acc;
const int c_conv_37_ich    = 128;
const int c_conv_37_och    = 512;
const int c_conv_37_ih     = 28;
const int c_conv_37_iw     = 28;
const int c_conv_37_ow     = 28;
const int c_conv_37_oh     = 28;
const int c_conv_37_fw     = 1;
const int c_conv_37_fh     = 1;
const int c_conv_37_relu   = 1;
const int c_input_136_split  = 0;
const int c_conv_37_stride = 1;
const int c_conv_37_pad    = 0;
const int c_conv_37_split  = 2;


typedef ap_uint<8> t_relu_377;
const int c_add_38_ich    = 512;
const int c_add_38_ih     = 28;
const int c_add_38_iw     = 28;


typedef ap_uint<8> t_input_136;
const int c_relu_39_ich    = 512;
const int c_relu_39_ih     = 28;
const int c_relu_39_iw     = 28;

typedef ap_uint<8> t_input_136_skip;


typedef uint8_t t_conv_551_st;
typedef ap_uint<8> t_conv_551;
const int c_conv_551_och = 128;
const int c_conv_551_ich = 512;
const int c_conv_551_ih  = 1;
const int c_conv_551_iw  = 1;

typedef ap_uint<8> t_conv_381;
typedef ap_uint<32> t_conv_40_acc;
const int c_conv_40_ich    = 512;
const int c_conv_40_och    = 128;
const int c_conv_40_ih     = 28;
const int c_conv_40_iw     = 28;
const int c_conv_40_ow     = 28;
const int c_conv_40_oh     = 28;
const int c_conv_40_fw     = 1;
const int c_conv_40_fh     = 1;
const int c_conv_40_relu   = 1;
const int c_conv_381_split  = 0;
const int c_conv_40_stride = 1;
const int c_conv_40_pad    = 0;
const int c_conv_40_split  = 2;


typedef ap_uint<8> t_conv_381;
const int c_relu_41_ich    = 128;
const int c_relu_41_ih     = 28;
const int c_relu_41_iw     = 28;


typedef uint8_t t_conv_554_st;
typedef ap_uint<8> t_conv_554;
const int c_conv_554_och = 128;
const int c_conv_554_ich = 128;
const int c_conv_554_ih  = 3;
const int c_conv_554_iw  = 3;

typedef ap_uint<8> t_conv_384;
typedef ap_uint<32> t_conv_42_acc;
const int c_conv_42_ich    = 128;
const int c_conv_42_och    = 128;
const int c_conv_42_ih     = 28;
const int c_conv_42_iw     = 28;
const int c_conv_42_ow     = 28;
const int c_conv_42_oh     = 28;
const int c_conv_42_fw     = 3;
const int c_conv_42_fh     = 3;
const int c_conv_42_relu   = 1;
const int c_conv_384_split  = 0;
const int c_conv_42_stride = 1;
const int c_conv_42_pad    = 1;
const int c_conv_42_split  = 10;


typedef ap_uint<8> t_conv_384;
const int c_relu_43_ich    = 128;
const int c_relu_43_ih     = 28;
const int c_relu_43_iw     = 28;


typedef uint8_t t_conv_557_st;
typedef ap_uint<8> t_conv_557;
const int c_conv_557_och = 512;
const int c_conv_557_ich = 128;
const int c_conv_557_ih  = 1;
const int c_conv_557_iw  = 1;

typedef ap_uint<8> t_input_160;
typedef ap_uint<32> t_conv_44_acc;
const int c_conv_44_ich    = 128;
const int c_conv_44_och    = 512;
const int c_conv_44_ih     = 28;
const int c_conv_44_iw     = 28;
const int c_conv_44_ow     = 28;
const int c_conv_44_oh     = 28;
const int c_conv_44_fw     = 1;
const int c_conv_44_fh     = 1;
const int c_conv_44_relu   = 1;
const int c_input_160_split  = 0;
const int c_conv_44_stride = 1;
const int c_conv_44_pad    = 0;
const int c_conv_44_split  = 2;


typedef ap_uint<8> t_relu_387;
const int c_add_45_ich    = 512;
const int c_add_45_ih     = 28;
const int c_add_45_iw     = 28;


typedef ap_uint<8> t_input_160;
const int c_relu_46_ich    = 512;
const int c_relu_46_ih     = 28;
const int c_relu_46_iw     = 28;

typedef ap_uint<8> t_input_160_skip;


typedef uint8_t t_conv_560_st;
typedef ap_uint<8> t_conv_560;
const int c_conv_560_och = 128;
const int c_conv_560_ich = 512;
const int c_conv_560_ih  = 1;
const int c_conv_560_iw  = 1;

typedef ap_uint<8> t_conv_391;
typedef ap_uint<32> t_conv_47_acc;
const int c_conv_47_ich    = 512;
const int c_conv_47_och    = 128;
const int c_conv_47_ih     = 28;
const int c_conv_47_iw     = 28;
const int c_conv_47_ow     = 28;
const int c_conv_47_oh     = 28;
const int c_conv_47_fw     = 1;
const int c_conv_47_fh     = 1;
const int c_conv_47_relu   = 1;
const int c_conv_391_split  = 0;
const int c_conv_47_stride = 1;
const int c_conv_47_pad    = 0;
const int c_conv_47_split  = 2;


typedef ap_uint<8> t_conv_391;
const int c_relu_48_ich    = 128;
const int c_relu_48_ih     = 28;
const int c_relu_48_iw     = 28;


typedef uint8_t t_conv_563_st;
typedef ap_uint<8> t_conv_563;
const int c_conv_563_och = 128;
const int c_conv_563_ich = 128;
const int c_conv_563_ih  = 3;
const int c_conv_563_iw  = 3;

typedef ap_uint<8> t_conv_394;
typedef ap_uint<32> t_conv_49_acc;
const int c_conv_49_ich    = 128;
const int c_conv_49_och    = 128;
const int c_conv_49_ih     = 28;
const int c_conv_49_iw     = 28;
const int c_conv_49_ow     = 28;
const int c_conv_49_oh     = 28;
const int c_conv_49_fw     = 3;
const int c_conv_49_fh     = 3;
const int c_conv_49_relu   = 1;
const int c_conv_394_split  = 0;
const int c_conv_49_stride = 1;
const int c_conv_49_pad    = 1;
const int c_conv_49_split  = 10;


typedef ap_uint<8> t_conv_394;
const int c_relu_50_ich    = 128;
const int c_relu_50_ih     = 28;
const int c_relu_50_iw     = 28;


typedef uint8_t t_conv_566_st;
typedef ap_uint<8> t_conv_566;
const int c_conv_566_och = 512;
const int c_conv_566_ich = 128;
const int c_conv_566_ih  = 1;
const int c_conv_566_iw  = 1;

typedef ap_uint<8> t_input_184;
typedef ap_uint<32> t_conv_51_acc;
const int c_conv_51_ich    = 128;
const int c_conv_51_och    = 512;
const int c_conv_51_ih     = 28;
const int c_conv_51_iw     = 28;
const int c_conv_51_ow     = 28;
const int c_conv_51_oh     = 28;
const int c_conv_51_fw     = 1;
const int c_conv_51_fh     = 1;
const int c_conv_51_relu   = 1;
const int c_input_184_split  = 2;
const int c_conv_51_stride = 1;
const int c_conv_51_pad    = 0;
const int c_conv_51_split  = 2;


typedef ap_uint<8> t_relu_397;
const int c_add_52_ich    = 512;
const int c_add_52_ih     = 28;
const int c_add_52_iw     = 28;


typedef ap_uint<8> t_input_184;
const int c_relu_53_ich    = 512;
const int c_relu_53_ih     = 28;
const int c_relu_53_iw     = 28;


typedef uint8_t t_conv_569_st;
typedef ap_uint<8> t_conv_569;
const int c_conv_569_och = 256;
const int c_conv_569_ich = 512;
const int c_conv_569_ih  = 1;
const int c_conv_569_iw  = 1;

typedef ap_uint<8> t_conv_401;
typedef ap_uint<32> t_conv_54_acc;
const int c_conv_54_ich    = 512;
const int c_conv_54_och    = 256;
const int c_conv_54_ih     = 28;
const int c_conv_54_iw     = 28;
const int c_conv_54_ow     = 28;
const int c_conv_54_oh     = 28;
const int c_conv_54_fw     = 1;
const int c_conv_54_fh     = 1;
const int c_conv_54_relu   = 1;
const int c_conv_401_split  = 0;
const int c_conv_54_stride = 1;
const int c_conv_54_pad    = 0;
const int c_conv_54_split  = 2;


typedef ap_uint<8> t_conv_401;
const int c_relu_55_ich    = 256;
const int c_relu_55_ih     = 28;
const int c_relu_55_iw     = 28;


typedef uint8_t t_conv_572_st;
typedef ap_uint<8> t_conv_572;
const int c_conv_572_och = 256;
const int c_conv_572_ich = 256;
const int c_conv_572_ih  = 3;
const int c_conv_572_iw  = 3;

typedef ap_uint<8> t_conv_404;
typedef ap_uint<32> t_conv_56_acc;
const int c_conv_56_ich    = 256;
const int c_conv_56_och    = 256;
const int c_conv_56_ih     = 28;
const int c_conv_56_iw     = 28;
const int c_conv_56_ow     = 14;
const int c_conv_56_oh     = 14;
const int c_conv_56_fw     = 3;
const int c_conv_56_fh     = 3;
const int c_conv_56_relu   = 1;
const int c_conv_404_split  = 0;
const int c_conv_56_stride = 2;
const int c_conv_56_pad    = 1;
const int c_conv_56_split  = 10;


typedef ap_uint<8> t_conv_404;
const int c_relu_57_ich    = 256;
const int c_relu_57_ih     = 14;
const int c_relu_57_iw     = 14;


typedef uint8_t t_conv_575_st;
typedef ap_uint<8> t_conv_575;
const int c_conv_575_och = 1024;
const int c_conv_575_ich = 256;
const int c_conv_575_ih  = 1;
const int c_conv_575_iw  = 1;

typedef ap_uint<8> t_input_212;
typedef ap_uint<32> t_conv_58_acc;
const int c_conv_58_ich    = 256;
const int c_conv_58_och    = 1024;
const int c_conv_58_ih     = 14;
const int c_conv_58_iw     = 14;
const int c_conv_58_ow     = 14;
const int c_conv_58_oh     = 14;
const int c_conv_58_fw     = 1;
const int c_conv_58_fh     = 1;
const int c_conv_58_relu   = 1;
const int c_input_212_split  = 0;
const int c_conv_58_stride = 1;
const int c_conv_58_pad    = 0;
const int c_conv_58_split  = 2;


typedef uint8_t t_conv_578_st;
typedef ap_uint<8> t_conv_578;
const int c_conv_578_och = 1024;
const int c_conv_578_ich = 512;
const int c_conv_578_ih  = 1;
const int c_conv_578_iw  = 1;

typedef ap_uint<8> t_add_577;
typedef ap_uint<32> t_conv_59_acc;
const int c_conv_59_ich    = 512;
const int c_conv_59_och    = 1024;
const int c_conv_59_ih     = 28;
const int c_conv_59_iw     = 28;
const int c_conv_59_ow     = 14;
const int c_conv_59_oh     = 14;
const int c_conv_59_fw     = 1;
const int c_conv_59_fh     = 1;
const int c_conv_59_relu   = 0;
const int c_add_577_split  = 0;
const int c_conv_59_stride = 2;
const int c_conv_59_pad    = 0;
const int c_conv_59_split  = 2;


typedef ap_uint<8> t_relu_409;
const int c_add_60_ich    = 1024;
const int c_add_60_ih     = 14;
const int c_add_60_iw     = 14;


typedef ap_uint<8> t_input_212;
const int c_relu_61_ich    = 1024;
const int c_relu_61_ih     = 14;
const int c_relu_61_iw     = 14;

typedef ap_uint<8> t_input_212_skip;


typedef uint8_t t_conv_581_st;
typedef ap_uint<8> t_conv_581;
const int c_conv_581_och = 256;
const int c_conv_581_ich = 1024;
const int c_conv_581_ih  = 1;
const int c_conv_581_iw  = 1;

typedef ap_uint<8> t_conv_413;
typedef ap_uint<32> t_conv_62_acc;
const int c_conv_62_ich    = 1024;
const int c_conv_62_och    = 256;
const int c_conv_62_ih     = 14;
const int c_conv_62_iw     = 14;
const int c_conv_62_ow     = 14;
const int c_conv_62_oh     = 14;
const int c_conv_62_fw     = 1;
const int c_conv_62_fh     = 1;
const int c_conv_62_relu   = 1;
const int c_conv_413_split  = 0;
const int c_conv_62_stride = 1;
const int c_conv_62_pad    = 0;
const int c_conv_62_split  = 2;


typedef ap_uint<8> t_conv_413;
const int c_relu_63_ich    = 256;
const int c_relu_63_ih     = 14;
const int c_relu_63_iw     = 14;


typedef uint8_t t_conv_584_st;
typedef ap_uint<8> t_conv_584;
const int c_conv_584_och = 256;
const int c_conv_584_ich = 256;
const int c_conv_584_ih  = 3;
const int c_conv_584_iw  = 3;

typedef ap_uint<8> t_conv_416;
typedef ap_uint<32> t_conv_64_acc;
const int c_conv_64_ich    = 256;
const int c_conv_64_och    = 256;
const int c_conv_64_ih     = 14;
const int c_conv_64_iw     = 14;
const int c_conv_64_ow     = 14;
const int c_conv_64_oh     = 14;
const int c_conv_64_fw     = 3;
const int c_conv_64_fh     = 3;
const int c_conv_64_relu   = 1;
const int c_conv_416_split  = 0;
const int c_conv_64_stride = 1;
const int c_conv_64_pad    = 1;
const int c_conv_64_split  = 10;


typedef ap_uint<8> t_conv_416;
const int c_relu_65_ich    = 256;
const int c_relu_65_ih     = 14;
const int c_relu_65_iw     = 14;


typedef uint8_t t_conv_587_st;
typedef ap_uint<8> t_conv_587;
const int c_conv_587_och = 1024;
const int c_conv_587_ich = 256;
const int c_conv_587_ih  = 1;
const int c_conv_587_iw  = 1;

typedef ap_uint<8> t_input_236;
typedef ap_uint<32> t_conv_66_acc;
const int c_conv_66_ich    = 256;
const int c_conv_66_och    = 1024;
const int c_conv_66_ih     = 14;
const int c_conv_66_iw     = 14;
const int c_conv_66_ow     = 14;
const int c_conv_66_oh     = 14;
const int c_conv_66_fw     = 1;
const int c_conv_66_fh     = 1;
const int c_conv_66_relu   = 1;
const int c_input_236_split  = 0;
const int c_conv_66_stride = 1;
const int c_conv_66_pad    = 0;
const int c_conv_66_split  = 2;


typedef ap_uint<8> t_relu_419;
const int c_add_67_ich    = 1024;
const int c_add_67_ih     = 14;
const int c_add_67_iw     = 14;


typedef ap_uint<8> t_input_236;
const int c_relu_68_ich    = 1024;
const int c_relu_68_ih     = 14;
const int c_relu_68_iw     = 14;

typedef ap_uint<8> t_input_236_skip;


typedef uint8_t t_conv_590_st;
typedef ap_uint<8> t_conv_590;
const int c_conv_590_och = 256;
const int c_conv_590_ich = 1024;
const int c_conv_590_ih  = 1;
const int c_conv_590_iw  = 1;

typedef ap_uint<8> t_conv_423;
typedef ap_uint<32> t_conv_69_acc;
const int c_conv_69_ich    = 1024;
const int c_conv_69_och    = 256;
const int c_conv_69_ih     = 14;
const int c_conv_69_iw     = 14;
const int c_conv_69_ow     = 14;
const int c_conv_69_oh     = 14;
const int c_conv_69_fw     = 1;
const int c_conv_69_fh     = 1;
const int c_conv_69_relu   = 1;
const int c_conv_423_split  = 0;
const int c_conv_69_stride = 1;
const int c_conv_69_pad    = 0;
const int c_conv_69_split  = 2;


typedef ap_uint<8> t_conv_423;
const int c_relu_70_ich    = 256;
const int c_relu_70_ih     = 14;
const int c_relu_70_iw     = 14;


typedef uint8_t t_conv_593_st;
typedef ap_uint<8> t_conv_593;
const int c_conv_593_och = 256;
const int c_conv_593_ich = 256;
const int c_conv_593_ih  = 3;
const int c_conv_593_iw  = 3;

typedef ap_uint<8> t_conv_426;
typedef ap_uint<32> t_conv_71_acc;
const int c_conv_71_ich    = 256;
const int c_conv_71_och    = 256;
const int c_conv_71_ih     = 14;
const int c_conv_71_iw     = 14;
const int c_conv_71_ow     = 14;
const int c_conv_71_oh     = 14;
const int c_conv_71_fw     = 3;
const int c_conv_71_fh     = 3;
const int c_conv_71_relu   = 1;
const int c_conv_426_split  = 0;
const int c_conv_71_stride = 1;
const int c_conv_71_pad    = 1;
const int c_conv_71_split  = 10;


typedef ap_uint<8> t_conv_426;
const int c_relu_72_ich    = 256;
const int c_relu_72_ih     = 14;
const int c_relu_72_iw     = 14;


typedef uint8_t t_conv_596_st;
typedef ap_uint<8> t_conv_596;
const int c_conv_596_och = 1024;
const int c_conv_596_ich = 256;
const int c_conv_596_ih  = 1;
const int c_conv_596_iw  = 1;

typedef ap_uint<8> t_input_260;
typedef ap_uint<32> t_conv_73_acc;
const int c_conv_73_ich    = 256;
const int c_conv_73_och    = 1024;
const int c_conv_73_ih     = 14;
const int c_conv_73_iw     = 14;
const int c_conv_73_ow     = 14;
const int c_conv_73_oh     = 14;
const int c_conv_73_fw     = 1;
const int c_conv_73_fh     = 1;
const int c_conv_73_relu   = 1;
const int c_input_260_split  = 0;
const int c_conv_73_stride = 1;
const int c_conv_73_pad    = 0;
const int c_conv_73_split  = 2;


typedef ap_uint<8> t_relu_429;
const int c_add_74_ich    = 1024;
const int c_add_74_ih     = 14;
const int c_add_74_iw     = 14;


typedef ap_uint<8> t_input_260;
const int c_relu_75_ich    = 1024;
const int c_relu_75_ih     = 14;
const int c_relu_75_iw     = 14;

typedef ap_uint<8> t_input_260_skip;


typedef uint8_t t_conv_599_st;
typedef ap_uint<8> t_conv_599;
const int c_conv_599_och = 256;
const int c_conv_599_ich = 1024;
const int c_conv_599_ih  = 1;
const int c_conv_599_iw  = 1;

typedef ap_uint<8> t_conv_433;
typedef ap_uint<32> t_conv_76_acc;
const int c_conv_76_ich    = 1024;
const int c_conv_76_och    = 256;
const int c_conv_76_ih     = 14;
const int c_conv_76_iw     = 14;
const int c_conv_76_ow     = 14;
const int c_conv_76_oh     = 14;
const int c_conv_76_fw     = 1;
const int c_conv_76_fh     = 1;
const int c_conv_76_relu   = 1;
const int c_conv_433_split  = 0;
const int c_conv_76_stride = 1;
const int c_conv_76_pad    = 0;
const int c_conv_76_split  = 2;


typedef ap_uint<8> t_conv_433;
const int c_relu_77_ich    = 256;
const int c_relu_77_ih     = 14;
const int c_relu_77_iw     = 14;


typedef uint8_t t_conv_602_st;
typedef ap_uint<8> t_conv_602;
const int c_conv_602_och = 256;
const int c_conv_602_ich = 256;
const int c_conv_602_ih  = 3;
const int c_conv_602_iw  = 3;

typedef ap_uint<8> t_conv_436;
typedef ap_uint<32> t_conv_78_acc;
const int c_conv_78_ich    = 256;
const int c_conv_78_och    = 256;
const int c_conv_78_ih     = 14;
const int c_conv_78_iw     = 14;
const int c_conv_78_ow     = 14;
const int c_conv_78_oh     = 14;
const int c_conv_78_fw     = 3;
const int c_conv_78_fh     = 3;
const int c_conv_78_relu   = 1;
const int c_conv_436_split  = 0;
const int c_conv_78_stride = 1;
const int c_conv_78_pad    = 1;
const int c_conv_78_split  = 10;


typedef ap_uint<8> t_conv_436;
const int c_relu_79_ich    = 256;
const int c_relu_79_ih     = 14;
const int c_relu_79_iw     = 14;


typedef uint8_t t_conv_605_st;
typedef ap_uint<8> t_conv_605;
const int c_conv_605_och = 1024;
const int c_conv_605_ich = 256;
const int c_conv_605_ih  = 1;
const int c_conv_605_iw  = 1;

typedef ap_uint<8> t_input_284;
typedef ap_uint<32> t_conv_80_acc;
const int c_conv_80_ich    = 256;
const int c_conv_80_och    = 1024;
const int c_conv_80_ih     = 14;
const int c_conv_80_iw     = 14;
const int c_conv_80_ow     = 14;
const int c_conv_80_oh     = 14;
const int c_conv_80_fw     = 1;
const int c_conv_80_fh     = 1;
const int c_conv_80_relu   = 1;
const int c_input_284_split  = 0;
const int c_conv_80_stride = 1;
const int c_conv_80_pad    = 0;
const int c_conv_80_split  = 2;


typedef ap_uint<8> t_relu_439;
const int c_add_81_ich    = 1024;
const int c_add_81_ih     = 14;
const int c_add_81_iw     = 14;


typedef ap_uint<8> t_input_284;
const int c_relu_82_ich    = 1024;
const int c_relu_82_ih     = 14;
const int c_relu_82_iw     = 14;

typedef ap_uint<8> t_input_284_skip;


typedef uint8_t t_conv_608_st;
typedef ap_uint<8> t_conv_608;
const int c_conv_608_och = 256;
const int c_conv_608_ich = 1024;
const int c_conv_608_ih  = 1;
const int c_conv_608_iw  = 1;

typedef ap_uint<8> t_conv_443;
typedef ap_uint<32> t_conv_83_acc;
const int c_conv_83_ich    = 1024;
const int c_conv_83_och    = 256;
const int c_conv_83_ih     = 14;
const int c_conv_83_iw     = 14;
const int c_conv_83_ow     = 14;
const int c_conv_83_oh     = 14;
const int c_conv_83_fw     = 1;
const int c_conv_83_fh     = 1;
const int c_conv_83_relu   = 1;
const int c_conv_443_split  = 0;
const int c_conv_83_stride = 1;
const int c_conv_83_pad    = 0;
const int c_conv_83_split  = 2;


typedef ap_uint<8> t_conv_443;
const int c_relu_84_ich    = 256;
const int c_relu_84_ih     = 14;
const int c_relu_84_iw     = 14;


typedef uint8_t t_conv_611_st;
typedef ap_uint<8> t_conv_611;
const int c_conv_611_och = 256;
const int c_conv_611_ich = 256;
const int c_conv_611_ih  = 3;
const int c_conv_611_iw  = 3;

typedef ap_uint<8> t_conv_446;
typedef ap_uint<32> t_conv_85_acc;
const int c_conv_85_ich    = 256;
const int c_conv_85_och    = 256;
const int c_conv_85_ih     = 14;
const int c_conv_85_iw     = 14;
const int c_conv_85_ow     = 14;
const int c_conv_85_oh     = 14;
const int c_conv_85_fw     = 3;
const int c_conv_85_fh     = 3;
const int c_conv_85_relu   = 1;
const int c_conv_446_split  = 0;
const int c_conv_85_stride = 1;
const int c_conv_85_pad    = 1;
const int c_conv_85_split  = 10;


typedef ap_uint<8> t_conv_446;
const int c_relu_86_ich    = 256;
const int c_relu_86_ih     = 14;
const int c_relu_86_iw     = 14;


typedef uint8_t t_conv_614_st;
typedef ap_uint<8> t_conv_614;
const int c_conv_614_och = 1024;
const int c_conv_614_ich = 256;
const int c_conv_614_ih  = 1;
const int c_conv_614_iw  = 1;

typedef ap_uint<8> t_input_308;
typedef ap_uint<32> t_conv_87_acc;
const int c_conv_87_ich    = 256;
const int c_conv_87_och    = 1024;
const int c_conv_87_ih     = 14;
const int c_conv_87_iw     = 14;
const int c_conv_87_ow     = 14;
const int c_conv_87_oh     = 14;
const int c_conv_87_fw     = 1;
const int c_conv_87_fh     = 1;
const int c_conv_87_relu   = 1;
const int c_input_308_split  = 0;
const int c_conv_87_stride = 1;
const int c_conv_87_pad    = 0;
const int c_conv_87_split  = 2;


typedef ap_uint<8> t_relu_449;
const int c_add_88_ich    = 1024;
const int c_add_88_ih     = 14;
const int c_add_88_iw     = 14;


typedef ap_uint<8> t_input_308;
const int c_relu_89_ich    = 1024;
const int c_relu_89_ih     = 14;
const int c_relu_89_iw     = 14;

typedef ap_uint<8> t_input_308_skip;


typedef uint8_t t_conv_617_st;
typedef ap_uint<8> t_conv_617;
const int c_conv_617_och = 256;
const int c_conv_617_ich = 1024;
const int c_conv_617_ih  = 1;
const int c_conv_617_iw  = 1;

typedef ap_uint<8> t_conv_453;
typedef ap_uint<32> t_conv_90_acc;
const int c_conv_90_ich    = 1024;
const int c_conv_90_och    = 256;
const int c_conv_90_ih     = 14;
const int c_conv_90_iw     = 14;
const int c_conv_90_ow     = 14;
const int c_conv_90_oh     = 14;
const int c_conv_90_fw     = 1;
const int c_conv_90_fh     = 1;
const int c_conv_90_relu   = 1;
const int c_conv_453_split  = 0;
const int c_conv_90_stride = 1;
const int c_conv_90_pad    = 0;
const int c_conv_90_split  = 2;


typedef ap_uint<8> t_conv_453;
const int c_relu_91_ich    = 256;
const int c_relu_91_ih     = 14;
const int c_relu_91_iw     = 14;


typedef uint8_t t_conv_620_st;
typedef ap_uint<8> t_conv_620;
const int c_conv_620_och = 256;
const int c_conv_620_ich = 256;
const int c_conv_620_ih  = 3;
const int c_conv_620_iw  = 3;

typedef ap_uint<8> t_conv_456;
typedef ap_uint<32> t_conv_92_acc;
const int c_conv_92_ich    = 256;
const int c_conv_92_och    = 256;
const int c_conv_92_ih     = 14;
const int c_conv_92_iw     = 14;
const int c_conv_92_ow     = 14;
const int c_conv_92_oh     = 14;
const int c_conv_92_fw     = 3;
const int c_conv_92_fh     = 3;
const int c_conv_92_relu   = 1;
const int c_conv_456_split  = 0;
const int c_conv_92_stride = 1;
const int c_conv_92_pad    = 1;
const int c_conv_92_split  = 10;


typedef ap_uint<8> t_conv_456;
const int c_relu_93_ich    = 256;
const int c_relu_93_ih     = 14;
const int c_relu_93_iw     = 14;


typedef uint8_t t_conv_623_st;
typedef ap_uint<8> t_conv_623;
const int c_conv_623_och = 1024;
const int c_conv_623_ich = 256;
const int c_conv_623_ih  = 1;
const int c_conv_623_iw  = 1;

typedef ap_uint<8> t_input_332;
typedef ap_uint<32> t_conv_94_acc;
const int c_conv_94_ich    = 256;
const int c_conv_94_och    = 1024;
const int c_conv_94_ih     = 14;
const int c_conv_94_iw     = 14;
const int c_conv_94_ow     = 14;
const int c_conv_94_oh     = 14;
const int c_conv_94_fw     = 1;
const int c_conv_94_fh     = 1;
const int c_conv_94_relu   = 1;
const int c_input_332_split  = 2;
const int c_conv_94_stride = 1;
const int c_conv_94_pad    = 0;
const int c_conv_94_split  = 2;


typedef ap_uint<8> t_relu_459;
const int c_add_95_ich    = 1024;
const int c_add_95_ih     = 14;
const int c_add_95_iw     = 14;


typedef ap_uint<8> t_input_332;
const int c_relu_96_ich    = 1024;
const int c_relu_96_ih     = 14;
const int c_relu_96_iw     = 14;


typedef uint8_t t_conv_626_st;
typedef ap_uint<8> t_conv_626;
const int c_conv_626_och = 512;
const int c_conv_626_ich = 1024;
const int c_conv_626_ih  = 1;
const int c_conv_626_iw  = 1;

typedef ap_uint<8> t_conv_463;
typedef ap_uint<32> t_conv_97_acc;
const int c_conv_97_ich    = 1024;
const int c_conv_97_och    = 512;
const int c_conv_97_ih     = 14;
const int c_conv_97_iw     = 14;
const int c_conv_97_ow     = 14;
const int c_conv_97_oh     = 14;
const int c_conv_97_fw     = 1;
const int c_conv_97_fh     = 1;
const int c_conv_97_relu   = 1;
const int c_conv_463_split  = 0;
const int c_conv_97_stride = 1;
const int c_conv_97_pad    = 0;
const int c_conv_97_split  = 2;


typedef ap_uint<8> t_conv_463;
const int c_relu_98_ich    = 512;
const int c_relu_98_ih     = 14;
const int c_relu_98_iw     = 14;


typedef uint8_t t_conv_629_st;
typedef ap_uint<8> t_conv_629;
const int c_conv_629_och = 512;
const int c_conv_629_ich = 512;
const int c_conv_629_ih  = 3;
const int c_conv_629_iw  = 3;

typedef ap_uint<8> t_conv_466;
typedef ap_uint<32> t_conv_99_acc;
const int c_conv_99_ich    = 512;
const int c_conv_99_och    = 512;
const int c_conv_99_ih     = 14;
const int c_conv_99_iw     = 14;
const int c_conv_99_ow     = 7;
const int c_conv_99_oh     = 7;
const int c_conv_99_fw     = 3;
const int c_conv_99_fh     = 3;
const int c_conv_99_relu   = 1;
const int c_conv_466_split  = 0;
const int c_conv_99_stride = 2;
const int c_conv_99_pad    = 1;
const int c_conv_99_split  = 10;


typedef ap_uint<8> t_conv_466;
const int c_relu_100_ich    = 512;
const int c_relu_100_ih     = 7;
const int c_relu_100_iw     = 7;


typedef uint8_t t_conv_632_st;
typedef ap_uint<8> t_conv_632;
const int c_conv_632_och = 2048;
const int c_conv_632_ich = 512;
const int c_conv_632_ih  = 1;
const int c_conv_632_iw  = 1;

typedef ap_uint<8> t_input_360;
typedef ap_uint<32> t_conv_101_acc;
const int c_conv_101_ich    = 512;
const int c_conv_101_och    = 2048;
const int c_conv_101_ih     = 7;
const int c_conv_101_iw     = 7;
const int c_conv_101_ow     = 7;
const int c_conv_101_oh     = 7;
const int c_conv_101_fw     = 1;
const int c_conv_101_fh     = 1;
const int c_conv_101_relu   = 1;
const int c_input_360_split  = 0;
const int c_conv_101_stride = 1;
const int c_conv_101_pad    = 0;
const int c_conv_101_split  = 2;


typedef uint8_t t_conv_635_st;
typedef ap_uint<8> t_conv_635;
const int c_conv_635_och = 2048;
const int c_conv_635_ich = 1024;
const int c_conv_635_ih  = 1;
const int c_conv_635_iw  = 1;

typedef ap_uint<8> t_add_634;
typedef ap_uint<32> t_conv_102_acc;
const int c_conv_102_ich    = 1024;
const int c_conv_102_och    = 2048;
const int c_conv_102_ih     = 14;
const int c_conv_102_iw     = 14;
const int c_conv_102_ow     = 7;
const int c_conv_102_oh     = 7;
const int c_conv_102_fw     = 1;
const int c_conv_102_fh     = 1;
const int c_conv_102_relu   = 0;
const int c_add_634_split  = 0;
const int c_conv_102_stride = 2;
const int c_conv_102_pad    = 0;
const int c_conv_102_split  = 2;


typedef ap_uint<8> t_relu_471;
const int c_add_103_ich    = 2048;
const int c_add_103_ih     = 7;
const int c_add_103_iw     = 7;


typedef ap_uint<8> t_input_360;
const int c_relu_104_ich    = 2048;
const int c_relu_104_ih     = 7;
const int c_relu_104_iw     = 7;

typedef ap_uint<8> t_input_360_skip;


typedef uint8_t t_conv_638_st;
typedef ap_uint<8> t_conv_638;
const int c_conv_638_och = 512;
const int c_conv_638_ich = 2048;
const int c_conv_638_ih  = 1;
const int c_conv_638_iw  = 1;

typedef ap_uint<8> t_conv_475;
typedef ap_uint<32> t_conv_105_acc;
const int c_conv_105_ich    = 2048;
const int c_conv_105_och    = 512;
const int c_conv_105_ih     = 7;
const int c_conv_105_iw     = 7;
const int c_conv_105_ow     = 7;
const int c_conv_105_oh     = 7;
const int c_conv_105_fw     = 1;
const int c_conv_105_fh     = 1;
const int c_conv_105_relu   = 1;
const int c_conv_475_split  = 0;
const int c_conv_105_stride = 1;
const int c_conv_105_pad    = 0;
const int c_conv_105_split  = 2;


typedef ap_uint<8> t_conv_475;
const int c_relu_106_ich    = 512;
const int c_relu_106_ih     = 7;
const int c_relu_106_iw     = 7;


typedef uint8_t t_conv_641_st;
typedef ap_uint<8> t_conv_641;
const int c_conv_641_och = 512;
const int c_conv_641_ich = 512;
const int c_conv_641_ih  = 3;
const int c_conv_641_iw  = 3;

typedef ap_uint<8> t_conv_478;
typedef ap_uint<32> t_conv_107_acc;
const int c_conv_107_ich    = 512;
const int c_conv_107_och    = 512;
const int c_conv_107_ih     = 7;
const int c_conv_107_iw     = 7;
const int c_conv_107_ow     = 7;
const int c_conv_107_oh     = 7;
const int c_conv_107_fw     = 3;
const int c_conv_107_fh     = 3;
const int c_conv_107_relu   = 1;
const int c_conv_478_split  = 0;
const int c_conv_107_stride = 1;
const int c_conv_107_pad    = 1;
const int c_conv_107_split  = 10;


typedef ap_uint<8> t_conv_478;
const int c_relu_108_ich    = 512;
const int c_relu_108_ih     = 7;
const int c_relu_108_iw     = 7;


typedef uint8_t t_conv_644_st;
typedef ap_uint<8> t_conv_644;
const int c_conv_644_och = 2048;
const int c_conv_644_ich = 512;
const int c_conv_644_ih  = 1;
const int c_conv_644_iw  = 1;

typedef ap_uint<8> t_input_384;
typedef ap_uint<32> t_conv_109_acc;
const int c_conv_109_ich    = 512;
const int c_conv_109_och    = 2048;
const int c_conv_109_ih     = 7;
const int c_conv_109_iw     = 7;
const int c_conv_109_ow     = 7;
const int c_conv_109_oh     = 7;
const int c_conv_109_fw     = 1;
const int c_conv_109_fh     = 1;
const int c_conv_109_relu   = 1;
const int c_input_384_split  = 0;
const int c_conv_109_stride = 1;
const int c_conv_109_pad    = 0;
const int c_conv_109_split  = 2;


typedef ap_uint<8> t_relu_481;
const int c_add_110_ich    = 2048;
const int c_add_110_ih     = 7;
const int c_add_110_iw     = 7;


typedef ap_uint<8> t_input_384;
const int c_relu_111_ich    = 2048;
const int c_relu_111_ih     = 7;
const int c_relu_111_iw     = 7;

typedef ap_uint<8> t_input_384_skip;


typedef uint8_t t_conv_647_st;
typedef ap_uint<8> t_conv_647;
const int c_conv_647_och = 512;
const int c_conv_647_ich = 2048;
const int c_conv_647_ih  = 1;
const int c_conv_647_iw  = 1;

typedef ap_uint<8> t_conv_485;
typedef ap_uint<32> t_conv_112_acc;
const int c_conv_112_ich    = 2048;
const int c_conv_112_och    = 512;
const int c_conv_112_ih     = 7;
const int c_conv_112_iw     = 7;
const int c_conv_112_ow     = 7;
const int c_conv_112_oh     = 7;
const int c_conv_112_fw     = 1;
const int c_conv_112_fh     = 1;
const int c_conv_112_relu   = 1;
const int c_conv_485_split  = 0;
const int c_conv_112_stride = 1;
const int c_conv_112_pad    = 0;
const int c_conv_112_split  = 2;


typedef ap_uint<8> t_conv_485;
const int c_relu_113_ich    = 512;
const int c_relu_113_ih     = 7;
const int c_relu_113_iw     = 7;


typedef uint8_t t_conv_650_st;
typedef ap_uint<8> t_conv_650;
const int c_conv_650_och = 512;
const int c_conv_650_ich = 512;
const int c_conv_650_ih  = 3;
const int c_conv_650_iw  = 3;

typedef ap_uint<8> t_conv_488;
typedef ap_uint<32> t_conv_114_acc;
const int c_conv_114_ich    = 512;
const int c_conv_114_och    = 512;
const int c_conv_114_ih     = 7;
const int c_conv_114_iw     = 7;
const int c_conv_114_ow     = 7;
const int c_conv_114_oh     = 7;
const int c_conv_114_fw     = 3;
const int c_conv_114_fh     = 3;
const int c_conv_114_relu   = 1;
const int c_conv_488_split  = 0;
const int c_conv_114_stride = 1;
const int c_conv_114_pad    = 1;
const int c_conv_114_split  = 10;


typedef ap_uint<8> t_conv_488;
const int c_relu_115_ich    = 512;
const int c_relu_115_ih     = 7;
const int c_relu_115_iw     = 7;


typedef uint8_t t_conv_653_st;
typedef ap_uint<8> t_conv_653;
const int c_conv_653_och = 2048;
const int c_conv_653_ich = 512;
const int c_conv_653_ih  = 1;
const int c_conv_653_iw  = 1;

typedef ap_uint<8> t_input_408;
typedef ap_uint<32> t_conv_116_acc;
const int c_conv_116_ich    = 512;
const int c_conv_116_och    = 2048;
const int c_conv_116_ih     = 7;
const int c_conv_116_iw     = 7;
const int c_conv_116_ow     = 7;
const int c_conv_116_oh     = 7;
const int c_conv_116_fw     = 1;
const int c_conv_116_fh     = 1;
const int c_conv_116_relu   = 1;
const int c_input_408_split  = 0;
const int c_conv_116_stride = 1;
const int c_conv_116_pad    = 0;
const int c_conv_116_split  = 2;


typedef ap_uint<8> t_relu_491;
const int c_add_117_ich    = 2048;
const int c_add_117_ih     = 7;
const int c_add_117_iw     = 7;


typedef ap_uint<8> t_input_408;
const int c_relu_118_ich    = 2048;
const int c_relu_118_ih     = 7;
const int c_relu_118_iw     = 7;


typedef ap_uint<8> t_gemm_494;
typedef ap_uint<8> t_globalaveragepool_119_acc;
const int c_globalaveragepool_119_ich    = 2048;
const int c_globalaveragepool_119_och    = 2048;
const int c_globalaveragepool_119_ih     = 7;
const int c_globalaveragepool_119_iw     = 7;
const int c_globalaveragepool_119_oh     = 1;
const int c_globalaveragepool_119_ow     = 1;
const int c_globalaveragepool_119_fh     = 1;
const int c_globalaveragepool_119_fw     = 1;
const int c_globalaveragepool_119_stride = 1;
const int c_globalaveragepool_119_pad    = 0;
const int c_globalaveragepool_119_pool   = 0;


typedef uint8_t t_fc_weight_st;
typedef ap_uint<8> t_fc_weight;
const int c_fc_weight_och = 1000;
const int c_fc_weight_ich = 2048;
const int c_fc_weight_ih  = 1;
const int c_fc_weight_iw  = 1;

typedef ap_uint<8> t_output;
typedef ap_uint<32> t_gemm_121_acc;
const int c_gemm_121_ich    = 2048;
const int c_gemm_121_och    = 1000;
const int c_gemm_121_ih     = 1;
const int c_gemm_121_iw     = 1;
const int c_gemm_121_ow     = 1;
const int c_gemm_121_oh     = 1;
const int c_gemm_121_fw     = 1;
const int c_gemm_121_fh     = 1;
const int c_gemm_121_relu   = 0;
const int c_output_split  = 0;
const int c_gemm_121_stride = 1;
const int c_gemm_121_pad    = 0;
const int c_gemm_121_split  = 2;

void Network(
	hls::stream<t_i_data> &i_data,
	hls::stream<t_o_data> &o_data
);
#endif