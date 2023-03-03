#ifndef __NETWORK__
#define __NETWORK__
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"
#include "hls_vector.h"
#include <stdint.h>
#define c_i_data 64
typedef ap_axiu<c_i_data, 0, 0, 0> t_i_data;
#define c_o_data 32
typedef ap_axis<c_o_data, 0, 0, 0> t_o_data;
typedef ap_uint<1> t_last;
typedef uint8_t t_x_4;
typedef struct {
	t_x_4 data;
	bool last;
} t_x_4_struct;

const int c_x_4_ich    = 3;
const int c_x_4_ih     = 32;
const int c_x_4_iw     = 32;


const int c_266_och = 10;
const int c_266_oh  = 1;
const int c_266_ow  = 1;


const int c_conv_5_ops  = 1;

typedef hls::vector<int8_t, c_conv_5_ops> t_53_st;
typedef hls::vector<int8_t, c_conv_5_ops> t_53;
const int c_53_och = 16;
const int c_53_ich = 3;
const int c_53_ih  = 3;
const int c_53_iw  = 3;
const int c_53_ops = 1;
const int c_53_index = 9;
const int c_53_iter  = 49;
const float c_53_scale = 0.007812;
const int c_53_scale_shift = -7;

typedef uint8_t t_55;
typedef struct {
	t_55 data;
	bool last;
} t_55_struct;
typedef ap_int<32> t_conv_5_acc;
const int c_conv_5_ich    = 3;
const int c_conv_5_och    = 16;
const int c_conv_5_ih     = 32;
const int c_conv_5_iw     = 32;
const int c_conv_5_ow     = 32;
const int c_conv_5_oh     = 32;
const int c_conv_5_fw     = 3;
const int c_conv_5_fh     = 3;
const int c_conv_5_relu   = 1;
const int c_55_a_split  = 0;
const int c_conv_5_stride = 1;
const int c_conv_5_pad    = 1;
const int c_conv_5_split  = 2;


const int c_relu_6_ich    = 16;
const int c_relu_6_ih     = 32;
const int c_relu_6_iw     = 32;

typedef uint8_t t_59_skip;


const int c_conv_16_ops  = 2;

typedef hls::vector<int8_t, c_conv_16_ops> t_64_st;
typedef hls::vector<int8_t, c_conv_16_ops> t_64;
const int c_64_och = 16;
const int c_64_ich = 16;
const int c_64_ih  = 3;
const int c_64_iw  = 3;
const int c_64_ops = 2;
const int c_64_index = 9;
const int c_64_iter  = 129;
const float c_64_scale = 0.007812;
const int c_64_scale_shift = -7;

typedef uint8_t t_66;
typedef struct {
	t_66 data;
	bool last;
} t_66_struct;
typedef ap_int<32> t_conv_16_acc;
const int c_conv_16_ich    = 16;
const int c_conv_16_och    = 16;
const int c_conv_16_ih     = 32;
const int c_conv_16_iw     = 32;
const int c_conv_16_ow     = 32;
const int c_conv_16_oh     = 32;
const int c_conv_16_fw     = 3;
const int c_conv_16_fh     = 3;
const int c_conv_16_relu   = 1;
const int c_66_a_split  = 0;
const int c_conv_16_stride = 1;
const int c_conv_16_pad    = 1;
const float c_conv_16_scale = 0.031250;
const int c_conv_16_scale_shift = -5;
const int c_conv_16_split  = 2;


const int c_relu_17_ich    = 16;
const int c_relu_17_ih     = 32;
const int c_relu_17_iw     = 32;


const int c_conv_27_ops  = 4;

typedef hls::vector<int8_t, c_conv_27_ops> t_75_st;
typedef hls::vector<int8_t, c_conv_27_ops> t_75;
const int c_75_och = 16;
const int c_75_ich = 16;
const int c_75_ih  = 3;
const int c_75_iw  = 3;
const int c_75_ops = 4;
const int c_75_index = 9;
const int c_75_iter  = 65;
const float c_75_scale = 0.007812;
const int c_75_scale_shift = -7;

typedef uint8_t t_78;
typedef struct {
	t_78 data;
	bool last;
} t_78_struct;
typedef ap_int<32> t_conv_27_acc;
const int c_conv_27_ich    = 16;
const int c_conv_27_och    = 16;
const int c_conv_27_ih     = 32;
const int c_conv_27_iw     = 32;
const int c_conv_27_ow     = 32;
const int c_conv_27_oh     = 32;
const int c_conv_27_fw     = 3;
const int c_conv_27_fh     = 3;
const int c_conv_27_relu   = 1;
const int c_78_a_split  = 0;
const int c_conv_27_stride = 1;
const int c_conv_27_pad    = 1;
const int c_conv_27_split  = 2;


typedef uint8_t t_77;
const int c_add_28_ich    = 16;
const int c_add_28_ih     = 32;
const int c_add_28_iw     = 32;


const int c_relu_29_ich    = 16;
const int c_relu_29_ih     = 32;
const int c_relu_29_iw     = 32;

typedef uint8_t t_79_skip;


const int c_conv_36_ops  = 2;

typedef hls::vector<int8_t, c_conv_36_ops> t_84_st;
typedef hls::vector<int8_t, c_conv_36_ops> t_84;
const int c_84_och = 16;
const int c_84_ich = 16;
const int c_84_ih  = 3;
const int c_84_iw  = 3;
const int c_84_ops = 2;
const int c_84_index = 9;
const int c_84_iter  = 129;
const float c_84_scale = 0.007812;
const int c_84_scale_shift = -7;

typedef uint8_t t_86;
typedef struct {
	t_86 data;
	bool last;
} t_86_struct;
typedef ap_int<32> t_conv_36_acc;
const int c_conv_36_ich    = 16;
const int c_conv_36_och    = 16;
const int c_conv_36_ih     = 32;
const int c_conv_36_iw     = 32;
const int c_conv_36_ow     = 32;
const int c_conv_36_oh     = 32;
const int c_conv_36_fw     = 3;
const int c_conv_36_fh     = 3;
const int c_conv_36_relu   = 1;
const int c_86_a_split  = 0;
const int c_conv_36_stride = 1;
const int c_conv_36_pad    = 1;
const float c_conv_36_scale = 0.031250;
const int c_conv_36_scale_shift = -5;
const int c_conv_36_split  = 2;


const int c_relu_37_ich    = 16;
const int c_relu_37_ih     = 32;
const int c_relu_37_iw     = 32;


const int c_conv_47_ops  = 2;

typedef hls::vector<int8_t, c_conv_47_ops> t_95_st;
typedef hls::vector<int8_t, c_conv_47_ops> t_95;
const int c_95_och = 16;
const int c_95_ich = 16;
const int c_95_ih  = 3;
const int c_95_iw  = 3;
const int c_95_ops = 2;
const int c_95_index = 9;
const int c_95_iter  = 129;
const float c_95_scale = 0.007812;
const int c_95_scale_shift = -7;

typedef uint8_t t_98;
typedef struct {
	t_98 data;
	bool last;
} t_98_struct;
typedef ap_int<32> t_conv_47_acc;
const int c_conv_47_ich    = 16;
const int c_conv_47_och    = 16;
const int c_conv_47_ih     = 32;
const int c_conv_47_iw     = 32;
const int c_conv_47_ow     = 32;
const int c_conv_47_oh     = 32;
const int c_conv_47_fw     = 3;
const int c_conv_47_fh     = 3;
const int c_conv_47_relu   = 1;
const int c_98_a_split  = 0;
const int c_conv_47_stride = 1;
const int c_conv_47_pad    = 1;
const int c_conv_47_split  = 2;


typedef uint8_t t_97;
const int c_add_48_ich    = 16;
const int c_add_48_ih     = 32;
const int c_add_48_iw     = 32;


const int c_relu_49_ich    = 16;
const int c_relu_49_ih     = 32;
const int c_relu_49_iw     = 32;

typedef uint8_t t_99_skip;


const int c_conv_56_ops  = 2;

typedef hls::vector<int8_t, c_conv_56_ops> t_104_st;
typedef hls::vector<int8_t, c_conv_56_ops> t_104;
const int c_104_och = 16;
const int c_104_ich = 16;
const int c_104_ih  = 3;
const int c_104_iw  = 3;
const int c_104_ops = 2;
const int c_104_index = 9;
const int c_104_iter  = 129;
const float c_104_scale = 0.007812;
const int c_104_scale_shift = -7;

typedef uint8_t t_106;
typedef struct {
	t_106 data;
	bool last;
} t_106_struct;
typedef ap_int<32> t_conv_56_acc;
const int c_conv_56_ich    = 16;
const int c_conv_56_och    = 16;
const int c_conv_56_ih     = 32;
const int c_conv_56_iw     = 32;
const int c_conv_56_ow     = 32;
const int c_conv_56_oh     = 32;
const int c_conv_56_fw     = 3;
const int c_conv_56_fh     = 3;
const int c_conv_56_relu   = 1;
const int c_106_a_split  = 0;
const int c_conv_56_stride = 1;
const int c_conv_56_pad    = 1;
const float c_conv_56_scale = 0.031250;
const int c_conv_56_scale_shift = -5;
const int c_conv_56_split  = 2;


const int c_relu_57_ich    = 16;
const int c_relu_57_ih     = 32;
const int c_relu_57_iw     = 32;


const int c_conv_67_ops  = 2;

typedef hls::vector<int8_t, c_conv_67_ops> t_115_st;
typedef hls::vector<int8_t, c_conv_67_ops> t_115;
const int c_115_och = 16;
const int c_115_ich = 16;
const int c_115_ih  = 3;
const int c_115_iw  = 3;
const int c_115_ops = 2;
const int c_115_index = 9;
const int c_115_iter  = 129;
const float c_115_scale = 0.007812;
const int c_115_scale_shift = -7;

typedef uint8_t t_118;
typedef struct {
	t_118 data;
	bool last;
} t_118_struct;
typedef ap_int<32> t_conv_67_acc;
const int c_conv_67_ich    = 16;
const int c_conv_67_och    = 16;
const int c_conv_67_ih     = 32;
const int c_conv_67_iw     = 32;
const int c_conv_67_ow     = 32;
const int c_conv_67_oh     = 32;
const int c_conv_67_fw     = 3;
const int c_conv_67_fh     = 3;
const int c_conv_67_relu   = 1;
const int c_118_a_split  = 0;
const int c_conv_67_stride = 1;
const int c_conv_67_pad    = 1;
const int c_conv_67_split  = 2;


typedef uint8_t t_117;
const int c_add_68_ich    = 16;
const int c_add_68_ih     = 32;
const int c_add_68_iw     = 32;


const int c_relu_69_ich    = 16;
const int c_relu_69_ih     = 32;
const int c_relu_69_iw     = 32;

typedef uint8_t t_119_skip;


const int c_conv_76_ops  = 1;

typedef hls::vector<int8_t, c_conv_76_ops> t_124_st;
typedef hls::vector<int8_t, c_conv_76_ops> t_124;
const int c_124_och = 32;
const int c_124_ich = 16;
const int c_124_ih  = 3;
const int c_124_iw  = 3;
const int c_124_ops = 1;
const int c_124_index = 9;
const int c_124_iter  = 513;
const float c_124_scale = 0.007812;
const int c_124_scale_shift = -7;

typedef uint8_t t_126;
typedef struct {
	t_126 data;
	bool last;
} t_126_struct;
typedef ap_int<32> t_conv_76_acc;
const int c_conv_76_ich    = 16;
const int c_conv_76_och    = 32;
const int c_conv_76_ih     = 32;
const int c_conv_76_iw     = 32;
const int c_conv_76_ow     = 16;
const int c_conv_76_oh     = 16;
const int c_conv_76_fw     = 3;
const int c_conv_76_fh     = 3;
const int c_conv_76_relu   = 1;
const int c_126_a_split  = 0;
const int c_conv_76_stride = 2;
const int c_conv_76_pad    = 1;
const float c_conv_76_scale = 0.031250;
const int c_conv_76_scale_shift = -5;
const int c_conv_76_split  = 2;


const int c_conv_93_ops  = 1;

typedef hls::vector<int8_t, c_conv_93_ops> t_141_st;
typedef hls::vector<int8_t, c_conv_93_ops> t_141;
const int c_141_och = 32;
const int c_141_ich = 16;
const int c_141_ih  = 1;
const int c_141_iw  = 1;
const int c_141_ops = 1;
const int c_141_index = 1;
const int c_141_iter  = 513;
const float c_141_scale = 0.007812;
const int c_141_scale_shift = -7;

typedef uint8_t t_142;
typedef struct {
	t_142 data;
	bool last;
} t_142_struct;
typedef ap_int<32> t_conv_93_acc;
const int c_conv_93_ich    = 16;
const int c_conv_93_och    = 32;
const int c_conv_93_ih     = 32;
const int c_conv_93_iw     = 32;
const int c_conv_93_ow     = 16;
const int c_conv_93_oh     = 16;
const int c_conv_93_fw     = 1;
const int c_conv_93_fh     = 1;
const int c_conv_93_relu   = 0;
const int c_142_a_split  = 0;
const int c_conv_93_stride = 2;
const int c_conv_93_pad    = 0;
const float c_conv_93_scale = 0.031250;
const int c_conv_93_scale_shift = -5;
const int c_conv_93_split  = 2;


const int c_relu_77_ich    = 32;
const int c_relu_77_ih     = 16;
const int c_relu_77_iw     = 16;


const int c_conv_87_ops  = 2;

typedef hls::vector<int8_t, c_conv_87_ops> t_135_st;
typedef hls::vector<int8_t, c_conv_87_ops> t_135;
const int c_135_och = 32;
const int c_135_ich = 32;
const int c_135_ih  = 3;
const int c_135_iw  = 3;
const int c_135_ops = 2;
const int c_135_index = 9;
const int c_135_iter  = 513;
const float c_135_scale = 0.007812;
const int c_135_scale_shift = -7;

typedef uint8_t t_144;
typedef struct {
	t_144 data;
	bool last;
} t_144_struct;
typedef ap_int<32> t_conv_87_acc;
const int c_conv_87_ich    = 32;
const int c_conv_87_och    = 32;
const int c_conv_87_ih     = 16;
const int c_conv_87_iw     = 16;
const int c_conv_87_ow     = 16;
const int c_conv_87_oh     = 16;
const int c_conv_87_fw     = 3;
const int c_conv_87_fh     = 3;
const int c_conv_87_relu   = 1;
const int c_144_a_split  = 0;
const int c_conv_87_stride = 1;
const int c_conv_87_pad    = 1;
const int c_conv_87_split  = 2;


typedef uint8_t t_143;
const int c_add_94_ich    = 32;
const int c_add_94_ih     = 16;
const int c_add_94_iw     = 16;


const int c_relu_95_ich    = 32;
const int c_relu_95_ih     = 16;
const int c_relu_95_iw     = 16;

typedef uint8_t t_145_skip;


const int c_conv_102_ops  = 2;

typedef hls::vector<int8_t, c_conv_102_ops> t_150_st;
typedef hls::vector<int8_t, c_conv_102_ops> t_150;
const int c_150_och = 32;
const int c_150_ich = 32;
const int c_150_ih  = 3;
const int c_150_iw  = 3;
const int c_150_ops = 2;
const int c_150_index = 9;
const int c_150_iter  = 513;
const float c_150_scale = 0.007812;
const int c_150_scale_shift = -7;

typedef uint8_t t_152;
typedef struct {
	t_152 data;
	bool last;
} t_152_struct;
typedef ap_int<32> t_conv_102_acc;
const int c_conv_102_ich    = 32;
const int c_conv_102_och    = 32;
const int c_conv_102_ih     = 16;
const int c_conv_102_iw     = 16;
const int c_conv_102_ow     = 16;
const int c_conv_102_oh     = 16;
const int c_conv_102_fw     = 3;
const int c_conv_102_fh     = 3;
const int c_conv_102_relu   = 1;
const int c_152_a_split  = 0;
const int c_conv_102_stride = 1;
const int c_conv_102_pad    = 1;
const float c_conv_102_scale = 0.031250;
const int c_conv_102_scale_shift = -5;
const int c_conv_102_split  = 2;


const int c_relu_103_ich    = 32;
const int c_relu_103_ih     = 16;
const int c_relu_103_iw     = 16;


const int c_conv_113_ops  = 2;

typedef hls::vector<int8_t, c_conv_113_ops> t_161_st;
typedef hls::vector<int8_t, c_conv_113_ops> t_161;
const int c_161_och = 32;
const int c_161_ich = 32;
const int c_161_ih  = 3;
const int c_161_iw  = 3;
const int c_161_ops = 2;
const int c_161_index = 9;
const int c_161_iter  = 513;
const float c_161_scale = 0.007812;
const int c_161_scale_shift = -7;

typedef uint8_t t_164;
typedef struct {
	t_164 data;
	bool last;
} t_164_struct;
typedef ap_int<32> t_conv_113_acc;
const int c_conv_113_ich    = 32;
const int c_conv_113_och    = 32;
const int c_conv_113_ih     = 16;
const int c_conv_113_iw     = 16;
const int c_conv_113_ow     = 16;
const int c_conv_113_oh     = 16;
const int c_conv_113_fw     = 3;
const int c_conv_113_fh     = 3;
const int c_conv_113_relu   = 1;
const int c_164_a_split  = 0;
const int c_conv_113_stride = 1;
const int c_conv_113_pad    = 1;
const int c_conv_113_split  = 2;


typedef uint8_t t_163;
const int c_add_114_ich    = 32;
const int c_add_114_ih     = 16;
const int c_add_114_iw     = 16;


const int c_relu_115_ich    = 32;
const int c_relu_115_ih     = 16;
const int c_relu_115_iw     = 16;

typedef uint8_t t_165_skip;


const int c_conv_122_ops  = 2;

typedef hls::vector<int8_t, c_conv_122_ops> t_170_st;
typedef hls::vector<int8_t, c_conv_122_ops> t_170;
const int c_170_och = 32;
const int c_170_ich = 32;
const int c_170_ih  = 3;
const int c_170_iw  = 3;
const int c_170_ops = 2;
const int c_170_index = 9;
const int c_170_iter  = 513;
const float c_170_scale = 0.007812;
const int c_170_scale_shift = -7;

typedef uint8_t t_172;
typedef struct {
	t_172 data;
	bool last;
} t_172_struct;
typedef ap_int<32> t_conv_122_acc;
const int c_conv_122_ich    = 32;
const int c_conv_122_och    = 32;
const int c_conv_122_ih     = 16;
const int c_conv_122_iw     = 16;
const int c_conv_122_ow     = 16;
const int c_conv_122_oh     = 16;
const int c_conv_122_fw     = 3;
const int c_conv_122_fh     = 3;
const int c_conv_122_relu   = 1;
const int c_172_a_split  = 0;
const int c_conv_122_stride = 1;
const int c_conv_122_pad    = 1;
const float c_conv_122_scale = 0.031250;
const int c_conv_122_scale_shift = -5;
const int c_conv_122_split  = 2;


const int c_relu_123_ich    = 32;
const int c_relu_123_ih     = 16;
const int c_relu_123_iw     = 16;


const int c_conv_133_ops  = 2;

typedef hls::vector<int8_t, c_conv_133_ops> t_181_st;
typedef hls::vector<int8_t, c_conv_133_ops> t_181;
const int c_181_och = 32;
const int c_181_ich = 32;
const int c_181_ih  = 3;
const int c_181_iw  = 3;
const int c_181_ops = 2;
const int c_181_index = 9;
const int c_181_iter  = 513;
const float c_181_scale = 0.007812;
const int c_181_scale_shift = -7;

typedef uint8_t t_184;
typedef struct {
	t_184 data;
	bool last;
} t_184_struct;
typedef ap_int<32> t_conv_133_acc;
const int c_conv_133_ich    = 32;
const int c_conv_133_och    = 32;
const int c_conv_133_ih     = 16;
const int c_conv_133_iw     = 16;
const int c_conv_133_ow     = 16;
const int c_conv_133_oh     = 16;
const int c_conv_133_fw     = 3;
const int c_conv_133_fh     = 3;
const int c_conv_133_relu   = 1;
const int c_184_a_split  = 0;
const int c_conv_133_stride = 1;
const int c_conv_133_pad    = 1;
const int c_conv_133_split  = 2;


typedef uint8_t t_183;
const int c_add_134_ich    = 32;
const int c_add_134_ih     = 16;
const int c_add_134_iw     = 16;


const int c_relu_135_ich    = 32;
const int c_relu_135_ih     = 16;
const int c_relu_135_iw     = 16;

typedef uint8_t t_185_skip;


const int c_conv_142_ops  = 1;

typedef hls::vector<int8_t, c_conv_142_ops> t_190_st;
typedef hls::vector<int8_t, c_conv_142_ops> t_190;
const int c_190_och = 64;
const int c_190_ich = 32;
const int c_190_ih  = 3;
const int c_190_iw  = 3;
const int c_190_ops = 1;
const int c_190_index = 9;
const int c_190_iter  = 2049;
const float c_190_scale = 0.007812;
const int c_190_scale_shift = -7;

typedef uint8_t t_192;
typedef struct {
	t_192 data;
	bool last;
} t_192_struct;
typedef ap_int<32> t_conv_142_acc;
const int c_conv_142_ich    = 32;
const int c_conv_142_och    = 64;
const int c_conv_142_ih     = 16;
const int c_conv_142_iw     = 16;
const int c_conv_142_ow     = 8;
const int c_conv_142_oh     = 8;
const int c_conv_142_fw     = 3;
const int c_conv_142_fh     = 3;
const int c_conv_142_relu   = 1;
const int c_192_a_split  = 0;
const int c_conv_142_stride = 2;
const int c_conv_142_pad    = 1;
const float c_conv_142_scale = 0.031250;
const int c_conv_142_scale_shift = -5;
const int c_conv_142_split  = 2;


const int c_conv_159_ops  = 1;

typedef hls::vector<int8_t, c_conv_159_ops> t_207_st;
typedef hls::vector<int8_t, c_conv_159_ops> t_207;
const int c_207_och = 64;
const int c_207_ich = 32;
const int c_207_ih  = 1;
const int c_207_iw  = 1;
const int c_207_ops = 1;
const int c_207_index = 1;
const int c_207_iter  = 2049;
const float c_207_scale = 0.007812;
const int c_207_scale_shift = -7;

typedef uint8_t t_208;
typedef struct {
	t_208 data;
	bool last;
} t_208_struct;
typedef ap_int<32> t_conv_159_acc;
const int c_conv_159_ich    = 32;
const int c_conv_159_och    = 64;
const int c_conv_159_ih     = 16;
const int c_conv_159_iw     = 16;
const int c_conv_159_ow     = 8;
const int c_conv_159_oh     = 8;
const int c_conv_159_fw     = 1;
const int c_conv_159_fh     = 1;
const int c_conv_159_relu   = 0;
const int c_208_a_split  = 0;
const int c_conv_159_stride = 2;
const int c_conv_159_pad    = 0;
const float c_conv_159_scale = 0.031250;
const int c_conv_159_scale_shift = -5;
const int c_conv_159_split  = 2;


const int c_relu_143_ich    = 64;
const int c_relu_143_ih     = 8;
const int c_relu_143_iw     = 8;


const int c_conv_153_ops  = 2;

typedef hls::vector<int8_t, c_conv_153_ops> t_201_st;
typedef hls::vector<int8_t, c_conv_153_ops> t_201;
const int c_201_och = 64;
const int c_201_ich = 64;
const int c_201_ih  = 3;
const int c_201_iw  = 3;
const int c_201_ops = 2;
const int c_201_index = 9;
const int c_201_iter  = 2049;
const float c_201_scale = 0.007812;
const int c_201_scale_shift = -7;

typedef uint8_t t_210;
typedef struct {
	t_210 data;
	bool last;
} t_210_struct;
typedef ap_int<32> t_conv_153_acc;
const int c_conv_153_ich    = 64;
const int c_conv_153_och    = 64;
const int c_conv_153_ih     = 8;
const int c_conv_153_iw     = 8;
const int c_conv_153_ow     = 8;
const int c_conv_153_oh     = 8;
const int c_conv_153_fw     = 3;
const int c_conv_153_fh     = 3;
const int c_conv_153_relu   = 1;
const int c_210_a_split  = 0;
const int c_conv_153_stride = 1;
const int c_conv_153_pad    = 1;
const int c_conv_153_split  = 2;


typedef uint8_t t_209;
const int c_add_160_ich    = 64;
const int c_add_160_ih     = 8;
const int c_add_160_iw     = 8;


const int c_relu_161_ich    = 64;
const int c_relu_161_ih     = 8;
const int c_relu_161_iw     = 8;

typedef uint8_t t_211_skip;


const int c_conv_168_ops  = 2;

typedef hls::vector<int8_t, c_conv_168_ops> t_216_st;
typedef hls::vector<int8_t, c_conv_168_ops> t_216;
const int c_216_och = 64;
const int c_216_ich = 64;
const int c_216_ih  = 3;
const int c_216_iw  = 3;
const int c_216_ops = 2;
const int c_216_index = 9;
const int c_216_iter  = 2049;
const float c_216_scale = 0.007812;
const int c_216_scale_shift = -7;

typedef uint8_t t_218;
typedef struct {
	t_218 data;
	bool last;
} t_218_struct;
typedef ap_int<32> t_conv_168_acc;
const int c_conv_168_ich    = 64;
const int c_conv_168_och    = 64;
const int c_conv_168_ih     = 8;
const int c_conv_168_iw     = 8;
const int c_conv_168_ow     = 8;
const int c_conv_168_oh     = 8;
const int c_conv_168_fw     = 3;
const int c_conv_168_fh     = 3;
const int c_conv_168_relu   = 1;
const int c_218_a_split  = 0;
const int c_conv_168_stride = 1;
const int c_conv_168_pad    = 1;
const float c_conv_168_scale = 0.031250;
const int c_conv_168_scale_shift = -5;
const int c_conv_168_split  = 2;


const int c_relu_169_ich    = 64;
const int c_relu_169_ih     = 8;
const int c_relu_169_iw     = 8;


const int c_conv_179_ops  = 2;

typedef hls::vector<int8_t, c_conv_179_ops> t_227_st;
typedef hls::vector<int8_t, c_conv_179_ops> t_227;
const int c_227_och = 64;
const int c_227_ich = 64;
const int c_227_ih  = 3;
const int c_227_iw  = 3;
const int c_227_ops = 2;
const int c_227_index = 9;
const int c_227_iter  = 2049;
const float c_227_scale = 0.007812;
const int c_227_scale_shift = -7;

typedef uint8_t t_230;
typedef struct {
	t_230 data;
	bool last;
} t_230_struct;
typedef ap_int<32> t_conv_179_acc;
const int c_conv_179_ich    = 64;
const int c_conv_179_och    = 64;
const int c_conv_179_ih     = 8;
const int c_conv_179_iw     = 8;
const int c_conv_179_ow     = 8;
const int c_conv_179_oh     = 8;
const int c_conv_179_fw     = 3;
const int c_conv_179_fh     = 3;
const int c_conv_179_relu   = 1;
const int c_230_a_split  = 0;
const int c_conv_179_stride = 1;
const int c_conv_179_pad    = 1;
const int c_conv_179_split  = 2;


typedef uint8_t t_229;
const int c_add_180_ich    = 64;
const int c_add_180_ih     = 8;
const int c_add_180_iw     = 8;


const int c_relu_181_ich    = 64;
const int c_relu_181_ih     = 8;
const int c_relu_181_iw     = 8;

typedef uint8_t t_231_skip;


const int c_conv_188_ops  = 2;

typedef hls::vector<int8_t, c_conv_188_ops> t_236_st;
typedef hls::vector<int8_t, c_conv_188_ops> t_236;
const int c_236_och = 64;
const int c_236_ich = 64;
const int c_236_ih  = 3;
const int c_236_iw  = 3;
const int c_236_ops = 2;
const int c_236_index = 9;
const int c_236_iter  = 2049;
const float c_236_scale = 0.007812;
const int c_236_scale_shift = -7;

typedef uint8_t t_238;
typedef struct {
	t_238 data;
	bool last;
} t_238_struct;
typedef ap_int<32> t_conv_188_acc;
const int c_conv_188_ich    = 64;
const int c_conv_188_och    = 64;
const int c_conv_188_ih     = 8;
const int c_conv_188_iw     = 8;
const int c_conv_188_ow     = 8;
const int c_conv_188_oh     = 8;
const int c_conv_188_fw     = 3;
const int c_conv_188_fh     = 3;
const int c_conv_188_relu   = 1;
const int c_238_a_split  = 0;
const int c_conv_188_stride = 1;
const int c_conv_188_pad    = 1;
const float c_conv_188_scale = 0.031250;
const int c_conv_188_scale_shift = -5;
const int c_conv_188_split  = 2;


const int c_relu_189_ich    = 64;
const int c_relu_189_ih     = 8;
const int c_relu_189_iw     = 8;


const int c_conv_199_ops  = 2;

typedef hls::vector<int8_t, c_conv_199_ops> t_247_st;
typedef hls::vector<int8_t, c_conv_199_ops> t_247;
const int c_247_och = 64;
const int c_247_ich = 64;
const int c_247_ih  = 3;
const int c_247_iw  = 3;
const int c_247_ops = 2;
const int c_247_index = 9;
const int c_247_iter  = 2049;
const float c_247_scale = 0.007812;
const int c_247_scale_shift = -7;

typedef uint8_t t_250;
typedef struct {
	t_250 data;
	bool last;
} t_250_struct;
typedef ap_int<32> t_conv_199_acc;
const int c_conv_199_ich    = 64;
const int c_conv_199_och    = 64;
const int c_conv_199_ih     = 8;
const int c_conv_199_iw     = 8;
const int c_conv_199_ow     = 8;
const int c_conv_199_oh     = 8;
const int c_conv_199_fw     = 3;
const int c_conv_199_fh     = 3;
const int c_conv_199_relu   = 1;
const int c_250_a_split  = 0;
const int c_conv_199_stride = 1;
const int c_conv_199_pad    = 1;
const int c_conv_199_split  = 2;


typedef uint8_t t_249;
const int c_add_200_ich    = 64;
const int c_add_200_ih     = 8;
const int c_add_200_iw     = 8;


const int c_relu_201_ich    = 64;
const int c_relu_201_ih     = 8;
const int c_relu_201_iw     = 8;


typedef uint8_t t_252;
typedef int32_t t_maxpool_203_acc;
typedef struct {
	t_252 data;
	bool last;
} t_252_struct;
const int c_maxpool_203_ich    = 64;
const int c_maxpool_203_och    = 64;
const int c_maxpool_203_ih     = 8;
const int c_maxpool_203_iw     = 8;
const int c_maxpool_203_oh     = 1;
const int c_maxpool_203_ow     = 1;
const int c_maxpool_203_fh     = 8;
const int c_maxpool_203_fw     = 8;
const int c_maxpool_203_stride = 1;
const int c_maxpool_203_pad    = 0;
const int c_maxpool_203_pool   = 1;


const int c_conv_209_ops  = 1;

typedef hls::vector<int8_t, c_conv_209_ops> t_257_st;
typedef hls::vector<int8_t, c_conv_209_ops> t_257;
const int c_257_och = 10;
const int c_257_ich = 64;
const int c_257_ih  = 1;
const int c_257_iw  = 1;
const int c_257_ops = 1;
const int c_257_index = 1;
const int c_257_iter  = 641;
const float c_257_scale = 0.007812;
const int c_257_scale_shift = -7;

typedef uint8_t t_258;
typedef struct {
	t_258 data;
	bool last;
} t_258_struct;
typedef ap_int<32> t_conv_209_acc;
const int c_conv_209_ich    = 64;
const int c_conv_209_och    = 10;
const int c_conv_209_ih     = 1;
const int c_conv_209_iw     = 1;
const int c_conv_209_ow     = 1;
const int c_conv_209_oh     = 1;
const int c_conv_209_fw     = 1;
const int c_conv_209_fh     = 1;
const int c_conv_209_relu   = 0;
const int c_258_a_split  = 0;
const int c_conv_209_stride = 1;
const int c_conv_209_pad    = 0;
const int c_conv_209_split  = 2;

void Network(
	hls::stream<t_i_data> &i_data,
	hls::stream<t_o_data> &o_data
);
#endif