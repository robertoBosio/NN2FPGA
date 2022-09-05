#include "Network.hpp"
#include "hls_stream.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "PackedConv.hpp"
#include "ActivationStreams.hpp"
#include "AddStreams.hpp"
#include "PoolStreams.hpp"
#include "Utils.hpp"

void Network(
	t_i_data* i_data,
	t_weight* i_weight,
	t_o_data* o_data
) {

	#pragma HLS interface m_axi port=i_data depth=10 offset=slave bundle=gmem0
	#pragma HLS interface m_axi port=i_weight depth=10 offset=slave bundle=gmem1 max_read_burst_length=256
	#pragma HLS interface m_axi port=o_data depth=10 offset=slave
	#pragma HLS INTERFACE mode=ap_ctrl_chain port=return
	#pragma HLS DATAFLOW

	hls::stream<t_input> s_input;
	#pragma HLS STREAM variable=s_input depth=1 type=fifo
	ap_uint<1> s_input_last[1];

	ProduceStream<
		t_i_data, 
		t_input,
		c_input_ich,
		c_input_iw,
		c_input_ih
	>(
		i_data,
		s_input,
		s_input_last
	);

	hls::stream<t_input_4> s_input_4;
	#pragma HLS STREAM variable=s_input_4 depth=1 type=fifo
	ap_uint<1> s_input_4_last[1];

	hls::stream<t_conv_208> s_conv_208;
	#pragma HLS STREAM variable=s_conv_208 depth=1 type=fifo
	ap_uint<1> s_conv_208_last[1];

	ProduceStream<
		t_conv_208_st,
		t_conv_208,
		c_conv_208_ich,
		c_conv_208_ich,
		c_conv_208_iw,
		c_conv_208_ih
	>(
		c_conv_208_st,
		s_conv_208
	);


	PackedConvBuffAcc<
		t_input,
		t_conv_208,
		t_input_4,
		t_conv_0_acc,
		c_conv_0_ich,
		c_conv_0_och,
		c_conv_0_iw,
		c_conv_0_ih,
		c_conv_0_ow,
		c_conv_0_oh,
		c_conv_0_fw,
		c_conv_0_fh,
		c_conv_0_stride,
		c_conv_0_pad
	> (
		s_input,
		s_conv_208,
		s_input_4,
		s_input_last,
		s_input_4_last
	);

	hls::stream<t_conv_135> s_conv_135;
	#pragma HLS STREAM variable=s_conv_135 depth=1 type=fifo
	ap_uint<1> s_conv_135_last[1];

	ReluStreams<
		t_input_4,
		t_conv_135
	> (
		s_input_4,
		s_conv_135,
		s_input_4_last,
		s_conv_135_last
	);

	hls::stream<t_conv_135_skip> s_conv_135_skip;
	#pragma HLS STREAM variable=s_conv_135_skip depth=1 type=fifo
	ap_uint<1> s_conv_135_skip_last[1];

	hls::stream<t_input_12> s_input_12;
	#pragma HLS STREAM variable=s_input_12 depth=1 type=fifo
	ap_uint<1> s_input_12_last[1];

	hls::stream<t_conv_211> s_conv_211;
	#pragma HLS STREAM variable=s_conv_211 depth=1 type=fifo
	ap_uint<1> s_conv_211_last[1];

	ProduceStream<
		t_conv_211_st,
		t_conv_211,
		c_conv_211_ich,
		c_conv_211_ich,
		c_conv_211_iw,
		c_conv_211_ih
	>(
		c_conv_211_st,
		s_conv_211
	);


	PackedConvBuffAcc<
		t_conv_135,
		t_conv_211,
		t_input_12,
		t_conv_2_acc,
		c_conv_2_ich,
		c_conv_2_och,
		c_conv_2_iw,
		c_conv_2_ih,
		c_conv_2_ow,
		c_conv_2_oh,
		c_conv_2_fw,
		c_conv_2_fh,
		c_conv_2_stride,
		c_conv_2_pad
	> (
		s_conv_135,
		s_conv_211,
		s_input_12,
		s_conv_135_skip,
		s_conv_135_last,
		s_conv_135_skip_last,
		s_input_12_last
	);

	hls::stream<t_conv_138> s_conv_138;
	#pragma HLS STREAM variable=s_conv_138 depth=1 type=fifo
	ap_uint<1> s_conv_138_last[1];

	ReluStreams<
		t_input_12,
		t_conv_138
	> (
		s_input_12,
		s_conv_138,
		s_input_12_last,
		s_conv_138_last
	);

	hls::stream<t_add_213> s_add_213;
	#pragma HLS STREAM variable=s_add_213 depth=1 type=fifo
	ap_uint<1> s_add_213_last[1];

	hls::stream<t_conv_214> s_conv_214;
	#pragma HLS STREAM variable=s_conv_214 depth=1 type=fifo
	ap_uint<1> s_conv_214_last[1];

	ProduceStream<
		t_conv_214_st,
		t_conv_214,
		c_conv_214_ich,
		c_conv_214_ich,
		c_conv_214_iw,
		c_conv_214_ih
	>(
		c_conv_214_st,
		s_conv_214
	);


	PackedConvBuffAcc<
		t_conv_138,
		t_conv_214,
		t_add_213,
		t_conv_4_acc,
		c_conv_4_ich,
		c_conv_4_och,
		c_conv_4_iw,
		c_conv_4_ih,
		c_conv_4_ow,
		c_conv_4_oh,
		c_conv_4_fw,
		c_conv_4_fh,
		c_conv_4_stride,
		c_conv_4_pad
	> (
		s_conv_138,
		s_conv_214,
		s_add_213,
		s_conv_138_last,
		s_add_213_last
	);

	hls::stream<t_relu_141> s_relu_141;
	#pragma HLS STREAM variable=s_relu_141 depth=1 type=fifo
	ap_uint<1> s_relu_141_last[1];

	AddStreams<
		t_add_213,
		t_relu_141
	> (
		s_add_213,
		s_conv_135_skip,
		s_relu_141,
		s_conv_135_skip_last,
		s_relu_141_last
	);

	hls::stream<t_input_20> s_input_20;
	#pragma HLS STREAM variable=s_input_20 depth=1 type=fifo
	ap_uint<1> s_input_20_last[1];

	ReluStreams<
		t_relu_141,
		t_input_20
	> (
		s_relu_141,
		s_input_20,
		s_relu_141_last,
		s_input_20_last
	);

	hls::stream<t_input_20_skip> s_input_20_skip;
	#pragma HLS STREAM variable=s_input_20_skip depth=1 type=fifo
	ap_uint<1> s_input_20_skip_last[1];

	hls::stream<t_input_28> s_input_28;
	#pragma HLS STREAM variable=s_input_28 depth=1 type=fifo
	ap_uint<1> s_input_28_last[1];

	hls::stream<t_conv_217> s_conv_217;
	#pragma HLS STREAM variable=s_conv_217 depth=1 type=fifo
	ap_uint<1> s_conv_217_last[1];

	ProduceStream<
		t_conv_217_st,
		t_conv_217,
		c_conv_217_ich,
		c_conv_217_ich,
		c_conv_217_iw,
		c_conv_217_ih
	>(
		c_conv_217_st,
		s_conv_217
	);


	PackedConvBuffAcc<
		t_input_20,
		t_conv_217,
		t_input_28,
		t_conv_7_acc,
		c_conv_7_ich,
		c_conv_7_och,
		c_conv_7_iw,
		c_conv_7_ih,
		c_conv_7_ow,
		c_conv_7_oh,
		c_conv_7_fw,
		c_conv_7_fh,
		c_conv_7_stride,
		c_conv_7_pad
	> (
		s_input_20,
		s_conv_217,
		s_input_28,
		s_input_20_skip,
		s_input_20_last,
		s_input_20_skip_last,
		s_input_28_last
	);

	hls::stream<t_conv_145> s_conv_145;
	#pragma HLS STREAM variable=s_conv_145 depth=1 type=fifo
	ap_uint<1> s_conv_145_last[1];

	ReluStreams<
		t_input_28,
		t_conv_145
	> (
		s_input_28,
		s_conv_145,
		s_input_28_last,
		s_conv_145_last
	);

	hls::stream<t_add_219> s_add_219;
	#pragma HLS STREAM variable=s_add_219 depth=1 type=fifo
	ap_uint<1> s_add_219_last[1];

	hls::stream<t_conv_220> s_conv_220;
	#pragma HLS STREAM variable=s_conv_220 depth=1 type=fifo
	ap_uint<1> s_conv_220_last[1];

	ProduceStream<
		t_conv_220_st,
		t_conv_220,
		c_conv_220_ich,
		c_conv_220_ich,
		c_conv_220_iw,
		c_conv_220_ih
	>(
		c_conv_220_st,
		s_conv_220
	);


	PackedConvBuffAcc<
		t_conv_145,
		t_conv_220,
		t_add_219,
		t_conv_9_acc,
		c_conv_9_ich,
		c_conv_9_och,
		c_conv_9_iw,
		c_conv_9_ih,
		c_conv_9_ow,
		c_conv_9_oh,
		c_conv_9_fw,
		c_conv_9_fh,
		c_conv_9_stride,
		c_conv_9_pad
	> (
		s_conv_145,
		s_conv_220,
		s_add_219,
		s_conv_145_last,
		s_add_219_last
	);

	hls::stream<t_relu_148> s_relu_148;
	#pragma HLS STREAM variable=s_relu_148 depth=1 type=fifo
	ap_uint<1> s_relu_148_last[1];

	AddStreams<
		t_add_219,
		t_relu_148
	> (
		s_add_219,
		s_input_20_skip,
		s_relu_148,
		s_input_20_skip_last,
		s_relu_148_last
	);

	hls::stream<t_input_36> s_input_36;
	#pragma HLS STREAM variable=s_input_36 depth=1 type=fifo
	ap_uint<1> s_input_36_last[1];

	ReluStreams<
		t_relu_148,
		t_input_36
	> (
		s_relu_148,
		s_input_36,
		s_relu_148_last,
		s_input_36_last
	);

	hls::stream<t_input_36_skip> s_input_36_skip;
	#pragma HLS STREAM variable=s_input_36_skip depth=1 type=fifo
	ap_uint<1> s_input_36_skip_last[1];

	hls::stream<t_input_44> s_input_44;
	#pragma HLS STREAM variable=s_input_44 depth=1 type=fifo
	ap_uint<1> s_input_44_last[1];

	hls::stream<t_conv_223> s_conv_223;
	#pragma HLS STREAM variable=s_conv_223 depth=1 type=fifo
	ap_uint<1> s_conv_223_last[1];

	ProduceStream<
		t_conv_223_st,
		t_conv_223,
		c_conv_223_ich,
		c_conv_223_ich,
		c_conv_223_iw,
		c_conv_223_ih
	>(
		c_conv_223_st,
		s_conv_223
	);


	PackedConvBuffAcc<
		t_input_36,
		t_conv_223,
		t_input_44,
		t_conv_12_acc,
		c_conv_12_ich,
		c_conv_12_och,
		c_conv_12_iw,
		c_conv_12_ih,
		c_conv_12_ow,
		c_conv_12_oh,
		c_conv_12_fw,
		c_conv_12_fh,
		c_conv_12_stride,
		c_conv_12_pad
	> (
		s_input_36,
		s_conv_223,
		s_input_44,
		s_input_36_skip,
		s_input_36_last,
		s_input_36_skip_last,
		s_input_44_last
	);

	hls::stream<t_conv_152> s_conv_152;
	#pragma HLS STREAM variable=s_conv_152 depth=1 type=fifo
	ap_uint<1> s_conv_152_last[1];

	ReluStreams<
		t_input_44,
		t_conv_152
	> (
		s_input_44,
		s_conv_152,
		s_input_44_last,
		s_conv_152_last
	);

	hls::stream<t_add_225> s_add_225;
	#pragma HLS STREAM variable=s_add_225 depth=1 type=fifo
	ap_uint<1> s_add_225_last[1];

	hls::stream<t_conv_226> s_conv_226;
	#pragma HLS STREAM variable=s_conv_226 depth=1 type=fifo
	ap_uint<1> s_conv_226_last[1];

	ProduceStream<
		t_conv_226_st,
		t_conv_226,
		c_conv_226_ich,
		c_conv_226_ich,
		c_conv_226_iw,
		c_conv_226_ih
	>(
		c_conv_226_st,
		s_conv_226
	);


	PackedConvBuffAcc<
		t_conv_152,
		t_conv_226,
		t_add_225,
		t_conv_14_acc,
		c_conv_14_ich,
		c_conv_14_och,
		c_conv_14_iw,
		c_conv_14_ih,
		c_conv_14_ow,
		c_conv_14_oh,
		c_conv_14_fw,
		c_conv_14_fh,
		c_conv_14_stride,
		c_conv_14_pad
	> (
		s_conv_152,
		s_conv_226,
		s_add_225,
		s_conv_152_last,
		s_add_225_last
	);

	hls::stream<t_relu_155> s_relu_155;
	#pragma HLS STREAM variable=s_relu_155 depth=1 type=fifo
	ap_uint<1> s_relu_155_last[1];

	AddStreams<
		t_add_225,
		t_relu_155
	> (
		s_add_225,
		s_input_36_skip,
		s_relu_155,
		s_input_36_skip_last,
		s_relu_155_last
	);

	hls::stream<t_input_52> s_input_52;
	#pragma HLS STREAM variable=s_input_52 depth=1 type=fifo
	ap_uint<1> s_input_52_last[1];

	ReluStreams<
		t_relu_155,
		t_input_52
	> (
		s_relu_155,
		s_input_52,
		s_relu_155_last,
		s_input_52_last
	);

	hls::stream<t_input_52_skip> s_input_52_skip;
	#pragma HLS STREAM variable=s_input_52_skip depth=1 type=fifo
	ap_uint<1> s_input_52_skip_last[1];

	hls::stream<t_input_60> s_input_60;
	#pragma HLS STREAM variable=s_input_60 depth=1 type=fifo
	ap_uint<1> s_input_60_last[1];

	hls::stream<t_conv_229> s_conv_229;
	#pragma HLS STREAM variable=s_conv_229 depth=1 type=fifo
	ap_uint<1> s_conv_229_last[1];

	ProduceStream<
		t_conv_229_st,
		t_conv_229,
		c_conv_229_ich,
		c_conv_229_ich,
		c_conv_229_iw,
		c_conv_229_ih
	>(
		c_conv_229_st,
		s_conv_229
	);


	PackedConvBuffAcc<
		t_input_52,
		t_conv_229,
		t_input_60,
		t_conv_17_acc,
		c_conv_17_ich,
		c_conv_17_och,
		c_conv_17_iw,
		c_conv_17_ih,
		c_conv_17_ow,
		c_conv_17_oh,
		c_conv_17_fw,
		c_conv_17_fh,
		c_conv_17_stride,
		c_conv_17_pad
	> (
		s_input_52,
		s_conv_229,
		s_input_60,
		s_input_52_skip,
		s_input_52_last,
		s_input_52_skip_last,
		s_input_60_last
	);

	hls::stream<t_conv_159> s_conv_159;
	#pragma HLS STREAM variable=s_conv_159 depth=1 type=fifo
	ap_uint<1> s_conv_159_last[1];

	ReluStreams<
		t_input_60,
		t_conv_159
	> (
		s_input_60,
		s_conv_159,
		s_input_60_last,
		s_conv_159_last
	);

	hls::stream<t_add_231> s_add_231;
	#pragma HLS STREAM variable=s_add_231 depth=1 type=fifo
	ap_uint<1> s_add_231_last[1];

	hls::stream<t_conv_232> s_conv_232;
	#pragma HLS STREAM variable=s_conv_232 depth=1 type=fifo
	ap_uint<1> s_conv_232_last[1];

	ProduceStream<
		t_conv_232_st,
		t_conv_232,
		c_conv_232_ich,
		c_conv_232_ich,
		c_conv_232_iw,
		c_conv_232_ih
	>(
		c_conv_232_st,
		s_conv_232
	);


	PackedConvBuffAcc<
		t_conv_159,
		t_conv_232,
		t_add_231,
		t_conv_19_acc,
		c_conv_19_ich,
		c_conv_19_och,
		c_conv_19_iw,
		c_conv_19_ih,
		c_conv_19_ow,
		c_conv_19_oh,
		c_conv_19_fw,
		c_conv_19_fh,
		c_conv_19_stride,
		c_conv_19_pad
	> (
		s_conv_159,
		s_conv_232,
		s_add_231,
		s_conv_159_last,
		s_add_231_last
	);

	hls::stream<t_add_234> s_add_234;
	#pragma HLS STREAM variable=s_add_234 depth=1 type=fifo
	ap_uint<1> s_add_234_last[1];

	hls::stream<t_conv_235> s_conv_235;
	#pragma HLS STREAM variable=s_conv_235 depth=1 type=fifo
	ap_uint<1> s_conv_235_last[1];

	ProduceStream<
		t_conv_235_st,
		t_conv_235,
		c_conv_235_ich,
		c_conv_235_ich,
		c_conv_235_iw,
		c_conv_235_ih
	>(
		c_conv_235_st,
		s_conv_235
	);


	PackedConvBuffAcc<
		t_input_52_skip,
		t_conv_235,
		t_add_234,
		t_conv_20_acc,
		c_conv_20_ich,
		c_conv_20_och,
		c_conv_20_iw,
		c_conv_20_ih,
		c_conv_20_ow,
		c_conv_20_oh,
		c_conv_20_fw,
		c_conv_20_fh,
		c_conv_20_stride,
		c_conv_20_pad
	> (
		s_input_52_skip,
		s_conv_235,
		s_add_234,
		s_input_52_skip_last,
		s_add_234_last
	);

	hls::stream<t_relu_164> s_relu_164;
	#pragma HLS STREAM variable=s_relu_164 depth=1 type=fifo
	ap_uint<1> s_relu_164_last[1];

	AddStreams<
		t_add_231,
		t_relu_164
	> (
		s_add_231,
		s_add_234,
		s_relu_164,
		s_add_234_last,
		s_relu_164_last
	);

	hls::stream<t_input_72> s_input_72;
	#pragma HLS STREAM variable=s_input_72 depth=1 type=fifo
	ap_uint<1> s_input_72_last[1];

	ReluStreams<
		t_relu_164,
		t_input_72
	> (
		s_relu_164,
		s_input_72,
		s_relu_164_last,
		s_input_72_last
	);

	hls::stream<t_input_72_skip> s_input_72_skip;
	#pragma HLS STREAM variable=s_input_72_skip depth=1 type=fifo
	ap_uint<1> s_input_72_skip_last[1];

	hls::stream<t_input_80> s_input_80;
	#pragma HLS STREAM variable=s_input_80 depth=1 type=fifo
	ap_uint<1> s_input_80_last[1];

	hls::stream<t_conv_238> s_conv_238;
	#pragma HLS STREAM variable=s_conv_238 depth=1 type=fifo
	ap_uint<1> s_conv_238_last[1];

	ProduceStream<
		t_conv_238_st,
		t_conv_238,
		c_conv_238_ich,
		c_conv_238_ich,
		c_conv_238_iw,
		c_conv_238_ih
	>(
		c_conv_238_st,
		s_conv_238
	);


	PackedConvBuffAcc<
		t_input_72,
		t_conv_238,
		t_input_80,
		t_conv_23_acc,
		c_conv_23_ich,
		c_conv_23_och,
		c_conv_23_iw,
		c_conv_23_ih,
		c_conv_23_ow,
		c_conv_23_oh,
		c_conv_23_fw,
		c_conv_23_fh,
		c_conv_23_stride,
		c_conv_23_pad
	> (
		s_input_72,
		s_conv_238,
		s_input_80,
		s_input_72_skip,
		s_input_72_last,
		s_input_72_skip_last,
		s_input_80_last
	);

	hls::stream<t_conv_168> s_conv_168;
	#pragma HLS STREAM variable=s_conv_168 depth=1 type=fifo
	ap_uint<1> s_conv_168_last[1];

	ReluStreams<
		t_input_80,
		t_conv_168
	> (
		s_input_80,
		s_conv_168,
		s_input_80_last,
		s_conv_168_last
	);

	hls::stream<t_add_240> s_add_240;
	#pragma HLS STREAM variable=s_add_240 depth=1 type=fifo
	ap_uint<1> s_add_240_last[1];

	hls::stream<t_conv_241> s_conv_241;
	#pragma HLS STREAM variable=s_conv_241 depth=1 type=fifo
	ap_uint<1> s_conv_241_last[1];

	ProduceStream<
		t_conv_241_st,
		t_conv_241,
		c_conv_241_ich,
		c_conv_241_ich,
		c_conv_241_iw,
		c_conv_241_ih
	>(
		c_conv_241_st,
		s_conv_241
	);


	PackedConvBuffAcc<
		t_conv_168,
		t_conv_241,
		t_add_240,
		t_conv_25_acc,
		c_conv_25_ich,
		c_conv_25_och,
		c_conv_25_iw,
		c_conv_25_ih,
		c_conv_25_ow,
		c_conv_25_oh,
		c_conv_25_fw,
		c_conv_25_fh,
		c_conv_25_stride,
		c_conv_25_pad
	> (
		s_conv_168,
		s_conv_241,
		s_add_240,
		s_conv_168_last,
		s_add_240_last
	);

	hls::stream<t_relu_171> s_relu_171;
	#pragma HLS STREAM variable=s_relu_171 depth=1 type=fifo
	ap_uint<1> s_relu_171_last[1];

	AddStreams<
		t_add_240,
		t_relu_171
	> (
		s_add_240,
		s_input_72_skip,
		s_relu_171,
		s_input_72_skip_last,
		s_relu_171_last
	);

	hls::stream<t_input_88> s_input_88;
	#pragma HLS STREAM variable=s_input_88 depth=1 type=fifo
	ap_uint<1> s_input_88_last[1];

	ReluStreams<
		t_relu_171,
		t_input_88
	> (
		s_relu_171,
		s_input_88,
		s_relu_171_last,
		s_input_88_last
	);

	hls::stream<t_input_88_skip> s_input_88_skip;
	#pragma HLS STREAM variable=s_input_88_skip depth=1 type=fifo
	ap_uint<1> s_input_88_skip_last[1];

	hls::stream<t_input_96> s_input_96;
	#pragma HLS STREAM variable=s_input_96 depth=1 type=fifo
	ap_uint<1> s_input_96_last[1];

	hls::stream<t_conv_244> s_conv_244;
	#pragma HLS STREAM variable=s_conv_244 depth=1 type=fifo
	ap_uint<1> s_conv_244_last[1];

	ProduceStream<
		t_conv_244_st,
		t_conv_244,
		c_conv_244_ich,
		c_conv_244_ich,
		c_conv_244_iw,
		c_conv_244_ih
	>(
		c_conv_244_st,
		s_conv_244
	);


	PackedConvBuffAcc<
		t_input_88,
		t_conv_244,
		t_input_96,
		t_conv_28_acc,
		c_conv_28_ich,
		c_conv_28_och,
		c_conv_28_iw,
		c_conv_28_ih,
		c_conv_28_ow,
		c_conv_28_oh,
		c_conv_28_fw,
		c_conv_28_fh,
		c_conv_28_stride,
		c_conv_28_pad
	> (
		s_input_88,
		s_conv_244,
		s_input_96,
		s_input_88_skip,
		s_input_88_last,
		s_input_88_skip_last,
		s_input_96_last
	);

	hls::stream<t_conv_175> s_conv_175;
	#pragma HLS STREAM variable=s_conv_175 depth=1 type=fifo
	ap_uint<1> s_conv_175_last[1];

	ReluStreams<
		t_input_96,
		t_conv_175
	> (
		s_input_96,
		s_conv_175,
		s_input_96_last,
		s_conv_175_last
	);

	hls::stream<t_add_246> s_add_246;
	#pragma HLS STREAM variable=s_add_246 depth=1 type=fifo
	ap_uint<1> s_add_246_last[1];

	hls::stream<t_conv_247> s_conv_247;
	#pragma HLS STREAM variable=s_conv_247 depth=1 type=fifo
	ap_uint<1> s_conv_247_last[1];

	ProduceStream<
		t_conv_247_st,
		t_conv_247,
		c_conv_247_ich,
		c_conv_247_ich,
		c_conv_247_iw,
		c_conv_247_ih
	>(
		c_conv_247_st,
		s_conv_247
	);


	PackedConvBuffAcc<
		t_conv_175,
		t_conv_247,
		t_add_246,
		t_conv_30_acc,
		c_conv_30_ich,
		c_conv_30_och,
		c_conv_30_iw,
		c_conv_30_ih,
		c_conv_30_ow,
		c_conv_30_oh,
		c_conv_30_fw,
		c_conv_30_fh,
		c_conv_30_stride,
		c_conv_30_pad
	> (
		s_conv_175,
		s_conv_247,
		s_add_246,
		s_conv_175_last,
		s_add_246_last
	);

	hls::stream<t_relu_178> s_relu_178;
	#pragma HLS STREAM variable=s_relu_178 depth=1 type=fifo
	ap_uint<1> s_relu_178_last[1];

	AddStreams<
		t_add_246,
		t_relu_178
	> (
		s_add_246,
		s_input_88_skip,
		s_relu_178,
		s_input_88_skip_last,
		s_relu_178_last
	);

	hls::stream<t_input_104> s_input_104;
	#pragma HLS STREAM variable=s_input_104 depth=1 type=fifo
	ap_uint<1> s_input_104_last[1];

	ReluStreams<
		t_relu_178,
		t_input_104
	> (
		s_relu_178,
		s_input_104,
		s_relu_178_last,
		s_input_104_last
	);

	hls::stream<t_input_104_skip> s_input_104_skip;
	#pragma HLS STREAM variable=s_input_104_skip depth=1 type=fifo
	ap_uint<1> s_input_104_skip_last[1];

	hls::stream<t_input_112> s_input_112;
	#pragma HLS STREAM variable=s_input_112 depth=1 type=fifo
	ap_uint<1> s_input_112_last[1];

	hls::stream<t_conv_250> s_conv_250;
	#pragma HLS STREAM variable=s_conv_250 depth=1 type=fifo
	ap_uint<1> s_conv_250_last[1];

	ProduceStream<
		t_conv_250_st,
		t_conv_250,
		c_conv_250_ich,
		c_conv_250_ich,
		c_conv_250_iw,
		c_conv_250_ih
	>(
		c_conv_250_st,
		s_conv_250
	);


	PackedConvBuffAcc<
		t_input_104,
		t_conv_250,
		t_input_112,
		t_conv_33_acc,
		c_conv_33_ich,
		c_conv_33_och,
		c_conv_33_iw,
		c_conv_33_ih,
		c_conv_33_ow,
		c_conv_33_oh,
		c_conv_33_fw,
		c_conv_33_fh,
		c_conv_33_stride,
		c_conv_33_pad
	> (
		s_input_104,
		s_conv_250,
		s_input_112,
		s_input_104_skip,
		s_input_104_last,
		s_input_104_skip_last,
		s_input_112_last
	);

	hls::stream<t_conv_182> s_conv_182;
	#pragma HLS STREAM variable=s_conv_182 depth=1 type=fifo
	ap_uint<1> s_conv_182_last[1];

	ReluStreams<
		t_input_112,
		t_conv_182
	> (
		s_input_112,
		s_conv_182,
		s_input_112_last,
		s_conv_182_last
	);

	hls::stream<t_add_252> s_add_252;
	#pragma HLS STREAM variable=s_add_252 depth=1 type=fifo
	ap_uint<1> s_add_252_last[1];

	hls::stream<t_conv_253> s_conv_253;
	#pragma HLS STREAM variable=s_conv_253 depth=1 type=fifo
	ap_uint<1> s_conv_253_last[1];

	ProduceStream<
		t_conv_253_st,
		t_conv_253,
		c_conv_253_ich,
		c_conv_253_ich,
		c_conv_253_iw,
		c_conv_253_ih
	>(
		c_conv_253_st,
		s_conv_253
	);


	PackedConvBuffAcc<
		t_conv_182,
		t_conv_253,
		t_add_252,
		t_conv_35_acc,
		c_conv_35_ich,
		c_conv_35_och,
		c_conv_35_iw,
		c_conv_35_ih,
		c_conv_35_ow,
		c_conv_35_oh,
		c_conv_35_fw,
		c_conv_35_fh,
		c_conv_35_stride,
		c_conv_35_pad
	> (
		s_conv_182,
		s_conv_253,
		s_add_252,
		s_conv_182_last,
		s_add_252_last
	);

	hls::stream<t_add_255> s_add_255;
	#pragma HLS STREAM variable=s_add_255 depth=1 type=fifo
	ap_uint<1> s_add_255_last[1];

	hls::stream<t_conv_256> s_conv_256;
	#pragma HLS STREAM variable=s_conv_256 depth=1 type=fifo
	ap_uint<1> s_conv_256_last[1];

	ProduceStream<
		t_conv_256_st,
		t_conv_256,
		c_conv_256_ich,
		c_conv_256_ich,
		c_conv_256_iw,
		c_conv_256_ih
	>(
		c_conv_256_st,
		s_conv_256
	);


	PackedConvBuffAcc<
		t_input_104_skip,
		t_conv_256,
		t_add_255,
		t_conv_36_acc,
		c_conv_36_ich,
		c_conv_36_och,
		c_conv_36_iw,
		c_conv_36_ih,
		c_conv_36_ow,
		c_conv_36_oh,
		c_conv_36_fw,
		c_conv_36_fh,
		c_conv_36_stride,
		c_conv_36_pad
	> (
		s_input_104_skip,
		s_conv_256,
		s_add_255,
		s_input_104_skip_last,
		s_add_255_last
	);

	hls::stream<t_relu_187> s_relu_187;
	#pragma HLS STREAM variable=s_relu_187 depth=1 type=fifo
	ap_uint<1> s_relu_187_last[1];

	AddStreams<
		t_add_252,
		t_relu_187
	> (
		s_add_252,
		s_add_255,
		s_relu_187,
		s_add_255_last,
		s_relu_187_last
	);

	hls::stream<t_input_124> s_input_124;
	#pragma HLS STREAM variable=s_input_124 depth=1 type=fifo
	ap_uint<1> s_input_124_last[1];

	ReluStreams<
		t_relu_187,
		t_input_124
	> (
		s_relu_187,
		s_input_124,
		s_relu_187_last,
		s_input_124_last
	);

	hls::stream<t_input_124_skip> s_input_124_skip;
	#pragma HLS STREAM variable=s_input_124_skip depth=1 type=fifo
	ap_uint<1> s_input_124_skip_last[1];

	hls::stream<t_input_132> s_input_132;
	#pragma HLS STREAM variable=s_input_132 depth=1 type=fifo
	ap_uint<1> s_input_132_last[1];

	hls::stream<t_conv_259> s_conv_259;
	#pragma HLS STREAM variable=s_conv_259 depth=1 type=fifo
	ap_uint<1> s_conv_259_last[1];

	ProduceStream<
		t_conv_259_st,
		t_conv_259,
		c_conv_259_ich,
		c_conv_259_ich,
		c_conv_259_iw,
		c_conv_259_ih
	>(
		c_conv_259_st,
		s_conv_259
	);


	PackedConvBuffAcc<
		t_input_124,
		t_conv_259,
		t_input_132,
		t_conv_39_acc,
		c_conv_39_ich,
		c_conv_39_och,
		c_conv_39_iw,
		c_conv_39_ih,
		c_conv_39_ow,
		c_conv_39_oh,
		c_conv_39_fw,
		c_conv_39_fh,
		c_conv_39_stride,
		c_conv_39_pad
	> (
		s_input_124,
		s_conv_259,
		s_input_132,
		s_input_124_skip,
		s_input_124_last,
		s_input_124_skip_last,
		s_input_132_last
	);

	hls::stream<t_conv_191> s_conv_191;
	#pragma HLS STREAM variable=s_conv_191 depth=1 type=fifo
	ap_uint<1> s_conv_191_last[1];

	ReluStreams<
		t_input_132,
		t_conv_191
	> (
		s_input_132,
		s_conv_191,
		s_input_132_last,
		s_conv_191_last
	);

	hls::stream<t_add_261> s_add_261;
	#pragma HLS STREAM variable=s_add_261 depth=1 type=fifo
	ap_uint<1> s_add_261_last[1];

	hls::stream<t_conv_262> s_conv_262;
	#pragma HLS STREAM variable=s_conv_262 depth=1 type=fifo
	ap_uint<1> s_conv_262_last[1];

	ProduceStream<
		t_conv_262_st,
		t_conv_262,
		c_conv_262_ich,
		c_conv_262_ich,
		c_conv_262_iw,
		c_conv_262_ih
	>(
		c_conv_262_st,
		s_conv_262
	);


	PackedConvBuffAcc<
		t_conv_191,
		t_conv_262,
		t_add_261,
		t_conv_41_acc,
		c_conv_41_ich,
		c_conv_41_och,
		c_conv_41_iw,
		c_conv_41_ih,
		c_conv_41_ow,
		c_conv_41_oh,
		c_conv_41_fw,
		c_conv_41_fh,
		c_conv_41_stride,
		c_conv_41_pad
	> (
		s_conv_191,
		s_conv_262,
		s_add_261,
		s_conv_191_last,
		s_add_261_last
	);

	hls::stream<t_relu_194> s_relu_194;
	#pragma HLS STREAM variable=s_relu_194 depth=1 type=fifo
	ap_uint<1> s_relu_194_last[1];

	AddStreams<
		t_add_261,
		t_relu_194
	> (
		s_add_261,
		s_input_124_skip,
		s_relu_194,
		s_input_124_skip_last,
		s_relu_194_last
	);

	hls::stream<t_input_140> s_input_140;
	#pragma HLS STREAM variable=s_input_140 depth=1 type=fifo
	ap_uint<1> s_input_140_last[1];

	ReluStreams<
		t_relu_194,
		t_input_140
	> (
		s_relu_194,
		s_input_140,
		s_relu_194_last,
		s_input_140_last
	);

	hls::stream<t_input_140_skip> s_input_140_skip;
	#pragma HLS STREAM variable=s_input_140_skip depth=1 type=fifo
	ap_uint<1> s_input_140_skip_last[1];

	hls::stream<t_input_148> s_input_148;
	#pragma HLS STREAM variable=s_input_148 depth=1 type=fifo
	ap_uint<1> s_input_148_last[1];

	hls::stream<t_conv_265> s_conv_265;
	#pragma HLS STREAM variable=s_conv_265 depth=1 type=fifo
	ap_uint<1> s_conv_265_last[1];

	ProduceStream<
		t_conv_265_st,
		t_conv_265,
		c_conv_265_ich,
		c_conv_265_ich,
		c_conv_265_iw,
		c_conv_265_ih
	>(
		c_conv_265_st,
		s_conv_265
	);


	PackedConvBuffAcc<
		t_input_140,
		t_conv_265,
		t_input_148,
		t_conv_44_acc,
		c_conv_44_ich,
		c_conv_44_och,
		c_conv_44_iw,
		c_conv_44_ih,
		c_conv_44_ow,
		c_conv_44_oh,
		c_conv_44_fw,
		c_conv_44_fh,
		c_conv_44_stride,
		c_conv_44_pad
	> (
		s_input_140,
		s_conv_265,
		s_input_148,
		s_input_140_skip,
		s_input_140_last,
		s_input_140_skip_last,
		s_input_148_last
	);

	hls::stream<t_conv_198> s_conv_198;
	#pragma HLS STREAM variable=s_conv_198 depth=1 type=fifo
	ap_uint<1> s_conv_198_last[1];

	ReluStreams<
		t_input_148,
		t_conv_198
	> (
		s_input_148,
		s_conv_198,
		s_input_148_last,
		s_conv_198_last
	);

	hls::stream<t_add_267> s_add_267;
	#pragma HLS STREAM variable=s_add_267 depth=1 type=fifo
	ap_uint<1> s_add_267_last[1];

	hls::stream<t_conv_268> s_conv_268;
	#pragma HLS STREAM variable=s_conv_268 depth=1 type=fifo
	ap_uint<1> s_conv_268_last[1];

	ProduceStream<
		t_conv_268_st,
		t_conv_268,
		c_conv_268_ich,
		c_conv_268_ich,
		c_conv_268_iw,
		c_conv_268_ih
	>(
		c_conv_268_st,
		s_conv_268
	);


	PackedConvBuffAcc<
		t_conv_198,
		t_conv_268,
		t_add_267,
		t_conv_46_acc,
		c_conv_46_ich,
		c_conv_46_och,
		c_conv_46_iw,
		c_conv_46_ih,
		c_conv_46_ow,
		c_conv_46_oh,
		c_conv_46_fw,
		c_conv_46_fh,
		c_conv_46_stride,
		c_conv_46_pad
	> (
		s_conv_198,
		s_conv_268,
		s_add_267,
		s_conv_198_last,
		s_add_267_last
	);

	hls::stream<t_relu_201> s_relu_201;
	#pragma HLS STREAM variable=s_relu_201 depth=1 type=fifo
	ap_uint<1> s_relu_201_last[1];

	AddStreams<
		t_add_267,
		t_relu_201
	> (
		s_add_267,
		s_input_140_skip,
		s_relu_201,
		s_input_140_skip_last,
		s_relu_201_last
	);

	hls::stream<t_pad_202> s_pad_202;
	#pragma HLS STREAM variable=s_pad_202 depth=1 type=fifo
	ap_uint<1> s_pad_202_last[1];

	ReluStreams<
		t_relu_201,
		t_pad_202
	> (
		s_relu_201,
		s_pad_202,
		s_relu_201_last,
		s_pad_202_last
	);

	hls::stream<t_averagepool_203> s_averagepool_203;
	#pragma HLS STREAM variable=s_averagepool_203 depth=1 type=fifo
	ap_uint<1> s_averagepool_203_last[1];

	t_pad_202 s_pad_202_tmp;
	s_pad_202.read(s_pad_202_tmp);
	s_averagepool_203.write((t_averagepool_203)(s_pad_202_tmp));

	hls::stream<t_input_156> s_input_156;
	#pragma HLS STREAM variable=s_input_156 depth=1 type=fifo
	ap_uint<1> s_input_156_last[1];

	AveragePoolStreams<
		t_averagepool_203,
		t_input_156,
		t_averagepool_50_acc,
		c_averagepool_50_och,
		c_averagepool_50_ow,
		c_averagepool_50_oh,
		c_averagepool_50_fw,
		c_averagepool_50_fh,
		c_averagepool_50_stride,
		c_averagepool_50_pad
	> (
		s_averagepool_203,
		s_input_156,
		s_averagepool_203_last,
		s_input_156_last
	);

	hls::stream<t_output> s_output;
	#pragma HLS STREAM variable=s_output depth=1 type=fifo
	ap_uint<1> s_output_last[1];

	hls::stream<t_conv_271> s_conv_271;
	#pragma HLS STREAM variable=s_conv_271 depth=1 type=fifo
	ap_uint<1> s_conv_271_last[1];

	ProduceStream<
		t_conv_271_st,
		t_conv_271,
		c_conv_271_ich,
		c_conv_271_ich,
		c_conv_271_iw,
		c_conv_271_ih
	>(
		c_conv_271_st,
		s_conv_271
	);


	PackedConvBuffAcc<
		t_input_156,
		t_conv_271,
		t_output,
		t_conv_51_acc,
		c_conv_51_ich,
		c_conv_51_och,
		c_conv_51_iw,
		c_conv_51_ih,
		c_conv_51_ow,
		c_conv_51_oh,
		c_conv_51_fw,
		c_conv_51_fh,
		c_conv_51_stride,
		c_conv_51_pad
	> (
		s_input_156,
		s_conv_271,
		s_output,
		s_input_156_last,
		s_output_last
	);

	ConsumeStream<
		t_output,
		t_o_data,
		c_output_och,
		c_output_ow,
		c_output_oh
	> (
		s_output,
		o_data,
		s_output_last
	);
}
