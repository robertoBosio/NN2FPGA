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
	hls::stream<t_i_data> &i_data,
	hls::stream<t_o_data> &o_data
) {

	#pragma HLS interface axis port=i_data
	#pragma HLS interface axis port=o_data
	#pragma HLS INTERFACE ap_ctrl_none port=return
	#pragma HLS DATAFLOW

	hls::stream<t_input> s_input("s_input");
	#pragma HLS STREAM variable=s_input depth=2*c_input_ich type=fifo
	#define c_last_depth 256
	hls::stream<t_last> s_last("s_last");
	#pragma HLS STREAM variable=s_last depth=c_last_depth type=fifo

	ProduceStream<
		t_i_data, 
		t_input,
		c_input_ich,
		c_input_iw,
		c_input_ih,
		c_i_data
	>(
		i_data,
		s_last,
		s_input
	);

	hls::stream<t_conv_5> s_conv_5("s_conv_5");
	#pragma HLS STREAM variable=s_conv_5 depth=c_conv_0_ich*c_conv_0_och type=fifo


	const int c_conv_0_scale = 0;
	const t_conv0_weight_st c_conv0_weight_st[c_conv_0_fh*c_conv_0_fw][c_conv_0_och*c_conv_0_ich/8+1] = {
{0xa633b3645a7755fd, 0xf94e98195a571da6, 0x0c4cf066b560b8fd, 0xeb4aa43f4e99cd97, 0xf44f53664e4cae01, 0xe94c3e306561bf95, 0},
{0x9fa2c34ca37743fc, 0xed44a7405d5fd566, 0xdbc12540f89ea8fc, 0xe344b93b9296bfad, 0xc15d5d5a66a3aefd, 0xcf456bf21b0bfceb, 0},
{0xa440b13bc7754eff, 0x094d8eefa656ec5f, 0x0959ee9ad15cabfc, 0xf94c8d409296e566, 0xe0605d58625ecefc, 0xe84cf6a990552b6d, 0},
{0x35b2bf61b3b04efd, 0xdf49b6a36c60209f, 0x1113c15db18b67ff, 0xf0435e126a96da93, 0xa8554559638d6902, 0x0c48a94b6e4f408f, 0},
{0x3aa3c8529e6c56ff, 0xe0405faebe6ac5a3, 0x1bbfbc9c648f6bfc, 0xf1405c325d98cb66, 0xa35d53456b926dfe, 0x0646524b98ac52a9, 0},
{0x3da4b7b2cf9544fd, 0xfa4699c69f62cf65, 0x3150b89dbb8a65ff, 0x0b46943a8e962f6b, 0xaa5e54c1688d6bfc, 0x1e4b07dc8c0d586f, 0},
{0x4cc0b1245fa88afe, 0x024c6faf645bc09e, 0x0ab7d9ce9c2e9f00, 0x0f456a4b5c98af95, 0xafc45bad605dadfd, 0x1f49ab236646f293, 0},
{0x52a8c9b7aaa88aff, 0xf6466e48af0abc9b, 0x25aa3a9eb461a601, 0x0442644b6095f063, 0xab3a659e6c6b67fe, 0x0d471a0160275767, 0},
{0x4aacb2a506978900, 0x214c67daac5ddb50, 0x1faf2795a84098ff, 0x294a1c3791974f61, 0xad4e63986964b002, 0x324c9caa8f245e6b, 0}
};

	#pragma HLS ARRAY_PARTITION variable=c_conv0_weight_st type=block factor=1 dim=1
	hls::stream<t_conv0_weight> s_conv0_weight[c_conv0_weight_ih*c_conv0_weight_iw];
	#pragma HLS STREAM variable=s_conv0_weight depth=c_conv0_weight_och type=fifo


	hls::stream<t_pad_7> s_pad_7("s_pad_7");
	#pragma HLS STREAM variable=s_pad_7 depth=c_conv_2_ich*c_conv_2_och type=fifo


	const int c_conv_2_scale = 0;
	const t_conv1_weight_st c_conv1_weight_st[c_conv_2_fh*c_conv_2_fw][c_conv_2_och*c_conv_2_ich/8+1] = {
{0x01ff00000000feff, 0xfffe00ff00fefe00, 0xd50bc131eb1a0c54, 0x36bfc2fe27d2e449, 0xb8d4bb45d93e1356, 0x49aac2fa33cadc49, 0xd5af2e1a183a2302, 0x20f3fefd082e0412, 0xd2eb1c00eb4328c8, 0xf9fada00f832e822, 0xdda9cde5e73a1533, 0x1edbd900ef491836, 0xeb0d18fbe2051803, 0x2bd3fafeebda503a, 0xe918f223efe2f90a, 0x18eee1011fdbf0ff, 0x00dc1bd7d2003ec9, 0xcf2507facf42e807, 0x0dda0105d4de2f0c, 0xe9e309fef126de1d, 0xf916261f4726cddb, 0xf7e62bfa1c4324d2, 0xfed81ad9c0e725ea, 0xed071b03c004d6f9, 0x4318af10e8a1be45, 0xea2520ff20b5d4be, 0xafaa0448d52dbde7, 0x2ac5f60445510948, 0xbe2c3c58d1473324, 0x38a824f953eb2036, 0xe700fa21f709f9fc, 0x04eb0d011711f60a, 0},
{0x0000ff0000ffffff, 0xff00ff010000ff00, 0x02ddc31bdb0ff04d, 0x2fdcf1fa0dc7db52, 0xcfdebe3ae80d1054, 0x3cc0d1fa2ac5fe3f, 0xf302051c091cfee0, 0xe6130ffb1a2ff7eb, 0xd8c915fafa3d32dd, 0xfefbd2faf1281518, 0x3caee6e6dff5ea13, 0xf71edefef228dc2f, 0xdc0b3ae63315e0f7, 0x02f015fcf2f25c31, 0x0715fd1cfed5ecf1, 0x06f7f30018d8f7ef, 0xf3f415cdb51740c9, 0xd52903fab931be0d, 0x3efa0df2f9b4fbe3, 0xca142ffcfee9f0e0, 0x434b450618cde8b2, 0xe00339f70637c5b7, 0x14ba1ccfb3d327d6, 0xee1d1f01c21bd505, 0x073dc51140b4d52b, 0xd6204dfd0bbc4ab9, 0xc2a5e736e825c2e8, 0x1cca02fe263e0a58, 0xb32b0d4bf952fe44, 0x5cc6eef153c8313f, 0xf002f71c090af3fc, 0xfdf211fe160fff0c, 0},
{0x00fffffefefefeff, 0xfefefefe00fefefe, 0xd520c73302240250, 0x43bfbefb2bc8ec3c, 0xb6efd140e83a0354, 0x49b7c8f83cdde94b, 0xe6ef242b162205ed, 0x0ff118f71837011e, 0xd5d42602113231d1, 0xffece0faf041291c, 0xf5cad9f31c02db11, 0xfdece8f9f8492709, 0xca0544ebf327f2fc, 0x25d2f7fae8ee4819, 0x0ef90820efd1edec, 0x14e6f3ff1ad5efe9, 0xf3fc1ae8bd1849dd, 0xf706f900d621d011, 0x1e0cff13d3c52a14, 0xd4042ffbf71dc610, 0x0a431e1c37e4f9eb, 0x27f734f62f1febc7, 0x15ca15b6b2e83aeb, 0xde2c08fbba02ca09, 0x2f36a9262bb9df2f, 0xfa121df731dc50c6, 0xafe5f4313045cded, 0x26d2fafc333e3a49, 0xcd51154cd944073d, 0x5802daef51be21d5, 0xe917f820f10d0100, 0x0aed0fff1c10e40a, 0},
{0xfffefefe000101fe, 0xff00ffffff0000ff, 0xbd21d124033bf757, 0x37bdaf010bca1940, 0xcadbc044db2b1b53, 0x3eb8b8fa36bcf24b, 0xd5d93df71c3ffcf8, 0x0de8d92be70f2bca, 0xd6e0fff7e33540e9, 0xfbefc703e80fd924, 0xd9b5df0efa121845, 0x29e515061816022b, 0xe911dd03ebfc3c28, 0x2fd2e8fb05e40d3a, 0xfe0fdffec2ef573e, 0x15c4e50903c6d11e, 0x11d91fd7ce0330dd, 0xde250707c215fb15, 0xf00534d10c06b3c4, 0xfa0b1f18c5235109, 0xf3d2fefef716c7d5, 0xe62627fef2221816, 0x23c823d7bbed1ce8, 0xd62b381fcbf3b6ee, 0x24ff06f4d8cebc06, 0x1a17182411d7fb13, 0xc0b9ea44fe4e1703, 0x1fe3e7f739471c38, 0x461e20131ed915be, 0xad1cffe4f63ad414, 0xf8092406050b06de, 0x04f30d0efe1a1efc, 0},
{0xfefe00fe00feff00, 0xff0000ff00fefe00, 0xe9e4df1bf620ee57, 0x3abed90d1fb30b45, 0xd8f1c834ed211758, 0x3dbfcbf72bb62f45, 0xdb4048f30b3bdede, 0xdee8cd37f50e2ec9, 0xd0e2f60de03a4efa, 0x15e6bf0300f8ea2b, 0x1aaffbe4efe0f42b, 0x23041206f3e4c52c, 0x0b1bfaed37e82003, 0x0bfa02fcf803172f, 0x2bfef7fbc2c8542d, 0xfac2021600b9e116, 0x160938d9abf841d4, 0xcd1b1312c211b100, 0x1d093ecc19d1afb7, 0xdc244518d1ed4dd9, 0x41252cf2cde0d6ac, 0xc8303df3ed20d9e9, 0x28af2fcfb4e53ae5, 0xdb31201cc309c0db, 0xcd1240f02f21def9, 0xf60b4224d90d4900, 0xe4aeff3feb1610f6, 0x06eefeec2e280240, 0x48b80dbb42d2092f, 0x2f3ebbdee442313e, 0xddf33f04f8190cee, 0x06e41b1cf81b3306, 0},
{0xfefefffffffefe00, 0xfefefe0100000000, 0xbb170d130242fe54, 0x3ac0b91506be2f41, 0xbee0c832fb3afe58, 0x43afbbf927bb1e49, 0xc8284214d43e0408, 0x26cdc536ed092e05, 0xd1ee1006133a47e4, 0xfae7d5feee3a201e, 0xf4bbd9f738e0f220, 0x0bfb110215020fe6, 0xec0efe02fe0c070f, 0x18d5f1f91ac1ed18, 0x34e4eafbb9cb5831, 0x04c3090e03b4d012, 0x080b29d1b70835e2, 0xd51c021bc508cd0a, 0xc81150df0d25acc7, 0xf3043a2dd01454ef, 0xfaeff8fde005e4e1, 0x0c2225f8fcd60024, 0x30ef1cbfb4d93eef, 0xd93e1c0dc3dfbbf0, 0xcac0350e0237e406, 0x32f8211d0d4d552f, 0xb40aef2c4e3d1706, 0x06f503ee2b1c5136, 0x554706bc41b81ce0, 0xbc47c2d9e042c2b6, 0xdcfd3606ef1809f7, 0x17e00f18fd202600, 0},
{0xfefffefeffff01fe, 0x00fffffe00fe0000, 0xbb27b73ee12ffb54, 0x4ab3b3fb33bafe27, 0xe5e1d531e5ff1552, 0x37c6bafe2bbbff43, 0x1a3b0c48c6dc460a, 0x13ddacfe1fd4afb2, 0xe3d20bfbec3b2ad7, 0xe003e004fb17e503, 0x1c33c32a0be42005, 0x003e270222dcddc6, 0xff42effae2f82004, 0x08f90702ffea07fa, 0xe40dec26c7f9fa21, 0x24f309001be8e113, 0x22fe06cddff002e0, 0xc42b0f06c2e0d9f1, 0xe5fb1216e8eb2f25, 0x12d924feea18f3e6, 0xdac5f9e1dcecf332, 0x0cfc0618eab71c1d, 0x4bc909cfaec20ce8, 0xbb314d1dbdd3b5ed, 0x2bdacadcd2d2ce29, 0x02dff8f1f803a81f, 0xc1b92535ff3411f4, 0x24e6f70a3a361b1a, 0x4de5ef0d13d0e3cd, 0xae3b47e8e20de4cb, 0xef0e0a20fefa0cf7, 0x10ec09fe180dfaf6, 0},
{0xfefefffefefefefe, 0xff00ff00fffe00ff, 0x06e4b834ebe4d94f, 0x48cad80122ae0038, 0xf0cccd37f9071051, 0x36c5c6fb2fb72749, 0x1a5ddc33f1e231f9, 0xfee8a1f431cfb7af, 0xf0f1fd0dd1202df1, 0xffebd1090e24f115, 0x4720bef7f9d312ca, 0xf94924052bc1b8cc, 0xe23a0bf32717f5fd, 0x03f721f8180934de, 0xfd01f11adcecec15, 0x0eff250210e0f509, 0x251610ddb0efefd8, 0xb3222e14bef5a8f3, 0x2f20fefdf4d1fae0, 0x00f7440b18d3f4c2, 0x31032de1b7c209cb, 0xe8061f0bf3b7d90a, 0x36af08c2a6cd13e8, 0xc23a3e14b7d4b6e9, 0xcffef3e83e26d914, 0xd6d549f4db46f20c, 0xe7a5102ce223ffea, 0x0cef030e231e2b3b, 0x4aabc8bb3e1be846, 0x334511e0e0db3333, 0xf711061915fbfdf4, 0x07f6140210100cf7, 0},
{0xfffe00fe01fffe00, 0xfefefeff0000fe00, 0xc41ac136f127ea4d, 0x49bbaff432ad0d23, 0xdae8dc33fb28f84c, 0x38c6c10028ba1a3a, 0x244de33cb0e0490e, 0x36e89bf127beacda, 0xe6020df628363cd7, 0xcf14e2fcf23e141f, 0x2b2dbe1541e7e0ca, 0xf947290244d7f0af, 0xf72e1b130013feda, 0x0bf506102ebb01e2, 0xf6fa0d14d1faf901, 0x1b0532010fe1f7fd, 0x31090cccade406e8, 0xb4291d13c2dfab05, 0x030dfb180fd11413, 0xf1fe440900fce2e0, 0xd1c9e2d9b6ed2d44, 0x4cef0015e3a4f942, 0x36e909bfb8d213fc, 0xca27280abeadcae0, 0xc4c0c5f22f4ddc39, 0x22c4c7e2f6602f43, 0xb8e00c0c4d3616f7, 0x14040706193c5d1f, 0x5830d0bb42c0f3fb, 0xbe4d0ddbd7d7c3b7, 0xf31afe1c05000af6, 0x14eb0cff1618f3f7, 0}
};

	#pragma HLS ARRAY_PARTITION variable=c_conv1_weight_st type=block factor=1 dim=1
	hls::stream<t_conv1_weight> s_conv1_weight[c_conv1_weight_ih*c_conv1_weight_iw];
	#pragma HLS STREAM variable=s_conv1_weight depth=c_conv1_weight_och type=fifo


	hls::stream<t_averagepool_9> s_averagepool_9("s_averagepool_9");
	#pragma HLS STREAM variable=s_averagepool_9 depth=3 type=fifo

	hls::stream<t_conv_10> s_conv_10("s_conv_10");
	#pragma HLS STREAM variable=s_conv_10 depth=c_averagepool_6_och type=fifo

	hls::stream<t_output> s_output("s_output");
	#pragma HLS STREAM variable=s_output depth=c_conv_7_ich*c_conv_7_och type=fifo


	const int c_conv_7_scale = 0;
	const t_fc_weight_st c_fc_weight_st[c_conv_7_fh*c_conv_7_fw][c_conv_7_och*c_conv_7_ich/1+1] = {
{0x60, 0x77, 0x73, 0xa2, 0xe2, 0x83, 0x7c, 0x83, 0x40, 0x72, 0x8a, 0x7b, 0xdd, 0x6d, 0x88, 0x63, 0x91, 0x93, 0x87, 0x6e, 0x9b, 0x78, 0x87, 0x8a, 0x89, 0x87, 0x88, 0x76, 0x66, 0x7b, 0x71, 0x85, 0x75, 0x98, 0x78, 0xa8, 0x86, 0x72, 0x77, 0x87, 0x76, 0x7a, 0x8e, 0x9a, 0x97, 0x8b, 0x8b, 0x85, 0x78, 0xac, 0x93, 0x87, 0x90, 0x78, 0x90, 0x78, 0x89, 0x6f, 0xa0, 0x6a, 0x33, 0x85, 0x78, 0x77, 0x8c, 0x78, 0x8e, 0x85, 0x7a, 0x86, 0x85, 0x84, 0x76, 0x7a, 0x79, 0x76, 0x7c, 0x86, 0x8d, 0x83, 0x8c, 0x78, 0x84, 0x9e, 0x99, 0x85, 0x7b, 0x75, 0x86, 0x77, 0x77, 0x86, 0x6b, 0x90, 0x6c, 0x8b, 0x84, 0x79, 0x6b, 0x72, 0x8c, 0x54, 0x95, 0x8b, 0x78, 0x70, 0x86, 0x7b, 0x86, 0xdb, 0x77, 0x78, 0xa7, 0xbb, 0x00, 0x8a, 0x88, 0x85, 0x79, 0x91, 0x47, 0x8b, 0xab, 0xbe, 0xb2, 0x95, 0xb5, 0x6c, 0xda, 0x76, 0x5b, 0x83, 0x74, 0x78, 0x76, 0x76, 0x7b, 0x74, 0x83, 0x84, 0x84, 0x84, 0xf8, 0x7a, 0x78, 0x74, 0x7c, 0x70, 0x86, 0x83, 0x78, 0x74, 0x8e, 0x9f, 0x95, 0x86, 0x96, 0x84, 0x77, 0x79, 0}
};

	#pragma HLS ARRAY_PARTITION variable=c_fc_weight_st type=block factor=1 dim=1
	hls::stream<t_fc_weight> s_fc_weight[c_fc_weight_ih*c_fc_weight_iw];
	#pragma HLS STREAM variable=s_fc_weight depth=c_fc_weight_och type=fifo


	ProduceStream<
		t_conv0_weight_st,
		t_conv0_weight,
		c_conv0_weight_ich,
		c_conv0_weight_och,
		c_conv_0_ow,
		c_conv_0_oh,
		c_conv_0_fw,
		c_conv_0_fh,
		c_conv_0_ops
	>(
		c_conv0_weight_st,
		s_conv0_weight
	);


	PackedConvBuffAcc<
		t_input,
		t_conv0_weight,
		t_conv_5,
		t_conv_0_acc,
		c_conv_0_ich,
		c_conv_0_och,
		c_conv_0_iw,
		c_conv_0_ih,
		c_conv_0_ow,
		c_conv_0_oh,
		c_conv_0_fw,
		c_conv_0_fh,
		c_conv_0_relu,
		c_conv_0_stride,
		c_conv_0_pad,
		c_conv_0_scale,
		c_conv_0_ops
	> (
		s_input,
		s_conv0_weight,
		s_conv_5
	);

	ProduceStream<
		t_conv1_weight_st,
		t_conv1_weight,
		c_conv1_weight_ich,
		c_conv1_weight_och,
		c_conv_2_ow,
		c_conv_2_oh,
		c_conv_2_fw,
		c_conv_2_fh,
		c_conv_2_ops
	>(
		c_conv1_weight_st,
		s_conv1_weight
	);


	PackedConvBuffAcc<
		t_conv_5,
		t_conv1_weight,
		t_pad_7,
		t_conv_2_acc,
		c_conv_2_ich,
		c_conv_2_och,
		c_conv_2_iw,
		c_conv_2_ih,
		c_conv_2_ow,
		c_conv_2_oh,
		c_conv_2_fw,
		c_conv_2_fh,
		c_conv_2_relu,
		c_conv_2_stride,
		c_conv_2_pad,
		c_conv_2_scale,
		c_conv_2_ops
	> (
		s_conv_5,
		s_conv1_weight,
		s_pad_7
	);

	PadStream<
		t_pad_7,
		t_averagepool_9,
		c_pad_5_ich,
		c_pad_5_och,
		c_pad_5_iw,
		c_pad_5_ih,
		c_pad_5_ow,
		c_pad_5_oh,
		c_pad_5_pad
	> (
		s_pad_7,
		s_averagepool_9
	);

	PoolStreams<
		t_averagepool_9,
		t_conv_10,
		t_averagepool_6_acc,
		c_averagepool_6_ich,
		c_averagepool_6_och,
		c_averagepool_6_iw,
		c_averagepool_6_ih,
		c_averagepool_6_ow,
		c_averagepool_6_oh,
		c_averagepool_6_fw,
		c_averagepool_6_fh,
		c_averagepool_6_stride,
		c_averagepool_6_pad,
		c_averagepool_6_pool
	> (
		s_averagepool_9,
		s_conv_10
	);

	ProduceStream<
		t_fc_weight_st,
		t_fc_weight,
		c_fc_weight_ich,
		c_fc_weight_och,
		c_conv_7_ow,
		c_conv_7_oh,
		c_conv_7_fw,
		c_conv_7_fh,
		c_conv_7_ops
	>(
		c_fc_weight_st,
		s_fc_weight
	);


	PackedConvBuffAcc<
		t_conv_10,
		t_fc_weight,
		t_output,
		t_conv_7_acc,
		c_conv_7_ich,
		c_conv_7_och,
		c_conv_7_iw,
		c_conv_7_ih,
		c_conv_7_ow,
		c_conv_7_oh,
		c_conv_7_relu,
		c_conv_7_stride,
		c_conv_7_pad,
		c_conv_7_scale,
		c_conv_7_ops
	> (
		s_conv_10,
		s_fc_weight,
		s_output
	);

	ConsumeStream<
		t_output,
		t_o_data,
		c_output_och,
		c_output_ow,
		c_output_oh
	> (
		s_output,
		s_last,
		o_data
	);
}
