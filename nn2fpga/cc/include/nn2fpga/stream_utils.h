#ifndef NN2FPGA_STREAM_UTILS_H_
#define NN2FPGA_STREAM_UTILS_H_

#include "nn2fpga/debug.h"
#include "nn2fpga/quantisation.h"
#include <iostream>
#include <fstream>

namespace nn2fpga {

// Read a stream, quantise it, stream it out.
template <typename din_wrap_t, typename din_t, typename dout_wrap_t,
          typename dout_t, typename d_format_t, unsigned ICH, unsigned IW, unsigned IH,
          unsigned c_ow_ops_out, unsigned BITS, unsigned OPS, unsigned PREPROC>
void produce_stream(hls::stream<din_wrap_t>& dinStream,
                    hls::stream<dout_wrap_t> doutStream[c_ow_ops_out]) {
  constexpr auto PAR = BITS / 8;
  constexpr auto ISZ = (ICH * IH * IW);
  const ap_ufixed<32,0> c_mean[3] = {0.485, 0.456, 0.406};
  const ap_ufixed<32,0> c_std[3] = {0.229, 0.224, 0.225};

#ifndef __SYNTHESIS__
  // std::ofstream output;
  // output.open("/home-ssd/roberto/Documents/nn2fpga-container/NN2FPGA/nn2fpga/conv1_act_tensor_csim.txt");
  // Writing in output the template of the function
  std::cout << "produce_stream act. ICH:" << ICH
            << " c_ow_ops_out:" << c_ow_ops_out << " BITS:" << BITS
            << " OPS:" << OPS << " PREPROC:" << PREPROC << std::endl;
  for (auto i = 0; i < 3; i++) {
    std::cout << c_mean[i] << " ";
    std::cout << c_std[i] << std::endl;
  }
#endif
  din_wrap_t dinWrap;
	ap_uint<BITS> din_par;
PRODSTR:
  for (auto i = 0; i < ISZ; i++) {
#pragma HLS pipeline style = stp
    auto par = i % PAR;
    auto ops = i % OPS;
    auto ich = i % ICH;
    auto ow_ops_out = (i / ICH) % c_ow_ops_out;

    ap_ufixed<8,0,AP_RND_ZERO> din;
    if (par == 0) {
      dinWrap = dinStream.read();
      din_par = dinWrap.data;
    }

    dout_wrap_t doutWrap;
    din.range(7,0) = din_par & 0xff;
    if constexpr(PREPROC == 1)
      doutWrap.data[0][ops] = dout_t((din-c_mean[ich])/c_std[ich]);
    else
      doutWrap.data[0][ops] = (dout_t(din));

    #ifndef __SYNTHESIS__
      #ifdef DEBUG
        std::cout << ap_uint<8>(din_par & 0xff) << " ";
        std::cout << std::setprecision(8) << din << " ";
        std::cout << (din-c_mean[ich])/c_std[ich] << " ";
        std::cout << doutWrap.data[0][ops] << " ";
      #endif
      #ifdef DEBUG_RES
        // std::cout << (din-c_mean[ich])/c_std[ich] << std::endl;
        std::cout << doutWrap.data[0][ops] << std::endl;
      #endif
    #endif

    if (par < (PAR - 1)) {
      doutWrap.last = false;
    } else {
      doutWrap.last = dinWrap.last;
    }

    if (ops == (OPS - 1)) {
      doutStream[ow_ops_out] << doutWrap;
      #ifndef __SYNTHESIS__
        #ifdef DEBUG
          std::cout << std::endl;
        #endif
      #endif
    }
    din_par >>= 8;
  }
  #ifndef __SYNTHESIS__
      std::cout << "end produce_stream act " << std::endl;
      // output.close();
  #endif
}

// Translate the stream to an array.
template <typename din_t, typename dout_t, int OCH, int OW, int OH, int ow_ops>
void consume_stream(hls::stream<din_t> dinStream[ow_ops], dout_t dout[OCH * OW * OH]) {
  constexpr unsigned OSZ = OCH * OH * OW;

  for (auto i = 0; i < OSZ; i++) {
    dout[i] = dout_t(dinStream[0].read());
  }
}

template <typename din_wrap_t, typename dout_wrap_t, int OCH, int OW, int OH, int ow_ops, int OPS>
void consume_stream(hls::stream<din_wrap_t> dinStream[ow_ops],
                    hls::stream<dout_wrap_t>& doutStream) {
  constexpr unsigned OSZ = OCH * OH * OW / OPS;

  // If synthesis pragma in not defined then print consume_stream
  // function name
  #ifndef __SYNTHESIS__
    std::cout << "consume_stream OCH:" << OCH << std::endl;
    
    if (dinStream[0].size() == 0) {
      std::cout << "ERROR: consume_stream: dinStream[0].size() " << dinStream[0].size() << " == 0\n";
      assert (dinStream[0].size() > 0);
    }
  #endif
  din_wrap_t wrap;
  for (auto i = 0; i < OSZ; i++) {
    for (auto s_ops = 0; s_ops < OPS; s_ops++) {
      if (s_ops == 0)
        wrap = dinStream[0].read();
      dout_wrap_t dout;
      dout.data = wrap.data[0][s_ops];
      #ifndef __SYNTHESIS__
        #ifdef DEBUG
        // if (c_depth == 1)
          std::cout << dout.data << std::endl;
        #endif
      #endif
      dout.last = wrap.last & (i == (OSZ - 1)) & (s_ops == (OPS - 1));
      dout.keep = -1;
      dout.strb = 1;
      doutStream << dout;
    }
  }
  #ifndef __SYNTHESIS__
    if (doutStream.size() == 0) {
      std::cout << "ERROR: consume_stream: doutStream.size() " << doutStream.size() << " == 0\n";
      assert (doutStream.size() != 0);
    }
    if (dinStream[0].size() > 0) {
      std::cout << "ERROR: consume_stream: dinStream[0].size() " << dinStream[0].size() << " > 0\n";
      assert (dinStream[0].size() == 0);
    }
    if (doutStream.size() != (OCH * OH * OW)) {
      std::cout << "ERROR: consume_stream: doutStream.size() " << doutStream.size() << " != " << (OCH * OH * OW) << "\n";
      assert (doutStream.size() == (OCH * OH * OW));
    }
    std::cout << "end consume_stream " << std::endl;
  #endif
}

#ifndef __SYNTHESIS__
template<typename data_t, int CH, int W, int H, int ch_step, int w_step>
void
act_tensor_hook(hls::stream<data_t> dinStream[w_step],
            hls::stream<data_t> doutStream[w_step],
            std::string name)
{
  /* This function is used to hook activation tensors. It is used to debug the
  layer. ch_step is the och_ops_out parameter of the convolution. w_step is the
  ow_ops_out parameter of the convolution. */

  std::cout << "HOOK FUNCTION ACTIVATION TENSOR" << std::endl;
  std::ofstream file_stream;
  file_stream.open(
    "/home/roberto/Documents/NN2FPGA/nn2fpga/tmp/logs/" +
    name + "_acts.txt");
  for (auto h = 0; h < H; h++) {
    for (auto w = 0; w < W; w += w_step) {
      for (auto ch = 0; ch < CH; ch += ch_step) {
        for (auto ow = 0; ow < w_step; ow++) {
          data_t data = dinStream[ow].read();
          for (auto op = 0; op < ch_step; op++) {

            // Conversion done to print 0 even when the value is -0
            float value = data.data[0][op];
            if (value == -0) {
              file_stream << std::setprecision(8) << "[" << ch + op << "," << h
                          << "," << w + ow << "] 0"
                          << std::endl;
            } else {
              file_stream << std::setprecision(8) << "[" << ch + op << "," << h
                          << "," << w + ow << "] " << data.data[0][op]
                          << std::endl;
            }
          }
          // file_stream.flush();
          doutStream[ow] << data;
        }
      }
    }
  }
  file_stream.close();
  std::cout << "END HOOK FUNCTION ACTIVATION TENSOR" << std::endl;
}

template<typename data_t, int CH, int W, int H, int ch_step, int w_step>
void
act_windowbuffer_hook(hls::stream<data_t> dinStream[1],
            hls::stream<data_t> doutStream[1],
            std::string name)
{
  /* This function is used to hook activation tensors. It is used to debug the
  layer. ch_step is the och_ops_out parameter of the convolution. w_step is the
  ow_ops_out parameter of the convolution. */

  std::cout << "HOOK FUNCTION ACTIVATION WINDOW BUFFER" << std::endl;
  std::ofstream file_stream;
  file_stream.open(
    "/home/roberto/Documents/NN2FPGA/nn2fpga/tmp/logs/" +
    name + "_acts.txt");
  for (auto h = 0; h < H; h++) {
    for (auto w = 0; w < W; w += w_step) {
      for (auto ch = 0; ch < CH; ch += ch_step) {
        data_t data = dinStream[0].read();
        for (auto ow = w_step - 1; ow >= 0; ow--) {
          for (auto op = 0; op < ch_step; op++) {

            // Conversion done to print 0 even when the value is -0
            float value = data.data[ow][op];
            if (value == -0) {
              file_stream << std::setprecision(8) << "[" << ch + op << "," << h
                          << "," << w + ow << "] 0"
                          << std::endl;
            } else {
              file_stream << std::setprecision(8) << "[" << ch + op << "," << h
                          << "," << w + ow << "] " << data.data[ow][op]
                          << std::endl;
            }
          }
          // file_stream.flush();
        }
        doutStream[0] << data;
      }
    }
  }
  file_stream.close();
  std::cout << "END HOOK FUNCTION ACTIVATION WINDOW BUFFER" << std::endl;
}

template<typename data_t, int CH, int W, int H, int ch_step, int w_step>
void
act_shiftop_hook(hls::stream<data_t> &dinStream,
            hls::stream<data_t> &doutStream,
            size_t w_pos,
            std::string name)
{
  /* This function is used to hook activation tensors. It is used to debug the
  layer. ch_step is the och_ops_out parameter of the convolution. w_step is the
  ow_ops_out parameter of the convolution. */

  std::cout << "HOOK FUNCTION ACTIVATION SHIFT_OP" << std::endl;
  std::ofstream file_stream;
  file_stream.open(
    "/home/roberto/Documents/NN2FPGA/nn2fpga/tmp/logs/" +
    name + "_acts.txt");
  for (auto h = 0; h < H; h++) {
    for (auto w = w_pos; w < W; w += w_step) {
      for (auto ch = 0; ch < CH; ch += ch_step) {
        data_t data = dinStream.read();
        for (auto op = 0; op < ch_step; op++) {

          // Conversion done to print 0 even when the value is -0
          float value = data.data[0][op];
          if (value == -0) {
            file_stream << std::setprecision(8) << "[" << ch + op << "," << h
                        << "," << w << "] 0" << std::endl;
          } else {
            file_stream << std::setprecision(8) << "[" << ch + op << "," << h
                        << "," << w << "] " << data.data[0][op]
                        << std::endl;
          }
        }
        // file_stream.flush();
        doutStream << data;
      }
    }
  }

  file_stream.close();
  std::cout << "END HOOK FUNCTION ACTIVATION SHIFT_OP" << std::endl;
}

template<typename data_t, int OCH, int ICH, int FH, int FW, int och_step, int ich_step>
void
weight_tensor_hook(hls::stream<data_t> dinStream[FH * FW],
                   hls::stream<data_t> doutStream[FH * FW],
                   std::string name)
{
  /* This function is used to hook weight tensors. It is used to debug the
  layer. ich_step is the ich_ops parameter of the convolution. och_step is the
  och_ops parameter of the convolution. */

  /* The order in which the filters are printed is based on och_ops and ich_ops
   * of the convolution, it is not a flatten of the tensor */
  std::cout << "weight_tensor_hook" << " OCH:" << OCH << " ICH:" << ICH << " FH:" << FH << " FW:" << FW << " och_step:" << och_step << " ich_step:" << ich_step << std::endl;
  std::ofstream file_stream;
  file_stream.open(
    "/home-ssd/roberto/Documents/nn2fpga-container/NN2FPGA/nn2fpga/tmp/logs/" +
    name + "_weights.txt");
  for (auto ich = 0; ich < ICH; ich += ich_step) {
    for (auto och = 0; och < OCH; och += och_step) {
      for (auto fh = 0; fh < FH; fh++) {
        for (auto fw = 0; fw < FW; fw++) {
          data_t data = dinStream[fw + (fh * FW)].read();
          for (auto och_op = 0; och_op < och_step; och_op++) {
            for (auto ich_op = 0; ich_op < ich_step; ich_op++) {
              file_stream << std::setprecision(8) << "[" << och + och_op << ","
                          << ich + ich_op << "," << FH - fh - 1 << ","
                          << FW - fw - 1 << "] " << data[ich_op][och_op]
                          << std::endl;
            }
          }
          doutStream[fw + (fh * FW)].write(data);
        }
      }
    }
  }

  while (!dinStream[0].empty()) {
    for (auto fw = 0; fw < FW; fw++) {
      for (auto fh = 0; fh < FH; fh++) {
        data_t data = dinStream[fw + (fh * FW)].read();
        doutStream[fw + (fh * FW)].write(data);
      }
    }
  }
  file_stream.close();
}

template<typename data_t, int OCH, int ICH, int FH, int FW, int och_step, int ich_step>
void
bias_tensor_hook(hls::stream<data_t> dinStream[1],
                   hls::stream<data_t> doutStream[1],
                   std::string name)
{
  /* This function is used to hook weight tensors. It is used to debug the
  layer. ich_step is the ich_ops parameter of the convolution. och_step is the
  och_ops parameter of the convolution. */

  /* The order in which the filters are printed is based on och_ops and ich_ops
   * of the convolution, it is not a flatten of the tensor */
  std::cout << "bias_tensor_hook" << " OCH:" << OCH << " ICH:" << ICH << " FH:" << FH << " FW:" << FW << " och_step:" << och_step << " ich_step:" << ich_step << std::endl;
  std::ofstream file_stream;
  file_stream.open(
    "/home-ssd/roberto/Documents/nn2fpga-container/NN2FPGA/nn2fpga/tmp/logs/" +
    name + "_bias.txt");
  for (auto ich = 0; ich < ICH; ich += ich_step) {
    for (auto och = 0; och < OCH; och += och_step) {
      data_t data = dinStream[0].read();
      for (auto och_op = 0; och_op < och_step; och_op++) {
        for (auto ich_op = 0; ich_op < ich_step; ich_op++) {
          file_stream << std::setprecision(16) << "[" << och + och_op << "] "
                      << data[0][och_op] << std::endl;
        }
      }
      doutStream[0].write(data);
    }
  }

  while (!dinStream[0].empty()) {
    data_t data = dinStream[0].read();
    doutStream[0].write(data);
  }
  file_stream.close();
}

#endif /* __SYNTHESIS__ */
}  // namespace nn2fpga

#endif  // NN2FPGA_STREAM_UTILS_H_
