#ifndef NN2FPGA_STREAM_UTILS_H_
#define NN2FPGA_STREAM_UTILS_H_

#include "nn2fpga/debug.h"
#include "nn2fpga/quantisation.h"
#include <iostream>
#include <fstream>

namespace nn2fpga {

// Read a stream, quantise it, stream it out.
template<typename din_wrap_t,  // Input stream type
         typename din_t,       // Input data type
         typename dout_wrap_t, // Output stream type
         typename dout_t,      // Output data type
         size_t ICH,           // Number of input channels
         size_t IW,            // Input width
         size_t IH,            // Input height
         size_t BITS,          // Number of bits in input data
         size_t OPS,           // Number of data packed in output stream
         size_t WIDTH,         // Width of the data type
         bool PREPROC>         // Preprocessing flag
void
produce_stream(hls::stream<din_wrap_t>& dinStream,
               hls::stream<dout_wrap_t> doutStream[1])
{

  constexpr auto ISZ = (ICH * IH * IW);
  static_assert(BITS % WIDTH == 0,
                "Width stream not a multiple of data width");
  constexpr auto PAR = BITS / WIDTH;
  static_assert(ISZ % OPS == 0, "ISZ \% OPS != 0");
  static_assert(PAR % OPS == 0, "PAR \% OPS != 0");

  constexpr auto OSZ = ISZ / PAR;                 // Number of reads
  constexpr auto OPS_PACKET_PER_READ = PAR / OPS; // Number of packets per read
  const ap_ufixed<32, 0> c_mean[3] = { 0.485, 0.456, 0.406 };
  const ap_ufixed<32, 0> c_std[3] = { 0.229, 0.224, 0.225 };

#ifndef __SYNTHESIS__
  std::cout << "INFO: Call to produce_stream" << std::endl;
  std::cout << "\t\tICH: " << ICH << std::endl;
  std::cout << "\t\tIW: " << IW << std::endl;
  std::cout << "\t\tIH: " << IH << std::endl;
  std::cout << "\t\tBITS: " << BITS << std::endl;
  std::cout << "\t\tOPS: " << OPS << std::endl;
  std::cout << "\t\tPREPROC: " << PREPROC << std::endl;
#endif

  din_wrap_t dinWrap;
  dout_wrap_t doutWrap;
  ap_uint<BITS> din_par = 0;
  auto n_act = 0;

PRODSTR:
  for (auto s_read = 0; s_read < OSZ; s_read++) {
    for (auto s_data = 0; s_data < OPS_PACKET_PER_READ; s_data++) {
#pragma HLS pipeline II = 1 style = stp
      for (auto s_ops = 0; s_ops < OPS; s_ops++) {
#pragma HLS unroll

        auto ich = n_act % ICH;
        ap_ufixed<WIDTH, 0, AP_RND_ZERO> din = 0;
        n_act++;

        if (s_ops == 0 && s_data == 0) {
          dinWrap = dinStream.read();
          din_par.range(BITS - 1, 0) = dinWrap.data.range(BITS - 1, 0);
        }

        din.range(WIDTH - 1, 0) =
          din_par.range((WIDTH * (s_data * OPS + s_ops + 1)) - 1,
                        (WIDTH * (s_data * OPS + s_ops)));

        if constexpr (PREPROC == 1) {
          doutWrap.data[0][s_ops] = dout_t((din - c_mean[ich]) / c_std[ich]);
        } else {
          doutWrap.data[0][s_ops] = (dout_t(din));
        }

        /* Write last only in case of last packet, since input can have more
         * than one out packet */
        if (s_data == (OPS_PACKET_PER_READ - 1)) {
          doutWrap.last = dinWrap.last;
        } else {
          doutWrap.last = false;
        }

        if (s_ops == (OPS - 1)) {
          doutStream[0].write(doutWrap);
        }
        // din_par >>= WIDTH;

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << ap_uint<8>(din_par & 0xff) << " ";
        std::cout << std::setprecision(8) << din << " ";
        std::cout << (din - c_mean[ich]) / c_std[ich] << " ";
        std::cout << doutWrap.data[0][ops] << " ";
#endif /* DEBUG */
#ifdef DEBUG_RES
        // std::cout << (din-c_mean[ich])/c_std[ich] << std::endl;
        std::cout << doutWrap.data[0][ops] << std::endl;
#endif /* DEBUG_RES */
#endif /* __SYNTHESIS__ */
      }
    }
  }

#ifndef __SYNTHESIS__
  if (dinStream.size() > 0) {
    std::cout << "ERROR: Not empty dinStream " << dinStream.size() << " > 0\n";
    assert(dinStream.size() == 0);
  }
  std::cout << "INFO: Finished produce_stream" << std::endl;
#endif
}

// Translate the stream to an array.
template<typename din_t, typename dout_t, int OCH, int OW, int OH, int ow_ops>
void
consume_stream(hls::stream<din_t> dinStream[ow_ops], dout_t dout[OCH * OW * OH])
{
  constexpr unsigned OSZ = OCH * OH * OW;

  for (auto i = 0; i < OSZ; i++) {
    dout[i] = dout_t(dinStream[0].read());
  }
}

template<typename din_wrap_t,
         typename dout_wrap_t,
         int OCH,
         int OW,
         int OH,
         int ow_ops,
         int OPS>
void
consume_stream(hls::stream<din_wrap_t> dinStream[ow_ops],
               hls::stream<dout_wrap_t>& doutStream)
{
  constexpr unsigned OSZ = OCH * OH * OW / OPS;

#ifndef __SYNTHESIS__
  std::cout << "INFO: Call to consume_stream" << std::endl;
  std::cout << "\t\tOCH: " << OCH << std::endl;

  if (dinStream[0].size() == 0) {
    std::cout << "ERROR: consume_stream: dinStream[0].size() "
              << dinStream[0].size() << " == 0\n";
    assert(dinStream[0].size() > 0);
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
      ap_uint<8> tmp = 0;
      tmp.range(7, 0) = dout.data.range(7, 0);
      std::cout << tmp.to_string(16) << std::endl;
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
    std::cout << "ERROR: consume_stream: doutStream.size() "
              << doutStream.size() << " == 0\n";
    assert(doutStream.size() != 0);
  }
  if (dinStream[0].size() > 0) {
    std::cout << "ERROR: consume_stream: dinStream[0].size() "
              << dinStream[0].size() << " > 0\n";
    assert(dinStream[0].size() == 0);
  }
  if (doutStream.size() != (OCH * OH * OW)) {
    std::cout << "ERROR: consume_stream: doutStream.size() "
              << doutStream.size() << " != " << (OCH * OH * OW) << "\n";
    assert(doutStream.size() == (OCH * OH * OW));
  }
  std::cout << "INFO: Finished consume_stream " << std::endl;
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
    "/home/roberto/Documents/NN2FPGA/nn2fpga/tmp/logs/" +
    // "/home-ssd/roberto/Documents/nn2fpga-container/NN2FPGA/nn2fpga/tmp/logs/" +
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
  file_stream.open("/home/roberto/Documents/NN2FPGA/nn2fpga/tmp/logs/" +
                   // "/home-ssd/roberto/Documents/nn2fpga-container/NN2FPGA/nn2fpga/tmp/logs/"
                   // +
                   name + "_bias.txt");
  for (auto och = 0; och < OCH; och += och_step) {
    data_t data = dinStream[0].read();
    for (auto och_op = 0; och_op < och_step; och_op++) {
        file_stream << std::setprecision(16) << "[" << och + och_op << "] "
                    << data[0][och_op] << std::endl;
    }
    doutStream[0].write(data);
  }

  while (!dinStream[0].empty()) {
    data_t data = dinStream[0].read();
    doutStream[0].write(data);
  }
  file_stream.close();
}

#endif /* __SYNTHESIS__ */
  }    // namespace nn2fpga

#endif  // NN2FPGA_STREAM_UTILS_H_
