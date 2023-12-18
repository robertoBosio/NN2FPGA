#ifndef NN2FPGA_LINE_BUFFER_UTILS_H_
#define NN2FPGA_LINE_BUFFER_UTILS_H_

#include "ap_int.h"
#include "hls_stream.h"
#include "debug.h"

namespace nn2fpga {

template <typename din_t, typename dout_t, int ICH, int IH, int IW, int c_fw, int c_fh,
          int c_str, int c_pad, int c_ow_ops, int c_ops, int c_ops_out>
void pad_input(hls::stream<din_t> din[(c_fw+(c_ow_ops-1)*c_str) * c_fh],
               hls::stream<dout_t> o_data[1]) {
  /* #pragma HLS inline */
  static_assert(c_ops % c_ops_out == 0, "c_ops must be a multiple of c_ops_out");

  /* This handles padding aware inputs */

  constexpr int c_pad_index_h = c_pad * (c_fh - 1) / 2;
  constexpr int c_pad_index_w = c_pad * (c_fw - 1) / 2;

  constexpr int IH_REM = IH - (IH % c_str)*(1-c_pad);
  constexpr int IW_REM = IW - (IW % c_str)*(1-c_pad);
  constexpr int IH_PAD = IH + c_pad_index_h * 2 - IH_REM*(1-c_pad);
  constexpr int IW_PAD = IW + c_pad_index_w * 2 - IW_REM*(1-c_pad);
  constexpr int FSZ = c_fh * (c_fw+(c_ow_ops-1)*c_str);
  constexpr int FW = (c_fw+(c_ow_ops-1)*c_str);
  constexpr int LAST_IDX = FSZ - 1 - (c_fw+(c_ow_ops-1)*c_str-c_pad_index_w)%c_ow_ops;

  bool s_last;
  
  din_t s_read[FSZ];
  #ifndef __SYNTHESIS__
      std::cout << "pad_input " << ICH << " " << c_pad << " " << c_ops << " " << c_ops_out << std::endl;
      // Printing the size
      for (auto s_i = 0; s_i < FSZ; s_i++) {
        std::cout << "s_read[" << s_i << "] = " << din[s_i].size() << std::endl;
      }
      std::cout << "IH_REM " << IH_REM << " IW_REM " << IW_REM << std::endl;
  #endif

  for (auto s_index_h = 0; s_index_h < IH_REM; s_index_h += c_str) {
    for (auto s_index_w = 0; s_index_w < IW_REM; s_index_w += c_str*c_ow_ops) {
      for (auto s_index_ich = 0; s_index_ich < ICH; s_index_ich+=c_ops) {
        for (auto s_ops = 0; s_ops < c_ops; s_ops+=c_ops_out) {
#pragma HLS pipeline style = stp II=1
          dout_t s_write;
          for (auto s_fh = 0; s_fh < c_fh; s_fh++) {
            for (auto s_fw = 0; s_fw < FW; s_fw++) {

              auto s_index = s_fh * FW + s_fw;

              bool s_data_read = true;

              s_data_read &= (s_index_h >= (c_pad_index_h - s_fh));
              s_data_read &= (s_index_h < (IH + c_pad_index_h - s_fh));
              s_data_read &= (s_index_w >= (c_pad_index_w - s_fw));
              s_data_read &= (s_index_w < (IW + c_pad_index_w - s_fw));

              if (s_data_read) {
                if (s_ops == 0) {
                  s_read[FSZ - s_index - 1] = din[FSZ - s_index - 1].read();
                  if (s_index == LAST_IDX) s_last = s_read[FSZ - s_index - 1].last;
                }
                for (auto s_i = 0; s_i < c_ops_out; s_i++) {
                  s_write.data[FSZ - s_index - 1][s_i] = s_read[FSZ - s_index - 1].data[0][s_ops + s_i];
                }
                s_write.last = s_read[FSZ - s_index - 1].last;
              } else {
                // for (auto s_i = 0; s_i < c_ops; s_i++) {
                // This is padding branch, if the data of the window should not be read
                // form the input stream then we pad it with zeros
                for (auto s_i = 0; s_i < c_ops_out; s_i++) {
                  s_write.data[FSZ - s_index - 1][s_i] = 0;
                }
                s_write.last = s_last;
              }
            }
          }
          o_data[0].write(s_write);
        }
      }
    }
  }

  #ifndef __SYNTHESIS__
    // Check that all the input streams are empty
    for (auto s_i = 0; s_i < FSZ; s_i++) {
      if (din[s_i].size() > 0) {
        std::cout << "#### Not empty input stream" << std::endl;
        std::cout << "din[" << s_i << "] = " << din[s_i].size() << std::endl;
      }
      assert (din[s_i].size() == 0);
    }
    // Check that all the output streams are not empty
    for (auto s_i = 0; s_i < 1; s_i++) {
      if (o_data[s_i].size() == 0) {
        std::cout << "#### Empty output stream" << std::endl;
        std::cout << "o_data[" << s_i << "] = " << o_data[s_i].size() << std::endl;
      }
      assert (o_data[s_i].size() > 0);
    }
    std::cout << "end pad_input " << ICH << " " << c_pad << " " << c_ops << " " << c_ops_out << std::endl;
  #endif
}

template <typename din_t, typename dout_t, int ICH, int IH, int IW,
          int c_ow_ops, int c_ow_ops_out, int c_ops>
void change_ow_ops(hls::stream<din_t> din[c_ow_ops], 
              hls::stream<dout_t> o_data[c_ow_ops_out]) {

  constexpr int c_i_index = IH * IW;
  constexpr int c_num_ich = ICH/c_ops;

  din_t s_read[c_ow_ops];
  for (auto s_index = 0; s_index < c_i_index; s_index++) {
    for (auto s_ich = 0; s_ich < c_num_ich; s_ich++) {
      #pragma HLS pipeline style = stp
      o_data[s_index % c_ow_ops_out].write(din[s_index % c_ow_ops].read());
    }
  }

}

template <typename din_t, typename dout_t, int ICH, int IH, int IW,
          int c_ow_ops, int c_ops, int c_ops_out> 
void bandwidth_adjust(
  hls::stream<din_t> &din,
  hls::stream<dout_t> &o_data
) {

  dout_t s_write;
  #ifndef __SYNTHESIS__
      std::cout << "bandwidth_adjust " << ICH << " " << c_ops << " " << c_ops_out << std::endl;
      // Printing the size
      for (auto s_i = 0; s_i < c_ow_ops; s_i++) {
        std::cout << "din[" << s_i << "] = " << din[s_i].size() << std::endl;
      }
  #endif
  for (auto s_index = 0; s_index < IH * IW * ICH; s_index+=c_ops_out*c_ow_ops) {
    for (auto s_i = 0; s_i < c_ops_out; s_i+=c_ops) {
      #pragma HLS pipeline style = stp
      for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
        din_t s_read = din[s_ow_ops].read();
        for (auto s_j = 0; s_j < c_ops; s_j++) {
          s_write.data[0][s_i+s_j] = s_read.data[0][s_j];
        }
        if (s_i == (c_ops_out - c_ops))
          o_data.write(s_write);
      }
    }
  }
}

/* Adjust the tensor by merging the c_ow_ops_in streams in c_ow_ops_out stream.
 * c_ow_ops_in should be always >= c_ow_ops_out. At the same time aggragate data
 * by reading enough c_ops_in packet to create a c_ops_out one */
template <typename din_t, typename dout_t, int ICH, int IH, int IW,
          int c_ow_ops_in, int c_ow_ops_out, int c_ops_in, int c_ops_out> 
void bandwidth_adjust_down(
  hls::stream<din_t> din[c_ow_ops_in],
  hls::stream<dout_t> o_data[c_ow_ops_out]
) {
  static_assert(c_ow_ops_in % c_ow_ops_out == 0, "c_ow_ops_in is not a multiple of c_ow_ops_out");
  static_assert(c_ow_ops_in >= c_ow_ops_out, "c_ow_ops_in is not bigger than c_ow_ops_out");
  static_assert(c_ops_out % c_ops_in == 0, "c_ops_out is not a multiple of c_ops_in");
  static_assert(c_ops_out >= c_ops_in, "c_ops_out is not bigger than c_ops_in");

  constexpr int c_ich_iter = ICH / c_ops_in;
  dout_t s_write[c_ow_ops_out];
  din_t s_read[c_ow_ops_in][c_ops_out];

  /* Loop on all the tensor with windows of dimension c_ow_ops_in*/
  for (auto s_index = 0; s_index < IH * IW;
      s_index += c_ow_ops_in) {
  
    /* Loop over the streams in input*/
    // The previous convolution writes the data in opposite order
    // because of the line buffer so they must be reordered accordingly
    // for (auto s_ow_ops_in = c_ow_ops_in-c_ow_ops_out; s_ow_ops_in > -1; s_ow_ops_in -= c_ow_ops_out) {
    for (auto s_ow_ops_in = 0; s_ow_ops_in < c_ow_ops_in; s_ow_ops_in += c_ow_ops_out) {

      /* Loop over the ICH dimension */
      for (auto s_ich = 0; s_ich < ICH; s_ich += c_ops_out) {

        /* Loop over the packets in the ICH dimension */
        for (auto s_i = 0; s_i < c_ops_out; s_i += c_ops_in) {
    #pragma HLS pipeline style = stp II = 1

          /* Loop over c_ow_ops_out stream in input in parallel */
          for (auto s_ow_ops_out = 0; s_ow_ops_out < c_ow_ops_out; s_ow_ops_out++) {

            // Select the input stream to read from
            auto s_i_read = s_ow_ops_in + s_ow_ops_out;
            s_read[s_i_read][s_i] = din[s_i_read].read();
            /* Loop over the c_ops_in packet inside a c_ops_out one */
            for (auto s_j = 0; s_j < c_ops_in; s_j++) {

              s_write[s_ow_ops_out].data[0][s_i+s_j] = s_read[s_i_read][s_i].data[0][s_j];
            }

            // If the packet is finished then write it
            if (s_i == (c_ops_out - c_ops_in)){
              s_write[s_ow_ops_out].last = s_read[s_i_read][s_i].last;
              #ifndef __SYNTHESIS__
                #ifdef DEBUG_BANDWIDTH
                  for (auto s_i = 0; s_i < c_ow_ops_out; s_i++) {
                    for (auto s_j = 0; s_j < c_ops_out; s_j++) {
                      std::cout << "s_bandwidth[" << s_i << "][" << s_j << "] = " << s_write[s_i].data[0][s_j] << std::endl;
                    }
                  }
                #endif
              #endif
              o_data[s_ow_ops_out].write(s_write[s_ow_ops_out]);
            }

          }

        }
      }
    }
  }
}

template <typename din_t, typename dout_t, int ICH, int IH, int IW,
          int c_ow_ops_in, int c_ow_ops_out, int c_ops_in, int c_ops_out> 
void bandwidth_adjust_up(
  hls::stream<din_t> din[c_ow_ops_in],
  hls::stream<dout_t> o_data[c_ow_ops_out]
) {
  static_assert(c_ow_ops_out % c_ow_ops_in == 0, "c_ow_ops_out is not a multiple of c_ow_ops_in");
  static_assert(c_ow_ops_out >= c_ow_ops_in, "c_ow_ops_out is not bigger than c_ow_ops_in");
  static_assert(c_ops_out % c_ops_in == 0, "c_ops_out is not a multiple of c_ops_in");
  static_assert(c_ops_out >= c_ops_in, "c_ops_out is not bigger than c_ops_in");

  constexpr int c_ich_iter = ICH / c_ops_in;
  dout_t s_write[c_ow_ops_out];
  din_t s_read[c_ow_ops_in][c_ops_out];

  /* Loop on all the tensor with windows of dimension c_ow_ops_in*/
  for (auto s_index = 0; s_index < IH * IW;
      s_index += c_ow_ops_out) {
  
    /* Loop over the streams in input*/
    // The previous convolution writes the data in opposite order
    // because of the line buffer so they must be reordered accordingly
    /* Loop over the ICH dimension */
    for (auto s_ich = 0; s_ich < ICH; s_ich += c_ops_out) {

      for (auto s_ow_ops_out = 0; s_ow_ops_out < c_ow_ops_out; s_ow_ops_out += c_ow_ops_in) {

        /* Loop over the packets in the ICH dimension */
        for (auto s_i = 0; s_i < c_ops_out; s_i += c_ops_in) {
    #pragma HLS pipeline style = stp II = 1

          /* Loop over c_ow_ops_out stream in input in parallel */
          for (auto s_ow_ops_in = 0; s_ow_ops_in < c_ow_ops_in; s_ow_ops_in++) {

            auto s_i_write = s_ow_ops_out + s_ow_ops_in;
            // Select the input stream to read from
            s_read[s_ow_ops_in][s_i] = din[s_ow_ops_in].read();
            /* Loop over the c_ops_in packet inside a c_ops_out one */
            for (auto s_j = 0; s_j < c_ops_in; s_j++) {

              s_write[s_i_write].data[0][s_i+s_j] = s_read[s_ow_ops_in][s_i].data[0][s_j];
            }

            // If the packet is finished then write it
            if (s_i == (c_ops_out - c_ops_in)) {
              s_write[s_i_write].last = s_read[s_ow_ops_in][s_i].last;
              #ifndef __SYNTHESIS__
                #ifdef DEBUG_BANDWIDTH
                  for (auto s_i = 0; s_i < c_ow_ops_out; s_i++) {
                    for (auto s_j = 0; s_j < c_ops_out; s_j++) {
                      std::cout << "s_bandwidth[" << s_i << "][" << s_j << "] = " << s_write[s_i].data[0][s_j] << std::endl;
                    }
                  }
                #endif
              #endif
              o_data[s_i_write].write(s_write[s_i_write]);
            }

          }

        }
      }
    }
  }
}
template <typename din_t, typename dout_t, int ICH, int IH, int IW,
          int c_ow_ops_in, int c_ow_ops_out, int c_ops_in, int c_ops_out> 
void bandwidth_adjust(
  hls::stream<din_t> din[c_ow_ops_in],
  hls::stream<dout_t> o_data[c_ow_ops_out]
) {
  
  #ifndef __SYNTHESIS__
    // Printing stuff to debug
    std::cout << "bandwidth_adjust " << ICH << " " << c_ops_in << " "
              << c_ow_ops_in << " " << c_ops_out << " " << c_ow_ops_out
              << " " << IH << " " << IW
              << std::endl;
    for (auto s_i = 0; s_i < c_ow_ops_in; s_i++) {
      std::cout << "din[" << s_i << "] = " << din[s_i].size() << std::endl;
    }
  #endif

  if constexpr(c_ow_ops_in == c_ow_ops_out) {
    bandwidth_adjust_down<din_t, dout_t, ICH, IH, IW, c_ow_ops_in, c_ow_ops_out, c_ops_in, c_ops_out>(din, o_data);
  } else if constexpr(c_ow_ops_in > c_ow_ops_out) {
    bandwidth_adjust_down<din_t, dout_t, ICH, IH, IW, c_ow_ops_in, c_ow_ops_out, c_ops_in, c_ops_out>(din, o_data);
  } else if constexpr(c_ow_ops_in < c_ow_ops_out) {
    bandwidth_adjust_up<din_t, dout_t, ICH, IH, IW, c_ow_ops_in, c_ow_ops_out, c_ops_in, c_ops_out>(din, o_data);
  }


  #ifndef __SYNTHESIS__
    // Check that all the input streams are empty
    for (auto s_i = 0; s_i < c_ow_ops_in; s_i++) {
      if (din[s_i].size() > 0) {
        std::cout << "#### Not empty input stream" << std::endl;
        std::cout << "din[" << s_i << "] = " << din[s_i].size() << std::endl;
      }
      assert (din[s_i].size() == 0);
    }
    // Check that all the output streams are not empty
    for (auto s_i = 0; s_i < c_ow_ops_out; s_i++) {
      if (o_data[s_i].size() == 0) {
        std::cout << "#### Empty output stream" << std::endl;
        std::cout << "o_data[" << s_i << "] = " << o_data[s_i].size() << std::endl;
      }
      assert (o_data[s_i].size() > 0);
    }
    std::cout << "end bandwidth_adjust " << ICH << " " << c_ops_in << " "
              << c_ow_ops_in << " " << c_ops_out << " " << c_ow_ops_out
              << std::endl;
  #endif
}

template <typename din_t, typename dcomp_t, typename dout_t, int ICH, int OCH, int IH, int IW, int OH, int OW,
          int c_fh, int c_fw, int c_str, int c_pad, int c_pos_h, int c_pos_w, int c_ow_ops, int c_ops, int c_ops_out>
void shift_op(hls::stream<din_t> &din, hls::stream<dcomp_t> &o_compute,
              hls::stream<dout_t> &o_data) {
  /* #pragma HLS inline */

  // Assert that c_ops is a multiple of c_ops_out
  static_assert(c_ops % c_ops_out == 0, "c_ops must be a multiple of c_ops_out");

  constexpr int c_pad_index_h = c_pad * (c_fh - 1) / 2;
  constexpr int c_pad_index_w = c_pad * (c_fw - 1) / 2;
  constexpr int IH_PAD = IH + c_pad_index_h * 2;
  constexpr int IW_PAD = IW + c_pad_index_w * 2;
  // constexpr int c_paddingh_shift = c_pos_h;
  // constexpr int c_paddingw_shift = c_pos_w;
  constexpr int c_strideh_shift = (c_str - 1);
  constexpr int c_stridew_shift = (c_str - 1);
  // constexpr int c_end_paddingh_shift = (c_fh - 1 - c_pos_h);
  // constexpr int c_end_paddingw_shift = (c_fw + c_ow_ops - 2 - c_pos_w);
  // constexpr int c_end_paddingw_shift = (c_fw - 1 - c_pos_w);

  /* Constants for new version */
  constexpr int c_i_index = IH_PAD * IW_PAD * ICH;

  constexpr int c_endh = IH_PAD - c_pad_index_h;
  constexpr int c_endw = IW_PAD - c_pad_index_w + (c_fw - c_pos_w) % (c_ow_ops);
  constexpr int FW = (c_fw+(c_ow_ops-1)*c_str);

  // The window output is written as soon as the first data which falls
  // in the windoow_ops position is available
  // The window position is affected by the padding which adds data to 
  // the beginning of the tensor width
  constexpr int c_paddingh_shift = c_pos_h - c_pad_index_h;
  constexpr int c_paddingw_shift = c_pos_w - c_pad_index_w;

  // No window output is written if the data is not in the windoow_ops position
  // At the end of the tensor width
  // The window position is affected by the padding which adds data to
  // the end of the tensor width
  constexpr int c_end_paddingh_shift = c_fh - 1 - c_pos_h - c_pad_index_h;
  constexpr int c_end_paddingw_shift = c_fw - 1 - c_pos_w - c_pad_index_w;

  // Given the weight stationary degree and the position in the window,
  // the first pixel received varies
  constexpr int c_starth = 0;
  constexpr int c_startw = (c_paddingw_shift % c_ow_ops > 0) ? (c_paddingw_shift % c_ow_ops) : (c_ow_ops+c_paddingw_shift)%c_ow_ops;
  // constexpr int c_startw = (c_paddingw_shift % c_ow_ops > 0) ? (c_paddingw_shift % c_ow_ops) : -1 * (c_paddingw_shift % c_ow_ops);
  const int c_str_adj = (c_str == 1) ? 1 : (c_ow_ops);

  // Stride selects which pixels are sent for computations
  constexpr int c_strw_adj = (c_str == 1) ? 1 : (FW%c_str);
  constexpr int c_strh = (c_paddingh_shift > 0) ? (c_paddingh_shift % c_str) : -1 * (c_paddingh_shift % c_str);
  constexpr int c_strw = (c_paddingw_shift > 0) ? (c_paddingw_shift % (c_str*c_str_adj)) : (FW - c_strw_adj + c_paddingw_shift) % (c_str*c_str_adj);
  
  din_t s_input;
  #pragma HLS aggregate variable=s_input
  #pragma HLS array_partition variable=s_input type=complete
  dcomp_t s_output;
  #pragma HLS array_partition variable=s_output.data type=complete
  #ifndef __SYNTHESIS__
    std::cout << "shift_op " << ICH << " " << IW << " " << IH << " " << c_ops << " " << c_ops_out << " " << c_ow_ops << std::endl;
    std::cout << c_strh << " " << c_strw << " " << c_paddingh_shift << " " << c_paddingw_shift << " " << c_end_paddingh_shift << " " << c_end_paddingw_shift << std::endl;
    std::cout << c_starth << " " << c_startw << " " << c_strw_adj << " " << c_str_adj << std::endl;
    std::cout << "din.size() " << din.size() << std::endl;
  #endif

  for (auto s_index_h = c_starth; s_index_h < IH; s_index_h++) {
    for (auto s_index_w = c_startw; s_index_w < IW; s_index_w+=c_ow_ops) {
      for (auto s_index_ich = 0; s_index_ich < ICH; s_index_ich+=c_ops) {
        for (auto s_index_read = 0; s_index_read < c_ops; s_index_read+=c_ops_out) {
          #pragma HLS pipeline style = stp II=1
          auto s_data_read = s_index_read == 0;
          if (s_data_read) s_input = din.read();
          bool s_compute_write = true;
          auto s_index_h_str = s_index_h % c_str;
          auto s_index_w_str = s_index_w % (c_str * c_str_adj);

          s_compute_write &= (s_index_h >= c_paddingh_shift);
          s_compute_write &= (s_index_h < (IH - c_end_paddingh_shift));
          s_compute_write &= (s_index_w >= c_paddingw_shift);
          s_compute_write &= (s_index_w < (IW - c_end_paddingw_shift));
          s_compute_write &= (s_index_h_str == (c_strh));
          s_compute_write &= (s_index_w_str == (c_strw));

          #ifndef __SYNTHESIS__
            #ifdef DEBUG_LINE
              if (c_ow_ops == IW) {
                // print s_compute_write sub-conditions
                std::cout << (s_index_h >= c_paddingh_shift) << " " << (s_index_h < (IH - c_end_paddingh_shift)) << " " << (s_index_w >= c_paddingw_shift) << " " << (s_index_w < (IW - c_end_paddingw_shift)) << " " << (s_index_h_str == (c_strh)) << " " << (s_index_w_str == (c_strw)) << std::endl;
                std::cout << "s_index_h " << s_index_h << " s_index_w " << s_index_w << " s_index_ich " << s_index_ich << " s_index_read " << s_index_read << std::endl;
                std::cout << "s_compute_write " << s_compute_write << std::endl;
              }
            #endif
          #endif
          for (auto s_index_ops = 0; s_index_ops < c_ops_out; s_index_ops++) {
            s_output.data[0][s_index_ops] = s_input.data[0][s_index_read+s_index_ops];
          }

          s_output.last = s_input.last;
          if (s_compute_write) o_compute.write(s_output);
          if constexpr(std::is_same<dout_t, std::nullptr_t>::value == false) o_data.write(s_output);
        }
      }
    }
  }
  #ifndef __SYNTHESIS__
      if (din.size() > 0) {
        std::cout << "#### Not empty input stream" << std::endl;
        std::cout << "din.size() " << din.size() << std::endl;
      }
      assert (din.size() == 0);
      if constexpr(std::is_same<dout_t, std::nullptr_t>::value == false) {
        if (o_data.size() == 0) {
          std::cout << "#### Empty compute stream" << std::endl;
          std::cout << "o_data.size() " << o_data.size() << std::endl;
        }
        assert (o_data.size() > 0);
      } 
      if ((IW != c_ow_ops*c_str)) {
        if (o_compute.size() == 0) {
          std::cout << "#### Empty compute stream" << std::endl;
          std::cout << "o_compute.size() " << o_compute.size() << std::endl;
        }
        assert (o_compute.size() > 0);
      }
      std::cout << "end shift_op " << ICH << " " << c_ops << " " << c_ops_out << std::endl;
  #endif
}

}  // namespace nn2fpga

#endif  // NN2FPGA_LINE_BUFFER_UTILS_H_
