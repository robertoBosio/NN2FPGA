#ifndef NN2FPGA_WEIGHTS_UTILS_H_
#define NN2FPGA_WEIGHTS_UTILS_H_

#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "nn2fpga/debug.h"

namespace nn2fpga {

template <typename din_t, typename dout_t, int ICH, int OCH, int OW, int OH,
          int c_fw, int c_fh, int c_ops, int c_ich_ops, int c_reuse>
void produce_stream(const din_t din[c_fh * c_fw][OCH * ICH / (c_ops*c_ich_ops)][c_ops*c_ich_ops],
                    hls::stream<dout_t> o_data[c_fh * c_fw]) {
  constexpr unsigned FSZ = c_fh * c_fw;
  constexpr unsigned c_ch = ICH * OCH / (c_ops*c_ich_ops);
  constexpr unsigned c_o_index = OH * OW / c_reuse;

  #ifndef __SYNTHESIS__
    std::cout << "produce_stream " << FSZ << " " << c_ch << " " << c_ich_ops << " " << c_ops << std::endl;
  #endif

  #ifndef __SYNTHESIS__
    #ifdef DEBUG_WEIGHTS
      for (auto och = 0; och < OCH; och++) {
        for (auto ich = 0; ich < ICH; ich++) {
          for (auto s_fh = 0; s_fh < c_fh; s_fh++) {
            for (auto s_fw = 0; s_fw < c_fw; s_fw++) {
              auto s_index = c_fh*c_fw - 1 - s_fh * c_fw - s_fw;
              auto s_ich_iter = ich / c_ich_ops;
              auto s_ich_ops = ich % c_ich_ops;
              auto s_ops = s_ich_ops * c_ops + och % c_ops;
              auto s_och_iter = och / c_ops;
              auto s_och_ops = och % c_ops;
              auto s_ch = s_ich_iter * OCH/c_ops + s_och_iter;
              std::cout << std::setprecision(10) << "s_weights[" << s_index << "][" << och << "][" << ich << "][" << s_ops << "] = " << (float)(din[s_index][s_ch][s_ops]) << std::endl;
            }
          }
        }
      }
    #endif
  #endif

  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
    for (auto s_ch = 0; s_ch < c_ch; s_ch++) {
#pragma HLS pipeline
      for (auto s_index = 0; s_index < FSZ; s_index++) {
        dout_t s_output;
        for (auto s_ich_ops = 0; s_ich_ops < c_ich_ops; s_ich_ops++) {
          for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
            auto s_ops_index = s_ich_ops * c_ops + s_ops;
            s_output[s_ich_ops][s_ops] = din[s_index][s_ch][s_ops_index];
          }
        }
        o_data[s_index].write(s_output);
      }
    }
  }
  #ifndef __SYNTHESIS__
    std::cout << "end produce_stream" << std::endl;
  #endif
}

template <typename din_t, typename din_stream_t, typename dout_t, int ICH, int OCH, int OW, int OH,
          int c_fw, int c_fh, int c_ops, int c_ich_ops, int c_reuse>
void produce_stream(din_t din[c_fh * c_fw][OCH * ICH / (c_ops*c_ich_ops)][c_ops*c_ich_ops],
                    hls::stream<din_t> i_data[c_fh*c_fw],
                    bool &s_init,
                    hls::stream<dout_t> o_data[c_fh * c_fw]) {
  constexpr unsigned FSZ = c_fh * c_fw;
  constexpr unsigned c_ch = ICH * OCH / (c_ops*c_ich_ops);
  constexpr unsigned c_o_index = OH * OW / c_reuse;

  #ifndef __SYNTHESIS__
    std::cout << "produce_stream " << FSZ << " " << c_ch << " " << c_ich_ops << " " << c_ops << std::endl;
  #endif
  if (!s_init) {
  STORE_WEIGHTS_LOOP:
    for (auto s_ch = 0; (s_ch < c_ch); s_ch++) {
      for (auto s_index = 0; s_index < FSZ; s_index++) {
        for (auto s_ops = 0; s_ops < c_ops*c_ich_ops; s_ops++) {
          #pragma HLS pipeline off
          din[s_index][s_ch][s_ops] = i_data[s_index].read();
        }
      }
    }
  }
  s_init = true;

  #ifndef __SYNTHESIS__
    #ifndef DEBUG_WEIGHTS
      std::cout << "finished init" << std::endl;
    #endif
  #endif

  #ifndef __SYNTHESIS__
    #ifdef DEBUG_WEIGHTS
      for (auto och = 0; och < OCH; och++) {
        for (auto ich = 0; ich < ICH; ich++) {
          for (auto s_fh = 0; s_fh < c_fh; s_fh++) {
            for (auto s_fw = 0; s_fw < c_fw; s_fw++) {
              auto s_index = c_fh*c_fw - 1 - s_fh * c_fw - s_fw;
              auto s_ich_iter = ich / c_ich_ops;
              auto s_ich_ops = ich % c_ich_ops;
              auto s_ops = s_ich_ops * c_ops + och % c_ops;
              auto s_och_iter = och / c_ops;
              auto s_och_ops = och % c_ops;
              auto s_ch = s_ich_iter * OCH/c_ops + s_och_iter;
              std::cout << "s_weights[" << s_index << "][" << och << "][" << ich << "][" << s_ops << "] = " << (float)(din[s_index][s_ch][s_ops]) << std::endl;
            }
          }
        }
      }
    #endif
  #endif

  SHIFT_WEIGHTS_LOOP:
  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
    for (auto s_ch = 0; s_ch < c_ch; s_ch++) {
#pragma HLS pipeline II=1
      for (auto s_index = 0; s_index < FSZ; s_index++) {
        dout_t s_output;
        for (auto s_ich_ops = 0; s_ich_ops < c_ich_ops; s_ich_ops++) {
          for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
            auto s_ops_index = s_ich_ops * c_ops + s_ops;
            s_output[s_ich_ops][s_ops] = din[s_index][s_ch][s_ops_index];
          }
        }
        o_data[s_index].write(s_output);
      }
    }
  }
  
  #ifndef __SYNTHESIS__
    std::cout << "end produce_stream" << std::endl;
  #endif
}

template <typename din_t, typename dout_tmp_t, typename dout_t, int DIM, 
          int INDEX, int BYTES, int BITS, int PACK, int OPS>
void produce_stream(hls::stream<din_t> &din,
                    bool init,
                    hls::stream<dout_tmp_t> dout[INDEX]) {
#pragma HLS inline
  const auto ITER = DIM / (INDEX * OPS * PACK);
  dout_tmp_t tmp = 0;
  din_t tmp_din;
#ifndef __SYNTHESIS__
  std::cout << "load_weights produce_stream " << ITER << " " << INDEX << " " << BYTES << " " << BITS << " " << PACK << " " << OPS << std::endl;
#endif
  if (!init) {
    for (auto i = 0; i < ITER; i++) {
      for (auto k = 0; k < INDEX; k++) {
        for (auto c = 0; c < OPS; c++) {
          for (auto j = 0; j < BYTES; j++) {
            for (auto m = 0; m < PACK; m++) {
#pragma HLS pipeline II=1
              if (j==0)
                  tmp = 0;
              if (m==0) {
                  tmp_din = din.read();
              }
              tmp <<= BITS;
              tmp.range(BITS-1, 0) = tmp_din.data.range(BITS*(m+1)-1, BITS*m);
              if (j==(BYTES-1)) {
                dout[k] << dout_tmp_t(tmp);
                #ifndef __SYNTHESIS__
                  #ifdef DEBUG_WEIGHTS
                    std::cout << "produce_stream[" << k << "] = " << tmp << std::endl;
                  #endif
                #endif
              }
            }
          }
        }
      }
    }
  }
#ifndef __SYNTHESIS__
  std::cout << "end load_weights produce_stream" << std::endl;
#endif
}

/* Function to transform an AXI stream to a hls::stream. */
template<typename din_t,  // input data type
         typename dout_t, // output weight data type
         size_t CYCLES>   // cycles
void
axi_to_stream(hls::stream<din_t>& in, bool& s_init, hls::stream<dout_t> out[1])
{
#pragma HLS inline
#ifndef __SYNTHESIS__
  std::cout << "INFO: Call to axi_to_stream" << std::endl;
  std::cout << "\t\tCYCLES: " << CYCLES << std::endl;
#endif

  if (!s_init) {
    for (auto i = 0; i < CYCLES; i++) {
#pragma HLS pipeline II = 1
      out[0].write(in.read().data);
    }
  }
  s_init = true;

#ifndef __SYNTHESIS__
  std::cout << "INFO: Finished axi_to_stream" << std::endl;
#endif
}

template<typename din_bias_t,       // input bias data type
         typename din_weight_t,     // input weight data type
         typename din_bias1x1_t,    // input bias1x1 data type
         typename din_weight1x1_t,  // input weight1x1 data type
         typename dout_bias_t,      // output bias data type
         typename dout_weight_t,    // output weight data type
         typename dout_bias1x1_t,   // output bias1x1 data type
         typename dout_weight1x1_t, // output weight1x1 data type
         size_t ICH,                // input channels
         size_t OCH,                // output channels
         size_t OW,                 // output width
         size_t OH,                 // output height
         size_t FW,                 // filter width
         size_t FH,                 // filter height
         size_t OCH_OPS,            // output channel operations
         size_t ICH_OPS,            // input channel operations
         size_t BIAS_OPS,           // bias operations
         size_t REUSE>              // reuse factor
void
produce_stream(
  din_bias_t mem_bias[OCH / BIAS_OPS][BIAS_OPS],
  din_weight_t mem_weights[FH * FW][OCH * ICH / (OCH_OPS * ICH_OPS)]
                          [OCH_OPS * ICH_OPS],
  din_bias1x1_t mem_bias_1x1[OCH / BIAS_OPS][BIAS_OPS],
  din_weight1x1_t mem_weights_1x1[1][OCH * ICH / (OCH_OPS * ICH_OPS)]
                                 [OCH_OPS * ICH_OPS],
  hls::stream<dout_bias_t> o_bias[1],
  hls::stream<dout_weight_t> o_weights[FH * FW],
  hls::stream<dout_bias1x1_t> o_bias_1x1[1],
  hls::stream<dout_weight1x1_t> o_weights_1x1[1])
{
  #pragma HLS dataflow
  
  constexpr unsigned FSZ = FH * FW;
  constexpr unsigned c_ch_weight = ICH * OCH / (OCH_OPS * ICH_OPS);
  constexpr unsigned c_ch_bias = OCH / BIAS_OPS;
  constexpr unsigned c_o_index = OH * OW / REUSE;

  /* Streaming in ouput weights */
STREAM_WEIGHTS_LOOP:
  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
    for (auto s_ch = 0; s_ch < c_ch_weight; s_ch++) {
#pragma HLS pipeline II = 1
      for (auto s_index = 0; s_index < FSZ; s_index++) {
        dout_weight_t s_output;
        for (auto s_ich_ops = 0; s_ich_ops < ICH_OPS; s_ich_ops++) {
          for (auto s_ops = 0; s_ops < OCH_OPS; s_ops++) {
            auto s_ops_index = s_ich_ops * OCH_OPS + s_ops;
            s_output[s_ich_ops][s_ops] = mem_weights[s_index][s_ch][s_ops_index];
          }
        }
        o_weights[s_index].write(s_output);
      }
    }
  }

  if constexpr (std::is_same<din_bias_t, std::nullptr_t>::value == false) {
    /* Streaming in ouput biases */
  STREAM_BIAS_LOOP:
    for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
      for (auto s_ch = 0; s_ch < c_ch_bias; s_ch++) {
#pragma HLS pipeline II = 1
        dout_bias_t s_output;
        for (auto s_ops = 0; s_ops < BIAS_OPS; s_ops++) {
          s_output[0][s_ops] = mem_bias[s_ch][s_ops];
        }
        o_bias[0].write(s_output);
      }
    }
  }

  if constexpr (std::is_same<din_weight1x1_t, std::nullptr_t>::value == false) {
    /* Streaming in ouput weights 1x1 */
  STREAM_WEIGHTS_1x1_LOOP:
    for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
      for (auto s_ch = 0; s_ch < c_ch_weight; s_ch++) {
#pragma HLS pipeline II = 1
        dout_weight1x1_t s_output;
        for (auto s_ich_ops = 0; s_ich_ops < ICH_OPS; s_ich_ops++) {
          for (auto s_ops = 0; s_ops < OCH_OPS; s_ops++) {
            auto s_ops_index = s_ich_ops * OCH_OPS + s_ops;
            s_output[s_ich_ops][s_ops] =
              mem_weights_1x1[0][s_ch][s_ops_index];
          }
        }
        o_weights_1x1[0].write(s_output);
      }
    }
  }

  if constexpr (std::is_same<din_bias1x1_t, std::nullptr_t>::value == false) {
    /* Streaming in ouput biases 1x1 */
  STREAM_BIAS_1x1_LOOP:
    for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
      for (auto s_ch = 0; s_ch < c_ch_bias; s_ch++) {
#pragma HLS pipeline II = 1
        dout_bias1x1_t s_output;
        for (auto s_ops = 0; s_ops < BIAS_OPS; s_ops++) {
          s_output[0][s_ops] = mem_bias_1x1[s_ch][s_ops];
        }
        o_bias_1x1[0].write(s_output);
      }
    }
  }
}

/* Function to read, store and shift biases and weights. Must support cases in
 * which the stream is larger than the data (PACK > 1) and viceversa (READ
 * > 1).
 * TODO: Support more than one data per read, right now only one data or part of
 * it.
 */
template<typename din_bias_t,       // input bias data type
         typename din_weight_t,     // input weight data type
         typename din_bias1x1_t,    // input bias1x1 data type
         typename din_weight1x1_t,  // input weight1x1 data type
         typename p_stream_t,       // input stream data type
         typename dout_bias_t,      // output bias data type
         typename dout_weight_t,    // output weight data type
         typename dout_bias1x1_t,   // output bias1x1 data type
         typename dout_weight1x1_t, // output weight1x1 data type
         size_t ICH,                // input channels
         size_t OCH,                // output channels
         size_t OW,                 // output width
         size_t OH,                 // output height
         size_t FW,                 // filter width
         size_t FH,                 // filter height
         size_t OCH_OPS,            // output channel operations
         size_t ICH_OPS,            // input channel operations
         size_t BIAS_OPS,           // bias operations
         size_t REUSE,              // reuse factor
         size_t DATA_TO_SHIFT,      // data to shift
         size_t STREAM_BITS,        // stream bits
         size_t READ_BIAS,          // number of read to have one original bias
         size_t READ_WEIGHT> // number of read to have one original weight
void
produce_shift_stream(
  din_bias_t mem_bias[OCH / BIAS_OPS][BIAS_OPS],
  din_weight_t mem_weights[FH * FW][OCH * ICH / (OCH_OPS * ICH_OPS)]
                          [OCH_OPS * ICH_OPS],
  din_bias1x1_t mem_bias_1x1[OCH / BIAS_OPS][BIAS_OPS],
  din_weight1x1_t mem_weights_1x1[1][OCH * ICH / (OCH_OPS * ICH_OPS)]
                                 [OCH_OPS * ICH_OPS],
  hls::stream<p_stream_t> p_in[1],
  bool& s_init,
  hls::stream<dout_bias_t> o_bias[1],
  hls::stream<dout_weight_t> o_weights[FH * FW],
  hls::stream<dout_bias1x1_t> o_bias_1x1[1],
  hls::stream<dout_weight1x1_t> o_weights_1x1[1],
  hls::stream<p_stream_t> p_out[1])
{

  constexpr unsigned FSZ = FH * FW;
  constexpr unsigned c_ch_weight = ICH * OCH / (OCH_OPS * ICH_OPS);

#ifndef __SYNTHESIS__
  std::cout << "INFO: Call to produce_shift_stream" << std::endl;
  std::cout << "\t\tICH: " << ICH << std::endl;
  std::cout << "\t\tOCH: " << OCH << std::endl;
  std::cout << "\t\tOW: " << OW << std::endl;
  std::cout << "\t\tOH: " << OH << std::endl;
  std::cout << "\t\tFW: " << FW << std::endl;
  std::cout << "\t\tFH: " << FH << std::endl;
  std::cout << "\t\tOCH_OPS: " << OCH_OPS << std::endl;
  std::cout << "\t\tICH_OPS: " << ICH_OPS << std::endl;
  std::cout << "\t\tBIAS_OPS: " << BIAS_OPS << std::endl;
  std::cout << "\t\tREUSE: " << REUSE << std::endl;
  std::cout << "\t\tDATA_TO_SHIFT: " << DATA_TO_SHIFT << std::endl;
  std::cout << "\t\tSTREAM_BITS: " << STREAM_BITS << std::endl;
  std::cout << "\t\tREAD_BIAS: " << READ_BIAS << std::endl;
  std::cout << "\t\tREAD_WEIGHT: " << READ_WEIGHT << std::endl;
#endif

  p_stream_t packed_data;

  if (!s_init) {

    /* Storing weights */
  STORE_WEIGHT_LOOP:
    for (auto s_ch = 0; s_ch < c_ch_weight; s_ch++) {
      for (auto s_index = 0; s_index < FSZ; s_index++) {
        for (auto s_ops = 0; s_ops < OCH_OPS * ICH_OPS; s_ops++) {
#pragma HLS pipeline off
          ap_uint<READ_WEIGHT * STREAM_BITS> unpacked_data = 0;
          for (auto s_read = 0; s_read < READ_WEIGHT; s_read++) {
#pragma HLS pipeline off

            packed_data = p_in[0].read();
            unpacked_data <<= STREAM_BITS;
            unpacked_data.range(STREAM_BITS - 1, 0) =
              packed_data.range(STREAM_BITS - 1, 0);
          }
          
          /* Step needed to support also 4 bit weights inside a 8 bit word*/
          din_weight_t tmp;
          tmp.range(din_weight_t::width - 1, 0) =
            unpacked_data.range(din_weight_t::width - 1, 0);
          mem_weights[s_index][s_ch][s_ops] = tmp;
        }
      }
    }

    if constexpr (std::is_same<din_bias_t, std::nullptr_t>::value == false) {

      /* Storing biases */
      const size_t c_loops_bias = OCH / BIAS_OPS;
    STORE_BIAS_LOOP:
      for (auto s_ch = 0; s_ch < c_loops_bias; s_ch++) {
        for (auto s_ops = 0; s_ops < BIAS_OPS; s_ops++) {
#pragma HLS pipeline off
          ap_uint<READ_BIAS * STREAM_BITS> unpacked_data = 0;
          for (auto s_read = 0; s_read < READ_BIAS; s_read++) {
#pragma HLS pipeline off

            packed_data = p_in[0].read();
            unpacked_data <<= STREAM_BITS;
            unpacked_data.range(STREAM_BITS - 1, 0) =
              packed_data.range(STREAM_BITS - 1, 0);
          }
          din_bias_t tmp;
          tmp.range(din_bias_t::width - 1, 0) =
            unpacked_data.range(din_bias_t::width - 1, 0);
          mem_bias[s_ch][s_ops] = tmp;
        }
      }
    }

    if constexpr (std::is_same<din_weight1x1_t, std::nullptr_t>::value ==
                  false) {

      /* Storing weights 1x1 */
      for (auto s_ch = 0; s_ch < c_ch_weight; s_ch++) {
        for (auto s_ops = 0; s_ops < OCH_OPS * ICH_OPS; s_ops++) {
#pragma HLS pipeline off
          ap_uint<READ_WEIGHT * STREAM_BITS> unpacked_data = 0;
          for (auto s_read = 0; s_read < READ_WEIGHT; s_read++) {
#pragma HLS pipeline off

            packed_data = p_in[0].read();
            unpacked_data <<= STREAM_BITS;
            unpacked_data.range(STREAM_BITS - 1, 0) =
              packed_data.range(STREAM_BITS - 1, 0);
          }
          din_weight1x1_t tmp;
          tmp.range(din_weight1x1_t::width - 1, 0) =
            unpacked_data.range(din_weight1x1_t::width - 1, 0);
          mem_weights_1x1[0][s_ch][s_ops] = tmp;
        }
      }
    }

    if constexpr (std::is_same<din_bias1x1_t, std::nullptr_t>::value == false) {

      /* Storing biases 1x1 */
      const size_t c_loops_bias_1x1 = OCH / BIAS_OPS;
      for (auto s_ch = 0; s_ch < c_loops_bias_1x1; s_ch++) {
        for (auto s_ops = 0; s_ops < BIAS_OPS; s_ops++) {
#pragma HLS pipeline off
          ap_uint<READ_BIAS * STREAM_BITS> unpacked_data = 0;
          for (auto s_read = 0; s_read < READ_BIAS; s_read++) {
#pragma HLS pipeline off

            packed_data = p_in[0].read();
            unpacked_data <<= STREAM_BITS;
            unpacked_data.range(STREAM_BITS - 1, 0) =
              packed_data.range(STREAM_BITS - 1, 0);
          }
          din_bias1x1_t tmp;
          tmp.range(din_bias1x1_t::width - 1, 0) =
            unpacked_data.range(din_bias1x1_t::width - 1, 0);
          mem_bias_1x1[s_ch][s_ops] = tmp;
        }
      }
    }

    /* Shift remaining parameters to following layers */
  SHIFT_LOOP:
    for (auto p_left = 0; p_left < DATA_TO_SHIFT; p_left++) {
      p_out[0].write(p_in[0].read());
    }

#ifndef __SYNTHESIS__
#ifndef SKIP_ASSERTIONS
    std::cout << "\tINFO: Finished saving parameters." << std::endl;
    /* Check that all the input streams are empty */
    if (p_in[0].size() > 0) {
      std::cout << "\tERROR: Not empty input stream. p_in[0].size() = "
                << p_in[0].size() << std::endl;
    }
    assert(p_in[0].size() == 0);

    /* Check that all the output streams are not empty */
    if (p_out[0].size() != DATA_TO_SHIFT) {
      std::cout << "\tERROR: Not full output stream. p_out.size() = "
                << p_out[0].size() << std::endl;
    }
    assert(p_out[0].size() == DATA_TO_SHIFT);
#endif /* SKIP_ASSERTIONS */
#endif
  }

  s_init = true;

  produce_stream<din_bias_t,
                 din_weight_t,
                 din_bias1x1_t,
                 din_weight1x1_t,
                 dout_bias_t,
                 dout_weight_t,
                 dout_bias1x1_t,
                 dout_weight1x1_t,
                 ICH,
                 OCH,
                 OW,
                 OH,
                 FW,
                 FH,
                 OCH_OPS,
                 ICH_OPS,
                 BIAS_OPS,
                 REUSE>(mem_bias,
                        mem_weights,
                        mem_bias_1x1,
                        mem_weights_1x1,
                        o_bias,
                        o_weights,
                        o_bias_1x1,
                        o_weights_1x1);

#ifndef __SYNTHESIS__
    std::cout << "\tINFO: Finished streaming parameters." << std::endl;
#endif
}

}  // namespace nn2fpga

#endif  // NN2FPGA_WEIGHTS_UTILS_H_
