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

template <typename din_t, typename din_stream_t, typename dout_t, int ICH, int OCH, int OW, int OH,
          int c_fw, int c_fh, int c_ops, int c_ich_ops, int c_reuse>
void produce_stream_II2(din_t din[c_fh * c_fw][OCH * ICH / (c_ops*c_ich_ops)][c_ops*c_ich_ops],
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
    for (auto s_ch = 0; (s_ch < c_ch); s_ch++) {
      for (auto s_index = 0; s_index < FSZ; s_index++) {
        for (auto s_ops = 0; s_ops < c_ops*c_ich_ops; s_ops++) {
          // #pragma HLS pipeline off
          din[s_index][s_ch][s_ops] = i_data[s_index].read();
        }
      }
    }
  }
  s_init = true;

  #ifndef __SYNTHESIS__
    std::cout << "finished init" << std::endl;
  #endif

  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
    for (auto s_ch = 0; s_ch < c_ch; s_ch++) {
#pragma HLS pipeline II=2
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

template<typename din_t,
         typename w_in_stream_t,
         typename w_out_stream_t,
         typename dout_t,
         int ICH,
         int OCH,
         int OW,
         int OH,
         int c_fw,
         int c_fh,
         int c_ops,
         int c_ich_ops,
         int c_reuse,
         int BYTES,
         int BITS,
         int PACK>
void
produce_shift_stream(
  din_t mem[c_fh * c_fw][OCH * ICH / (c_ops * c_ich_ops)][c_ops * c_ich_ops],
  hls::stream<w_in_stream_t>& w_in,
  bool& s_init,
  hls::stream<dout_t> o_data[c_fh * c_fw],
  hls::stream<w_out_stream_t>& w_out)
{

  constexpr unsigned FSZ = c_fh * c_fw;
  constexpr unsigned c_ch = ICH * OCH / (c_ops * c_ich_ops);
  constexpr unsigned c_o_index = OH * OW / c_reuse;

#ifndef __SYNTHESIS__
  std::cout << "produce_stream " << FSZ << " " << c_ch << " " << c_ich_ops
            << " " << c_ops << std::endl;
#endif

  din_t tmp = 0;
  w_in_stream_t tmp_din;

  if (!s_init) {
    for (auto s_ch = 0; (s_ch < c_ch); s_ch++) {
      for (auto s_index = 0; s_index < FSZ; s_index++) {
        for (auto s_ops = 0; s_ops < c_ops * c_ich_ops; s_ops++) {
          for (auto j = 0; j < BYTES; j++) {
            for (auto m = 0; m < PACK; m++) {
#pragma HLS pipeline II = 1
              if ((j == 0) && (m == 0))
                tmp = 0;
              if (m == 0) {
                tmp_din = w_in.read();
              }
              tmp <<= BITS;
              tmp.range(BITS - 1, 0) =
                tmp_din.data.range(BITS * (m + 1) - 1, BITS * m);
              if (j == (BYTES - 1)) {
                mem[s_index][s_ch][s_ops] = dout_tmp_t(tmp);
              }
            }
          }
        }
      }
    }
    
    /* Shift remaining weights to other layers */
    while(!tmp_din.last){
      tmp_din = w_in.read();
      w_out.write(tmp_din);
    }

  }
  s_init = true;

#ifndef __SYNTHESIS__
    std::cout << "finished init" << std::endl;
  #endif

  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
    for (auto s_ch = 0; s_ch < c_ch; s_ch++) {
#pragma HLS pipeline II=1
      for (auto s_index = 0; s_index < FSZ; s_index++) {
        dout_t s_output;
        for (auto s_ich_ops = 0; s_ich_ops < c_ich_ops; s_ich_ops++) {
          for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
            auto s_ops_index = s_ich_ops * c_ops + s_ops;
            s_output[s_ich_ops][s_ops] = mem[s_index][s_ch][s_ops_index];
          }
        }
        o_data[s_index].write(s_output);
      }
    }
  }
}

template<typename din_t,
         typename w_in_stream_t,
         typename dout_t,
         int ICH,
         int OCH,
         int OW,
         int OH,
         int c_fw,
         int c_fh,
         int c_ops,
         int c_ich_ops,
         int c_reuse,
         int BYTES,
         int BITS,
         int PACK>
void
produce_shift_stream_end(
  din_t mem[c_fh * c_fw][OCH * ICH / (c_ops * c_ich_ops)][c_ops * c_ich_ops],
  hls::stream<w_in_stream_t>& w_in,
  bool& s_init,
  hls::stream<dout_t> o_data[c_fh * c_fw])
{

  constexpr unsigned FSZ = c_fh * c_fw;
  constexpr unsigned c_ch = ICH * OCH / (c_ops * c_ich_ops);
  constexpr unsigned c_o_index = OH * OW / c_reuse;

#ifndef __SYNTHESIS__
  std::cout << "produce_stream " << FSZ << " " << c_ch << " " << c_ich_ops
            << " " << c_ops << std::endl;
#endif

  din_t tmp = 0;
  w_in_stream_t tmp_din;

  if (!s_init) {
    for (auto s_ch = 0; (s_ch < c_ch); s_ch++) {
      for (auto s_index = 0; s_index < FSZ; s_index++) {
        for (auto s_ops = 0; s_ops < c_ops * c_ich_ops; s_ops++) {
          for (auto j = 0; j < BYTES; j++) {
            for (auto m = 0; m < PACK; m++) {
#pragma HLS pipeline II = 1
              if ((j == 0) && (m == 0))
                tmp = 0;
              if (m == 0) {
                tmp_din = w_in.read();
              }
              tmp <<= BITS;
              tmp.range(BITS - 1, 0) =
                tmp_din.data.range(BITS * (m + 1) - 1, BITS * m);
              if (j == (BYTES - 1)) {
                mem[s_index][s_ch][s_ops] = dout_tmp_t(tmp);
              }
            }
          }
        }
      }
    }
  }
  s_init = true;

#ifndef __SYNTHESIS__
    std::cout << "finished init" << std::endl;
  #endif

  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
    for (auto s_ch = 0; s_ch < c_ch; s_ch++) {
#pragma HLS pipeline II=1
      for (auto s_index = 0; s_index < FSZ; s_index++) {
        dout_t s_output;
        for (auto s_ich_ops = 0; s_ich_ops < c_ich_ops; s_ich_ops++) {
          for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
            auto s_ops_index = s_ich_ops * c_ops + s_ops;
            s_output[s_ich_ops][s_ops] = mem[s_index][s_ch][s_ops_index];
          }
        }
        o_data[s_index].write(s_output);
      }
    }
  }
}

}  // namespace nn2fpga

#endif  // NN2FPGA_WEIGHTS_UTILS_H_
