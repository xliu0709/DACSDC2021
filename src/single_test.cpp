#include "config.h"
#include "conv1x1DSP2.hpp"
#include "conv2d.h"
#include "conv2d_DSPopt.hpp"
#include "conv2d_l0.hpp"
#include "debug.hpp"
#include "param.h"
#include "weight3.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_video.h>
#include <stdint.h>
#define IN_IMAGE_WIDTH 640
#define IN_IMAGE_HEIGHT 360

#define RESIZE_IMAGE_WIDTH 320
#define RESIZE_IMAGE_HEIGHT 160

// void conv3x3_bn_act_DSPopt_hls_wrapper(
// stream<ap_uint<CONV_8_IN_BIT * CONV_8_INPE>> &in,
// const ap_uint<CONV_8_SIMD_DSP6 * CONV_8_W_BIT>
//     weights[CONV_8_PE][3][((CONV_8_IFM_CH * 3) / CONV_8_SIMD_DSP6) *
//                           (CONV_8_OFM_CH / CONV_8_PE)],
// const ap_int<CONV_8_INC_BIT> inc[CONV_8_PE][CONV_8_OFM_CH / CONV_8_PE],
// const ap_int<CONV_8_BIAS_BIT> bias[CONV_8_PE][CONV_8_OFM_CH / CONV_8_PE],
// stream<ap_uint<CONV_8_OUT_BIT * CONV_8_PE_DSP2 * 2>> &out) {

// #pragma HLS array_partition variable = conv_0_w_dspopt dim = 1 complete
// #pragma HLS array_partition variable = conv_0_w_dspopt dim = 2 complete
// #pragma HLS array_partition variable = conv_0_inc dim = 1 complete
// #pragma HLS array_partition variable = conv_0_bias dim = 1 complete
//   conv3x3_l0_bn_act_DSPopt<CONV_8_IFM_ROW, CONV_8_IFM_COL, CONV_8_IFM_CH,
//                            CONV_8_IN_BIT, CONV_8_OFM_CH, CONV_8_OUT_BIT,
//                            CONV_8_W_BIT, 21, CONV_8_INC_BIT,
//                            CONV_8_BIAS_BIT, CONV_8_SIMD_DSP6, 3,
//                            CONV_8_INPE, CONV_8_PE_DSP6, CONV_8_L_SHIFT>(in,
//                            conv_0_w_dspopt, conv_0_inc,
//                                            conv_0_bias, out);
// }
// void conv3x3_bn_act_hls_wrapper(
// stream<ap_uint<CONV_8_IN_BIT * CONV_8_IFM_CH>> &in,
// const ap_uint<CONV_8_SIMD_DSP6 * CONV_8_W_BIT>
//     weights[CONV_8_PE][3][((CONV_8_IFM_CH * 3) / CONV_8_SIMD_DSP6) *
//                           (CONV_8_OFM_CH / CONV_8_PE)],
// const ap_int<CONV_8_INC_BIT> inc[CONV_8_PE][CONV_8_OFM_CH / CONV_8_PE],
// const ap_int<CONV_8_BIAS_BIT> bias[CONV_8_PE][CONV_8_OFM_CH / CONV_8_PE],
// stream<ap_uint<CONV_8_OUT_BIT * CONV_8_OFM_CH>> &out) {

// #pragma HLS array_partition variable = conv_0_w dim = 1 complete
// #pragma HLS array_partition variable = conv_0_inc dim = 1 complete
// #pragma HLS array_partition variable = conv_0_bias dim = 1 complete

// conv3x3_bn_act<CONV_8_IFM_ROW, CONV_8_IFM_COL, CONV_8_IFM_CH,
// CONV_8_IN_BIT,

//                CONV_8_OFM_CH, CONV_8_OUT_BIT,

//                CONV_8_W_BIT, 32, CONV_8_INC_BIT, CONV_8_BIAS_BIT,

//                CONV_8_SIMD, CONV_8_PE, CONV_8_L_SHIFT>(
//     in, conv_0_w, conv_0_inc, conv_0_bias, out, 1);
// }
void conv1x1_dsp2_hls_wrapper(
    stream<ap_uint<CONV_8_IN_BIT * CONV_8_INPE * 2>> &in,
    // const ap_uint<CONV_8_SIMD_DSP6 * CONV_8_W_BIT>
    //     weights[CONV_8_PE][3][((CONV_8_IFM_CH * 3) / CONV_8_SIMD_DSP6) *
    //                           (CONV_8_OFM_CH / CONV_8_PE)],
    // const ap_int<CONV_8_INC_BIT> inc[CONV_8_PE][CONV_8_OFM_CH / CONV_8_PE],
    // const ap_int<CONV_8_BIAS_BIT> bias[CONV_8_PE][CONV_8_OFM_CH / CONV_8_PE],
    stream<ap_uint<32 * CONV_8_PE_DSP2>> &out) {

#pragma HLS array_partition variable = conv_8_w dim = 1 complete
#pragma HLS array_partition variable = conv_8_w dim = 2 complete

  conv1x1_DSPopt<CONV_8_IFM_ROW, CONV_8_IFM_COL, CONV_8_IFM_CH, CONV_8_IN_BIT,
                 CONV_8_OFM_CH, CONV_8_W_BIT, 32, CONV_8_SIMD_DSP2,
                 CONV_8_PE_DSP2, CONV_8_INPE>(in, conv_8_w_dspopt, out);
}

void conv1x1_hls_wrapper(
    stream<ap_uint<CONV_8_IN_BIT * CONV_8_SIMD>> &in,
    // const ap_uint<CONV_8_SIMD_DSP6 * CONV_8_W_BIT>
    //     weights[CONV_8_PE][3][((CONV_8_IFM_CH * 3) / CONV_8_SIMD_DSP6) *
    //                           (CONV_8_OFM_CH / CONV_8_PE)],
    // const ap_int<CONV_8_INC_BIT> inc[CONV_8_PE][CONV_8_OFM_CH / CONV_8_PE],
    // const ap_int<CONV_8_BIAS_BIT> bias[CONV_8_PE][CONV_8_OFM_CH / CONV_8_PE],
    stream<ap_uint<32 * CONV_8_PE>> &out) {

#pragma HLS array_partition variable = conv_8_w dim = 1 complete
#pragma HLS array_partition variable = conv_8_w dim = 2 complete

  conv1x1<CONV_8_IFM_ROW, CONV_8_IFM_COL, CONV_8_IFM_CH, CONV_8_IN_BIT,
          CONV_8_OFM_CH, CONV_8_W_BIT, 32, CONV_8_SIMD, CONV_8_PE>(in, conv_8_w,
                                                                   out);
}

template <unsigned K, unsigned IN_CH, unsigned OUT_CH, unsigned PE,
          unsigned SIMD, unsigned W_BIT>
void initialziation(
    ap_uint<SIMD * W_BIT> weights[PE][K][K * OUT_CH / PE * IN_CH / SIMD],
    string method) {

  for (int kr = 0; kr < K; kr++)
    for (int i = 0; i < IN_CH; i += SIMD)
      for (int o = 0; o < OUT_CH; o += PE)
        for (int kc = 0; kc < K; kc++)
          for (int p = 0; p < PE; p++) {
            ap_uint<SIMD * W_BIT> data;
            for (int s = 0; s < SIMD; s++) {
              if (method == "odepth") {
                data((s + 1) * W_BIT - 1, s * W_BIT) = p;
              } else if (method == "kernel") {
                data((s + 1) * W_BIT - 1, s * W_BIT) = kr * K + kc;
              } else {
                if (kr == K / 2 && kc == K / 2)
                  data((s + 1) * W_BIT - 1, s * W_BIT) = 1;
                else
                  data((s + 1) * W_BIT - 1, s * W_BIT) = 0;
              }
            }
            weights[p][kc][o / PE * K * IN_CH / SIMD + kr * IN_CH / SIMD +
                           i / SIMD] = data;
          }
}

template <unsigned IN_CH, unsigned OUT_CH, unsigned PE, unsigned SIMD,
          unsigned W_BIT>
void initialziation1x1(
    ap_uint<SIMD * W_BIT> weights[PE][OUT_CH / PE * IN_CH / SIMD],
    string method) {

  for (int i = 0; i < IN_CH; i += SIMD)
    for (int o = 0; o < OUT_CH; o += PE)
      for (int p = 0; p < PE; p++) {
        ap_uint<SIMD * W_BIT> data;
        for (int s = 0; s < SIMD; s++) {
          if (method == "odepth") {
            data((s + 1) * W_BIT - 1, s * W_BIT) = (ap_int<W_BIT>)(o + p);
          } else if (method == "kernel") {
            data((s + 1) * W_BIT - 1, s * W_BIT) = 1;
          }
        }
        weights[p][o / PE * IN_CH / SIMD + i / SIMD] = data;
      }
}

int main(int argc, char **argv) {

  hls::stream<ap_uint<CONV_8_IN_BIT * CONV_8_SIMD>> golden_in("golden_in");
  hls::stream<ap_uint<CONV_8_IN_BIT * CONV_8_INPE * 2>> test_in("test_in");

  ap_uint<CONV_8_IN_BIT> IFM[CONV_8_IFM_CH][CONV_8_IFM_ROW][CONV_8_IFM_COL];
  for (int i = 0; i < CONV_8_IFM_CH; i++) {
    for (int r = 0; r < CONV_8_IFM_ROW; r++)
      for (int c = 0; c < CONV_8_IFM_COL; c++) {

        IFM[0][r][c] = random();
      }
  }

  for (int r = 0; r < CONV_8_IFM_ROW; r++) {
    for (int i = 0; i < CONV_8_IFM_CH; i += CONV_8_INPE) {
      for (int c = 0; c < CONV_8_IFM_COL; c += 2) {
        ap_uint<CONV_8_IN_BIT * CONV_8_INPE> data0;
        ap_uint<CONV_8_IN_BIT * CONV_8_INPE> data1;
        for (int s = 0; s < CONV_8_INPE; s++) {
          data0((s + 1) * CONV_8_IN_BIT - 1, s * CONV_8_IN_BIT) =
              IFM[i + s][r][c];
          data1((s + 1) * CONV_8_IN_BIT - 1, s * CONV_8_IN_BIT) =
              IFM[i + s][r][c + 1];
        }
        test_in << (data1, data0);
      }
    }
  }

  for (int r = 0; r < CONV_8_IFM_ROW; r++) {
    for (int c = 0; c < CONV_8_IFM_COL; c++) {

      for (int i = 0; i < CONV_8_IFM_CH; i += CONV_8_SIMD) {
        ap_uint<CONV_8_IN_BIT * CONV_8_SIMD> data;
        for (int s = 0; s < CONV_8_SIMD; s++) {
          data((s + 1) * CONV_8_IN_BIT - 1, s * CONV_8_IN_BIT) =
              IFM[i + s][r][c];
        }
        golden_in << data;
      }
    }
  }
  // test_in << data;
  // initialziation1x1<CONV_8_IFM_CH, CONV_8_OFM_CH, CONV_8_PE_DSP2,
  //                   CONV_8_SIMD_DSP2, CONV_8_W_BIT>(conv_8_w_dspopt,
  //                   "odepth");

  hls::stream<ap_uint<32 * CONV_8_PE>> golden_out("golden_out");

  conv1x1_hls_wrapper(golden_in, golden_out);

  print_pe_stream_through<CONV_8_OFM_ROW, CONV_8_OFM_COL, CONV_8_OFM_CH,
                          CONV_8_PE, 32>(golden_out, "conv_ultranet_out.txt");

  hls::stream<ap_uint<32 * CONV_8_PE_DSP2>> test_out("test_out");

  conv1x1_dsp2_hls_wrapper(test_in, test_out);

  print_pe_stream_through<CONV_8_OFM_ROW, CONV_8_OFM_COL, CONV_8_OFM_CH,
                          CONV_8_PE, 32>(test_out, "conv_DSP2_out.txt");
}
