#include "config.h"
#include "conv2d.h"
#include "conv2d_DSPopt.hpp"
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

void conv3x3_bn_act_DSPopt_hls_wrapper(
    stream<ap_uint<CONV_2_IN_BIT * CONV_2_INPE * 2>> &in,
    // const ap_uint<CONV_2_SIMD_DSP6 * CONV_2_W_BIT>
    //     weights[CONV_2_PE][3][((CONV_2_IFM_CH * 3) / CONV_2_SIMD_DSP6) *
    //                           (CONV_2_OFM_CH / CONV_2_PE)],
    // const ap_int<CONV_2_INC_BIT> inc[CONV_2_PE][CONV_2_OFM_CH / CONV_2_PE],
    // const ap_int<CONV_2_BIAS_BIT> bias[CONV_2_PE][CONV_2_OFM_CH / CONV_2_PE],
    stream<ap_uint<CONV_2_OUT_BIT * CONV_2_PE * 2>> &out) {

#pragma HLS array_partition variable = conv_2_w_dspopt dim = 1 complete
#pragma HLS array_partition variable = conv_2_w_dspopt dim = 2 complete
#pragma HLS array_partition variable = conv_2_inc dim = 1 complete
#pragma HLS array_partition variable = conv_2_bias dim = 1 complete
  conv3x3_bn_act_DSPopt<CONV_2_IFM_ROW, CONV_2_IFM_COL, CONV_2_IFM_CH,
                        CONV_2_IN_BIT, CONV_2_OFM_CH, CONV_2_OUT_BIT,
                        CONV_2_W_BIT, 27, CONV_2_INC_BIT, CONV_2_BIAS_BIT,
                        CONV_2_SIMD_DSP6, 4, CONV_2_INPE, CONV_2_PE,
                        CONV_2_L_SHIFT>(in, conv_2_w_dspopt, conv_2_inc,
                                        conv_2_bias, out);
}

void conv3x3_bn_act_hls_wrapper(
    stream<ap_uint<CONV_2_IN_BIT * CONV_2_IFM_CH>> &in,
    // const ap_uint<CONV_2_SIMD_DSP6 * CONV_2_W_BIT>
    //     weights[CONV_2_PE][3][((CONV_2_IFM_CH * 3) / CONV_2_SIMD_DSP6) *
    //                           (CONV_2_OFM_CH / CONV_2_PE)],
    // const ap_int<CONV_2_INC_BIT> inc[CONV_2_PE][CONV_2_OFM_CH / CONV_2_PE],
    // const ap_int<CONV_2_BIAS_BIT> bias[CONV_2_PE][CONV_2_OFM_CH / CONV_2_PE],
    stream<ap_uint<CONV_2_OUT_BIT * CONV_2_OFM_CH>> &out) {

#pragma HLS array_partition variable = conv_2_w dim = 1 complete
#pragma HLS array_partition variable = conv_2_inc dim = 1 complete
#pragma HLS array_partition variable = conv_2_bias dim = 1 complete

  conv3x3_bn_act<CONV_2_IFM_ROW, CONV_2_IFM_COL, CONV_2_IFM_CH, CONV_2_IN_BIT,

                 CONV_2_OFM_CH, CONV_2_OUT_BIT,

                 CONV_2_W_BIT, 32, CONV_2_INC_BIT, CONV_2_BIAS_BIT,

                 CONV_2_SIMD, CONV_2_PE, CONV_2_L_SHIFT>(
      in, conv_2_w, conv_2_inc, conv_2_bias, out, 1);
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

int main(int argc, char **argv) {

  hls::stream<ap_uint<CONV_2_IN_BIT * CONV_2_IFM_CH>> golden_in("golden_in");
  hls::stream<ap_uint<CONV_2_IN_BIT * CONV_2_INPE * 2>> test_in("test_in");

  ap_uint<4> IFM[CONV_2_IFM_CH][CONV_2_IFM_ROW][CONV_2_IFM_COL];
  for (int i = 0; i < CONV_2_IFM_CH; i++) {
    for (int r = 0; r < CONV_2_IFM_ROW; r++)
      for (int c = 0; c < CONV_2_IFM_COL; c++) {
        IFM[i][r][c] = random();
      }
  }

  for (int r = 0; r < CONV_2_IFM_ROW; r++) {
    for (int i = 0; i < CONV_2_IFM_CH; i += CONV_2_INPE) {
      for (int c = 0; c < CONV_2_IFM_COL; c += 2) {
        ap_uint<CONV_2_IN_BIT * CONV_2_INPE> data0;
        ap_uint<CONV_2_IN_BIT * CONV_2_INPE> data1;
        for (int s = 0; s < CONV_2_INPE; s++) {
          data0((s + 1) * CONV_2_IN_BIT - 1, s * CONV_2_IN_BIT) =
              IFM[i + s][r][c];
          data1((s + 1) * CONV_2_IN_BIT - 1, s * CONV_2_IN_BIT) =
              IFM[i + s][r][c + 1];
        }
        test_in << (data1, data0);
      }
    }
  }

  for (int r = 0; r < CONV_2_IFM_ROW; r++) {
    for (int c = 0; c < CONV_2_IFM_COL; c++) {
      ap_uint<CONV_2_IN_BIT * CONV_2_IFM_CH> data;
      for (int i = 0; i < CONV_2_IFM_CH; i++) {
        data((i + 1) * CONV_2_IN_BIT - 1, i * CONV_2_IN_BIT) = IFM[i][r][c];
      }
      golden_in << data;
    }
  }

  // initialziation<CONV_2_K, CONV_2_IFM_CH, CONV_2_OFM_CH, CONV_2_PE,
  // CONV_2_SIMD,
  //                CONV_2_W_BIT>(conv_2_w_dspopt, "center");

  hls::stream<ap_uint<CONV_2_OUT_BIT * CONV_2_OFM_CH>> golden_out("golden_out");

  // conv3x3_bn_act<CONV_2_IFM_ROW, CONV_2_IFM_COL, CONV_2_IFM_CH,
  // CONV_2_IN_BIT,

  //                CONV_2_OFM_CH, CONV_2_OUT_BIT,

  //                CONV_2_W_BIT, 32, CONV_2_INC_BIT, CONV_2_BIAS_BIT,

  //                CONV_2_SIMD, CONV_2_PE, CONV_2_L_SHIFT>(
  //     golden_in, conv_2_w, conv_2_inc, conv_2_bias, golden_out, 1);
  conv3x3_bn_act_hls_wrapper(golden_in, golden_out);

  // print_mavu_stream_through<CONV_2_OFM_ROW, CONV_2_OFM_COL, CONV_2_OFM_CH,
  //                           CONV_2_PE, CONV_2_OUT_BIT>(golden_out,
  //                                                      "conv_ultranet_out.txt");

  hls::stream<ap_uint<CONV_2_OUT_BIT * CONV_2_PE * 2>> test_out("test_out");

  conv3x3_bn_act_DSPopt_hls_wrapper(test_in, test_out);

  print_mavu_DSPopt_stream_through<CONV_2_OFM_ROW, CONV_2_OFM_COL,
                                   CONV_2_OFM_CH, CONV_2_PE, CONV_2_OUT_BIT>(
      test_out, "conv_DSP6_out.txt");
}
