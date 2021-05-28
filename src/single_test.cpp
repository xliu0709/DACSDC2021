#include "config.h"
#include "conv2d.h"
#include "conv2d_DSPopt.hpp"
#include "param.h"
#include "weight3.hpp"
#include <ap_int.h>
#include <hls_video.h>
#include <stdint.h>

#define IN_IMAGE_WIDTH 640
#define IN_IMAGE_HEIGHT 360

#define RESIZE_IMAGE_WIDTH 320
#define RESIZE_IMAGE_HEIGHT 160

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
  hls::stream<ap_uint<CONV_2_IN_BIT * CONV_2_IFM_CH>> test_in("test_in");

  for (int r = 0; r < CONV_2_IFM_ROW; r++)
    for (int c = 0; c < CONV_2_IFM_COL; c++) {
      ap_uint<CONV_2_IN_BIT *CONV_2_IFM_CH> data = 0;

      for (int i = 0; i < CONV_2_IFM_CH; i += CONV_2_SIMD) {

        ap_uint<32> row = random();
        ap_uint<32> col = random();
        // ap_uint<4> ch = i / CONV_2_SIMD;
        // ap_uint<8> ch = random() % 256;
        data(i * CONV_2_IN_BIT + 63, i * CONV_2_IN_BIT) = (row, col);
      }
      golden_in << data;
      test_in << data;
    }
  cout << RAND_MAX << endl;

  // initialziation<CONV_2_K, CONV_2_IFM_CH, CONV_2_OFM_CH, CONV_2_PE,
  // CONV_2_SIMD,
  //                CONV_2_W_BIT>(conv_2_w_dspopt, "center");

  hls::stream<ap_uint<CONV_2_OUT_BIT * CONV_2_OFM_CH>> golden_out("golden_out");
  cout << "entering conv3x3bn" << endl;
  conv3x3_bn_act<CONV_2_IFM_ROW, CONV_2_IFM_COL, CONV_2_IFM_CH, CONV_2_IN_BIT,

                 CONV_2_OFM_CH, CONV_2_OUT_BIT,

                 CONV_2_W_BIT, 32, CONV_2_INC_BIT, CONV_2_BIAS_BIT,

                 CONV_2_SIMD, CONV_2_PE, CONV_2_L_SHIFT>(
      golden_in, conv_2_w, conv_2_inc, conv_2_bias, golden_out, 1);

  conv3x3_bn_act_DSPopt<CONV_2_IFM_ROW, CONV_2_IFM_COL, CONV_2_IFM_CH,
                        CONV_2_IN_BIT,

                        CONV_2_OFM_CH, CONV_2_OUT_BIT,

                        CONV_2_W_BIT, 32, CONV_2_INC_BIT, CONV_2_BIAS_BIT,

                        CONV_2_SIMD, CONV_2_PE, CONV_2_L_SHIFT>(
      test_in, conv_2_w_dspopt, conv_2_inc, conv_2_bias, golden_out, 1);
}