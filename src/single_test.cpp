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

int main(int argc, char **argv) {

  hls::stream<ap_uint<CONV_2_IN_BIT * CONV_2_IFM_CH>> golden_in("golden_in");
  hls::stream<ap_uint<CONV_2_IN_BIT * CONV_2_IFM_CH>> test_in("test_in");

  for (int r = 0; r < CONV_2_IFM_ROW; r++)
    for (int c = 0; c < CONV_2_IFM_COL; c++) {
      ap_uint<CONV_2_IN_BIT *CONV_2_IFM_CH> data = 0;

      for (int i = 0; i < CONV_2_IFM_CH; i += CONV_2_SIMD) {

        ap_uint<8> row = random() % 256;
        ap_uint<8> col = random() % 256;
        ap_uint<8> ch = random() % 256;
        data(i * CONV_2_IN_BIT + 23, i * CONV_2_IN_BIT) = (ch, row, col);
      }
      golden_in << data;
      test_in << data;
    }

  hls::stream<ap_uint<CONV_2_OUT_BIT * CONV_2_OFM_CH>> golden_out("golden_out");
  cout << "entering conv3x3bn" << endl;
  // conv3x3_bn_act<CONV_2_IFM_ROW, CONV_2_IFM_COL, CONV_2_IFM_CH,
  // CONV_2_IN_BIT,

  //                CONV_2_OFM_CH, CONV_2_OUT_BIT,

  //                CONV_2_W_BIT, 32, CONV_2_INC_BIT, CONV_2_BIAS_BIT,

  //                CONV_2_SIMD, CONV_2_PE, CONV_2_L_SHIFT>(
  //     golden_in, conv_2_w, conv_2_inc, conv_2_bias, golden_out, 1);

  conv3x3_bn_act_DSPopt<CONV_2_IFM_ROW, CONV_2_IFM_COL, CONV_2_IFM_CH,
                        CONV_2_IN_BIT,

                        CONV_2_OFM_CH, CONV_2_OUT_BIT,

                        CONV_2_W_BIT, 32, CONV_2_INC_BIT, CONV_2_BIAS_BIT,

                        CONV_2_SIMD, CONV_2_PE, CONV_2_L_SHIFT>(
      test_in, conv_2_w_dspopt, conv_2_inc, conv_2_bias, golden_out, 1);
}