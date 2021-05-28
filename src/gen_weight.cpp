

#include "config.h"
#include "param.h"
#include <ap_int.h>
#include <fstream>
#include <string>
using namespace std;

template <unsigned K, unsigned IN_CH, unsigned OUT_CH, unsigned W_BIT,
          unsigned PE, unsigned SIMD>
void weightReorderDSP6(
    ap_int<W_BIT> weight[K][K][IN_CH][OUT_CH],
    ap_uint<W_BIT * SIMD> weightDSP6[PE][3][K * IN_CH / SIMD * OUT_CH / PE]) {
  for (int kr = 0; kr < K; kr++)
    for (int kc = 0; kc < K; kc++)
      for (int oc = 0; oc < OUT_CH; oc++) {
        for (int ic = 0; ic < IN_CH; ic += SIMD) {
          ap_uint<W_BIT * SIMD> data;
          for (int s = 0; s < SIMD; s++) {
            data(W_BIT * (s + 1) - 1, W_BIT * s) = weight[kr][kc][ic + s][oc];
          }
          weightDSP6[oc % PE][kc][oc / PE * K * IN_CH / SIMD +
                                  kr * IN_CH / SIMD + ic / SIMD] = data;
        }
      }
}

template <unsigned K, unsigned IN_CH, unsigned OUT_CH, unsigned W_BIT,
          unsigned PE, unsigned SIMD>
void ultranetWeightToWeight(
    const ap_uint<W_BIT * SIMD> weightU[PE][K * K * OUT_CH / PE * IN_CH / SIMD],
    ap_int<W_BIT> weight[K][K][IN_CH][OUT_CH]) {
  for (int kr = 0; kr < K; kr++)
    for (int kc = 0; kc < K; kc++)
      for (int oc = 0; oc < OUT_CH; oc++) {
        for (int ic = 0; ic < IN_CH; ic += SIMD) {
          ap_uint<W_BIT * SIMD> data;
          data = weightU[oc % PE][oc / PE * K * K * IN_CH / SIMD +
                                  kr * K * IN_CH / SIMD + kc * IN_CH / SIMD +
                                  ic / SIMD];

          for (int s = 0; s < SIMD; s++) {
            weight[kr][kc][ic + s][oc] = data(W_BIT * (s + 1) - 1, W_BIT * s);
          }
        }
      }
}

// template <unsigned K, unsigned SIMD, unsigned PE, unsigned IN_CH,
//           unsigned OUT_CH, unsigned W_BIT>
// void reorder_weight(
//     const ap_uint<W_BIT * SIMD> weights[PE][K * K * OUT_CH / PE * IN_CH /
//                                             SIMD], // weights:
//                                             PEMUM->K->K->SIMD
//     ap_uint<W_BIT * SIMD> weights3[PE][K][K * OUT_CH / PE * IN_CH / SIMD]) {

//   const unsigned PENUM = OUT_CH / PE;
//   const unsigned SIMDNUM = IN_CH / SIMD;

//   for (int peIdx = 0; peIdx < PENUM; peIdx++)
//     for (int kr = 0; kr < K; kr++)
//       for (int kc = 0; kc < K; kc++)
//         for (int simdIdx = 0; simdIdx < SIMDNUM; simdIdx++)
//           for (int p = 0; p < PE; p++) {
//             unsigned weights3Addr =
//                 peIdx * K * SIMDNUM + kr * SIMDNUM + simdIdx;
//             unsigned weightsAddr = peIdx * K * K * SIMDNUM + kr * K * SIMDNUM
//             +
//                                    kc * SIMDNUM + simdIdx;
//             weights3[p][kc][weights3Addr] = weights[p][weightsAddr];
//           }
// }

// template <unsigned K, unsigned IN_CH, unsigned U_SIMD, unsigned U_PE,
//           unsigned SIMD, unsigned PE, unsigned OUT_CH, unsigned W_BIT>
// void reorder_weight(
//     const ap_uint<W_BIT * U_SIMD> weightU[U_PE]
//                                          [K * K * OUT_CH / PE * IN_CH /
//                                          U_SIMD],
//     ap_uint<W_BIT * SIMD> weightDSP6[PE][K][K * OUT_CH / PE * IN_CH / SIMD])
//     {

//   ap_int<W_BIT> weight[K][K][IN_CH][OUT_CH];

// }

template <unsigned K, unsigned SIMD, unsigned PE, unsigned IN_CH,
          unsigned OUT_CH, unsigned W_BIT>
void print_header(
    ap_uint<W_BIT * SIMD> weights3[PE][K][K * OUT_CH / PE * IN_CH / SIMD],
    std::string weight_name, ofstream &of) {

  of << "const ap_uint<" << W_BIT * SIMD << ">" << weight_name << "[" << PE
     << "]"
     << "[" << K << "]"
     << "[" << K * OUT_CH / PE * IN_CH / SIMD << "]=" << endl;
  of << "{";
  for (int i = 0; i < PE; i++) {
    of << "{";
    for (int j = 0; j < K; j++) {
      of << "{";
      for (int k = 0; k < K * OUT_CH / PE * IN_CH / SIMD; k++) {
        of << weights3[i][j][k].to_string(16);
        if (k != K * OUT_CH / PE * IN_CH / SIMD - 1)
          of << ",";
      }
      if (j != K - 1)
        of << "},\n";
      else
        of << "}\n";
    }
    if (i != PE - 1)
      of << "},\n";
    else
      of << "}\n";
  }
  of << "};\n";
}

template <unsigned K, unsigned IN_CH, unsigned OUT_CH, unsigned USIMD,
          unsigned UPE, unsigned DSP6SIMD, unsigned DSP6PE, unsigned W_BIT>
void transform_weight_print(
    const ap_uint<W_BIT * USIMD> weightU[UPE]
                                        [K * K * OUT_CH / UPE * IN_CH / USIMD],
    string weight_name, ofstream &of) {

  ap_uint<W_BIT * DSP6SIMD> weightDSP6[DSP6PE][K]
                                      [K * OUT_CH / DSP6PE * IN_CH / DSP6SIMD];

  ap_int<W_BIT> weight[K][K][IN_CH][OUT_CH];
  ultranetWeightToWeight<K, IN_CH, OUT_CH, W_BIT, UPE, USIMD>(weightU, weight);
  weightReorderDSP6<K, IN_CH, OUT_CH, W_BIT, DSP6PE, DSP6SIMD>(weight,
                                                               weightDSP6);
  // reorder_weight<K, SIMD, PE, IN_CH, OUT_CH, W_BIT>(weights, weights3);
  print_header<K, DSP6SIMD, DSP6PE, IN_CH, OUT_CH, W_BIT>(weightDSP6,
                                                          weight_name, of);
}

int main() {
  ofstream of;
  of.open("src/weight3.hpp", ofstream::out);
  of << "#include <ap_int.h>\n";
  transform_weight_print<CONV_0_K, CONV_0_IFM_CH, CONV_0_OFM_CH, CONV_0_SIMD,
                         CONV_0_PE, CONV_0_SIMD_DSP6, CONV_0_PE, CONV_0_W_BIT>(
      conv_0_w, "conv_0_w_dspopt", of);

  transform_weight_print<CONV_1_K, CONV_1_IFM_CH, CONV_1_OFM_CH, CONV_1_SIMD,
                         CONV_1_PE, CONV_1_SIMD_DSP6, CONV_1_PE, CONV_1_W_BIT>(
      conv_1_w, "conv_1_w_dspopt", of);

  transform_weight_print<CONV_2_K, CONV_2_IFM_CH, CONV_2_OFM_CH, CONV_2_SIMD,
                         CONV_2_PE, CONV_2_SIMD_DSP6, CONV_2_PE, CONV_2_W_BIT>(
      conv_2_w, "conv_2_w_dspopt", of);
  of.close();
}