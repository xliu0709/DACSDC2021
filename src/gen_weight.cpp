

#include "config.h"
#include "param.h"
#include <ap_int.h>
#include <fstream>
#include <string>
using namespace std;

template <unsigned K, unsigned SIMD, unsigned PE, unsigned IN_CH,
          unsigned OUT_CH, unsigned W_BIT>
void reorder_weight(
    const ap_uint<W_BIT * SIMD> weights[PE][K * K * OUT_CH / PE * IN_CH /
                                            SIMD], // weights: PEMUM->K->K->SIMD
    ap_uint<W_BIT * SIMD> weights3[PE][K][K * OUT_CH / PE * IN_CH / SIMD]) {

  const unsigned PENUM = OUT_CH / PE;
  const unsigned SIMDNUM = IN_CH / SIMD;

  for (int peIdx = 0; peIdx < PENUM; peIdx++)
    for (int kr = 0; kr < K; kr++)
      for (int kc = 0; kc < K; kc++)
        for (int simdIdx = 0; simdIdx < SIMDNUM; simdIdx++)
          for (int p = 0; p < PE; p++) {
            unsigned weights3Addr =
                peIdx * K * SIMDNUM + kr * SIMDNUM + simdIdx;
            unsigned weightsAddr = peIdx * K * K * SIMDNUM + kr * K * SIMDNUM +
                                   kc * SIMDNUM + simdIdx;
            weights3[p][kc][weights3Addr] = weights[p][weightsAddr];
          }
}

template <unsigned K, unsigned SIMD, unsigned PE, unsigned IN_CH,
          unsigned OUT_CH, unsigned W_BIT>
void print_header(
    ap_uint<W_BIT * SIMD> weights3[PE][K][K * OUT_CH / PE * IN_CH / SIMD],
    std::string weight_name, ofstream &of) {

  of << "ap_uint<" << W_BIT * SIMD << ">" << weight_name << "[" << PE << "]"
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

template <unsigned K, unsigned SIMD, unsigned PE, unsigned IN_CH,
          unsigned OUT_CH, unsigned W_BIT>
void transform_weight_print(
    const ap_uint<W_BIT * SIMD> weights[PE][K * K * OUT_CH / PE * IN_CH / SIMD],
    string weight_name, ofstream &of) {

  ap_uint<W_BIT * SIMD> weights3[PE][K][K * OUT_CH / PE * IN_CH / SIMD];

  reorder_weight<K, SIMD, PE, IN_CH, OUT_CH, W_BIT>(weights, weights3);
  print_header<K, SIMD, PE, IN_CH, OUT_CH, W_BIT>(weights3, weight_name, of);
}

int main() {
  ofstream of;
  of.open("src/weight3.hpp", ofstream::out | ofstream::app);
  of << "#include <ap_int.h>\n";
  transform_weight_print<CONV_2_K, CONV_2_SIMD, CONV_2_PE, CONV_2_IFM_CH,
                         CONV_2_OFM_CH, CONV_2_W_BIT>(conv_2_w,
                                                      "conv_2_w_dspopt", of);
  of.close();
}