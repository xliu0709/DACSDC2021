

#include "config.h"
#include "param.h"
#include <ap_int.h>
#include <fstream>
#include <string>
using namespace std;

template <unsigned BIT, unsigned OUT_CH, unsigned PE>
void packVec(const ap_int<BIT> vecPartition[PE][OUT_CH / PE],
             ap_int<BIT> vec[OUT_CH]) {

  for (int i = 0; i < OUT_CH / PE; i++)
    for (int p = 0; p < PE; p++) {
      vec[i * PE + p] = vecPartition[p][i];
    }
}

template <unsigned BIT, unsigned OUT_CH, unsigned PE>
void printVecHeader(ap_int<BIT> vec[OUT_CH], ofstream &file, string varname) {
  file << "const ap_int<" << BIT << "> " << varname << "[" << PE << "]["
       << OUT_CH / PE << "]=" << endl;
  file << "{";
  for (int i = 0; i < PE; i++) {
    file << "{";
    for (int j = 0; j < OUT_CH / PE; j++) {
      file << "\"" << vec[j * PE + i].to_string(16) << "\"";
      if (j != OUT_CH / PE - 1) {
        file << ",";
      }
    }
    if (i != PE - 1)
      file << "},\n";
    else
      file << "}\n";
  }
  file << "};\n";
}

template <unsigned BIT, unsigned OUT_CH, unsigned PE1, unsigned PE2>
void transformVecAndPrint(const ap_int<BIT> inVec[PE1][OUT_CH / PE1],
                          ofstream &file, string vectorname) {
  ap_int<BIT> vec[OUT_CH];
  packVec<BIT, OUT_CH, PE1>(inVec, vec);
  printVecHeader<BIT, OUT_CH, PE2>(vec, file, vectorname);
}

template <unsigned K, unsigned IN_CH, unsigned OUT_CH, unsigned W_BIT,
          unsigned PE, unsigned SIMD>
void weightReorderDSP6(
    ap_int<W_BIT> weight[K][K][OUT_CH][IN_CH],
    ap_uint<W_BIT * SIMD> weightDSP6[PE][3][K * IN_CH / SIMD * OUT_CH / PE],
    bool first) {
  for (int kr = 0; kr < K; kr++)
    for (int kc = 0; kc < K; kc++)
      for (int oc = 0; oc < OUT_CH; oc++) {
        for (int ic = 0; ic < IN_CH; ic += SIMD) {

          ap_uint<W_BIT * SIMD> data;
          if (first) {
            for (int s = 0; s < SIMD; s++) {
              data(W_BIT * (s + 1) - 1, W_BIT * s) = weight[kr][kc][oc][ic + s];
            }
            weightDSP6[oc % PE][kr][oc / PE * K * IN_CH / SIMD +
                                    kc * IN_CH / SIMD + ic / SIMD] = data;
          } else {
            for (int s = 0; s < SIMD; s++) {
              data(W_BIT * (s + 1) - 1, W_BIT * s) = weight[kr][kc][oc][ic + s];
            }
            weightDSP6[oc % PE][kc][oc / PE * K * IN_CH / SIMD +
                                    kr * IN_CH / SIMD + ic / SIMD] = data;
          }
        }
      }
}

template <unsigned K, unsigned IN_CH, unsigned OUT_CH, unsigned W_BIT,
          unsigned PE, unsigned SIMD>
void ultranetConv3x3WeightToWeight(
    const ap_uint<W_BIT * SIMD> weightU[PE][K * K * OUT_CH / PE * IN_CH / SIMD],
    ap_int<W_BIT> weight[K][K][OUT_CH][IN_CH]) {
  for (int kr = 0; kr < K; kr++)
    for (int kc = 0; kc < K; kc++)
      for (int oc = 0; oc < OUT_CH; oc++) {
        for (int ic = 0; ic < IN_CH; ic += SIMD) {
          ap_uint<W_BIT * SIMD> data;
          data = weightU[oc % PE][oc / PE * K * K * IN_CH / SIMD +
                                  kr * K * IN_CH / SIMD + kc * IN_CH / SIMD +
                                  ic / SIMD];

          for (int s = 0; s < SIMD; s++) {
            weight[kr][kc][oc][ic + s] = data(W_BIT * (s + 1) - 1, W_BIT * s);
          }
        }
      }
}

template <unsigned IN_CH, unsigned OUT_CH, unsigned W_BIT, unsigned PE,
          unsigned SIMD>
void ultranetConv1x1WeightToWeight(
    const ap_uint<W_BIT * SIMD> weightU[PE][OUT_CH / PE * IN_CH / SIMD],
    ap_int<W_BIT> weight[OUT_CH][IN_CH]) {

  for (int oc = 0; oc < OUT_CH; oc++) {
    for (int ic = 0; ic < IN_CH; ic += SIMD) {
      ap_uint<W_BIT * SIMD> data;
      data = weightU[oc % PE][oc / PE * IN_CH / SIMD + ic / SIMD];

      for (int s = 0; s < SIMD; s++) {
        weight[oc][ic + s] = data(W_BIT * (s + 1) - 1, W_BIT * s);
      }
    }
  }
}

template <unsigned SIMD, unsigned PE, unsigned IN_CH, unsigned OUT_CH,
          unsigned W_BIT>
void printConv1x1Header(ap_int<W_BIT> weights[OUT_CH][IN_CH],
                        std::string weight_name, ofstream &of) {

  of << "const ap_uint<" << W_BIT * SIMD << ">" << weight_name << "[" << PE
     << "][" << OUT_CH / PE * IN_CH / SIMD << "]=" << endl;
  of << "{";

  for (int i = 0; i < PE; i++) {
    of << "{";
    for (int peIdx = 0; peIdx < OUT_CH / PE; peIdx++)
      for (int simdIdx = 0; simdIdx < IN_CH / SIMD; simdIdx++) {

        ap_uint<SIMD * W_BIT> wdata;
        for (int s = 0; s < SIMD; s++) {
          wdata((s + 1) * W_BIT - 1, s * W_BIT) =
              weights[peIdx * PE + i][simdIdx * SIMD + s];
        }

        of << "\"" << wdata.to_string(16) << "\"";
        if (simdIdx != IN_CH / SIMD - 1 || peIdx != OUT_CH / PE - 1)
          of << ",";
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
void printConv3x3Header(
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
void transformConv3x3Print(
    const ap_uint<W_BIT * USIMD> weightU[UPE]
                                        [K * K * OUT_CH / UPE * IN_CH / USIMD],
    string weight_name, ofstream &of, bool first = false) {

  ap_uint<W_BIT * DSP6SIMD> weightDSP6[DSP6PE][K]
                                      [K * OUT_CH / DSP6PE * IN_CH / DSP6SIMD];

  ap_int<W_BIT> weight[K][K][OUT_CH][IN_CH];
  ultranetConv3x3WeightToWeight<K, IN_CH, OUT_CH, W_BIT, UPE, USIMD>(weightU,
                                                                     weight);

  weightReorderDSP6<K, IN_CH, OUT_CH, W_BIT, DSP6PE, DSP6SIMD>(
      weight, weightDSP6, first);

  // reorder_weight<K, SIMD, PE, IN_CH, OUT_CH, W_BIT>(weights, weights3);
  printConv3x3Header<K, DSP6SIMD, DSP6PE, IN_CH, OUT_CH, W_BIT>(
      weightDSP6, weight_name, of);
}

template <unsigned IN_CH, unsigned OUT_CH, unsigned USIMD, unsigned UPE,
          unsigned DSP2SIMD, unsigned DSP2PE, unsigned W_BIT>
void transformConv1x1Print(
    const ap_uint<W_BIT * USIMD> weightU[UPE][OUT_CH / UPE * IN_CH / USIMD],
    string weight_name, ofstream &of) {

  ap_int<W_BIT> weight[OUT_CH][IN_CH];
  ultranetConv1x1WeightToWeight<IN_CH, OUT_CH, W_BIT, UPE, USIMD>(weightU,
                                                                  weight);

  // reorder_weight<K, SIMD, PE, IN_CH, OUT_CH, W_BIT>(weights, weights3);
  printConv1x1Header<DSP2SIMD, DSP2PE, IN_CH, OUT_CH, W_BIT>(weight,
                                                             weight_name, of);
}

int main() {
  ofstream of;
  of.open("src/weight3.hpp", ofstream::out);
  of << "#include <ap_int.h>\n";
  transformConv3x3Print<CONV_0_K, CONV_0_IFM_CH, CONV_0_OFM_CH, CONV_0_SIMD,
                        CONV_0_PE, CONV_0_SIMD_DSP6, CONV_0_PE_DSP6,
                        CONV_0_W_BIT>(conv_0_w, "conv_0_w_dspopt", of, true);
  transformVecAndPrint<CONV_0_BIAS_BIT, CONV_0_OFM_CH, CONV_0_PE,
                       CONV_0_PE_DSP6>(conv_0_bias, of, "conv_0_bias_dspopt");
  transformVecAndPrint<CONV_0_INC_BIT, CONV_0_OFM_CH, CONV_0_PE,
                       CONV_0_PE_DSP6>(conv_0_inc, of, "conv_0_inc_dspopt");

  transformConv3x3Print<CONV_1_K, CONV_1_IFM_CH, CONV_1_OFM_CH, CONV_1_SIMD,
                        CONV_1_PE, CONV_1_SIMD_DSP6, CONV_1_PE_DSP6,
                        CONV_1_W_BIT>(conv_1_w, "conv_1_w_dspopt", of);
  transformVecAndPrint<CONV_1_BIAS_BIT, CONV_1_OFM_CH, CONV_1_PE,
                       CONV_1_PE_DSP6>(conv_1_bias, of, "conv_1_bias_dspopt");
  transformVecAndPrint<CONV_1_INC_BIT, CONV_1_OFM_CH, CONV_1_PE,
                       CONV_1_PE_DSP6>(conv_1_inc, of, "conv_1_inc_dspopt");

  transformConv3x3Print<CONV_2_K, CONV_2_IFM_CH, CONV_2_OFM_CH, CONV_2_SIMD,
                        CONV_2_PE, CONV_2_SIMD_DSP6, CONV_2_PE_DSP6,
                        CONV_2_W_BIT>(conv_2_w, "conv_2_w_dspopt", of);
  transformVecAndPrint<CONV_2_BIAS_BIT, CONV_2_OFM_CH, CONV_2_PE,
                       CONV_2_PE_DSP6>(conv_2_bias, of, "conv_2_bias_dspopt");
  transformVecAndPrint<CONV_2_INC_BIT, CONV_2_OFM_CH, CONV_2_PE,
                       CONV_2_PE_DSP6>(conv_2_inc, of, "conv_2_inc_dspopt");

  transformConv3x3Print<CONV_3_K, CONV_3_IFM_CH, CONV_3_OFM_CH, CONV_3_SIMD,
                        CONV_3_PE, CONV_3_SIMD_DSP6, CONV_3_PE_DSP6,
                        CONV_3_W_BIT>(conv_3_w, "conv_3_w_dspopt", of);
  transformVecAndPrint<CONV_3_BIAS_BIT, CONV_3_OFM_CH, CONV_3_PE,
                       CONV_3_PE_DSP6>(conv_3_bias, of, "conv_3_bias_dspopt");
  transformVecAndPrint<CONV_3_INC_BIT, CONV_3_OFM_CH, CONV_3_PE,
                       CONV_3_PE_DSP6>(conv_3_inc, of, "conv_3_inc_dspopt");

  transformConv3x3Print<CONV_4_K, CONV_4_IFM_CH, CONV_4_OFM_CH, CONV_4_SIMD,
                        CONV_4_PE, CONV_4_SIMD_DSP6, CONV_4_PE_DSP6,
                        CONV_4_W_BIT>(conv_4_w, "conv_4_w_dspopt", of);
  transformVecAndPrint<CONV_4_BIAS_BIT, CONV_4_OFM_CH, CONV_4_PE,
                       CONV_4_PE_DSP6>(conv_4_bias, of, "conv_4_bias_dspopt");
  transformVecAndPrint<CONV_4_INC_BIT, CONV_4_OFM_CH, CONV_4_PE,
                       CONV_4_PE_DSP6>(conv_4_inc, of, "conv_4_inc_dspopt");

  transformConv3x3Print<CONV_5_K, CONV_5_IFM_CH, CONV_5_OFM_CH, CONV_5_SIMD,
                        CONV_5_PE, CONV_5_SIMD_DSP6, CONV_5_PE_DSP6,
                        CONV_5_W_BIT>(conv_5_w, "conv_5_w_dspopt", of);
  transformVecAndPrint<CONV_5_BIAS_BIT, CONV_5_OFM_CH, CONV_5_PE,
                       CONV_5_PE_DSP6>(conv_5_bias, of, "conv_5_bias_dspopt");
  transformVecAndPrint<CONV_5_INC_BIT, CONV_5_OFM_CH, CONV_5_PE,
                       CONV_5_PE_DSP6>(conv_5_inc, of, "conv_5_inc_dspopt");

  transformConv3x3Print<CONV_6_K, CONV_6_IFM_CH, CONV_6_OFM_CH, CONV_6_SIMD,
                        CONV_6_PE, CONV_6_SIMD_DSP6, CONV_6_PE_DSP6,
                        CONV_6_W_BIT>(conv_6_w, "conv_6_w_dspopt", of);
  transformVecAndPrint<CONV_6_BIAS_BIT, CONV_6_OFM_CH, CONV_6_PE,
                       CONV_6_PE_DSP6>(conv_6_bias, of, "conv_6_bias_dspopt");
  transformVecAndPrint<CONV_6_INC_BIT, CONV_6_OFM_CH, CONV_6_PE,
                       CONV_6_PE_DSP6>(conv_6_inc, of, "conv_6_inc_dspopt");

  transformConv3x3Print<CONV_7_K, CONV_7_IFM_CH, CONV_7_OFM_CH, CONV_7_SIMD,
                        CONV_7_PE, CONV_7_SIMD_DSP6, CONV_7_PE_DSP6,
                        CONV_7_W_BIT>(conv_7_w, "conv_7_w_dspopt", of);
  transformVecAndPrint<CONV_7_BIAS_BIT, CONV_7_OFM_CH, CONV_7_PE,
                       CONV_7_PE_DSP6>(conv_7_bias, of, "conv_7_bias_dspopt");
  transformVecAndPrint<CONV_7_INC_BIT, CONV_7_OFM_CH, CONV_7_PE,
                       CONV_7_PE_DSP6>(conv_7_inc, of, "conv_7_inc_dspopt");
  transformConv1x1Print<CONV_8_IFM_CH, CONV_8_OFM_CH, CONV_8_SIMD, CONV_8_PE,
                        CONV_8_SIMD_DSP2, CONV_8_PE_DSP2, CONV_8_W_BIT>(
      conv_8_w, "conv_8_dspopt", of);
  of.close();
}