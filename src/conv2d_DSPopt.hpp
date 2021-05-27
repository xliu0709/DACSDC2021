#ifndef __CONV2D_DSPOPT_HPP__
#define __CONV2D_DSPOPT_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "debug.hpp"
#include "function.h"
#include "matrix_vector_unit.h"
#include "sliding_window_unit.h"
#include "stream_tools.h"

#define CEILDIV(x, y) (((x) + (y)-1) / (y))

template <unsigned K, unsigned S, unsigned Din_H, unsigned Din_W, unsigned Cin,
          unsigned SIMD, unsigned Ibit, unsigned PENUM>
void SWU_reordered(stream<ap_uint<Cin * Ibit>> &in,
                   stream<ap_uint<SIMD * Ibit * 2>> &out,
                   const unsigned reps = 1) {
  static_assert(S == 1, "S is not 1");
  static_assert(Din_H % 2 == 0, "Din_H mod 2 is not 0");
  static_assert((Din_W - K) % S == 0, "(Din_W-K) mod S is not 0");
  static_assert((Din_H - K) % S == 0, "(Din_H-K) mod S is not 0");
  static_assert(Cin % SIMD == 0, "Cin mod SIMD is not 0");
  static_assert(K >= S, "K is not >= than S");

  constexpr unsigned line_buffer_size = K * Din_W;
  // const unsigned steps = (Din_W ;

  constexpr unsigned SIMDbit = SIMD * Ibit;
  constexpr unsigned SIMDgroupNum = Cin / SIMD;

  ap_uint<SIMD * Ibit> line_buffer[SIMDgroupNum][line_buffer_size];
#pragma HLS ARRAY_PARTITION variable linebuffer dim = 1 complete
#pragma HLS RESOURCE variable line_buffer core = RAM_2P

  ap_uint<Cin * Ibit> temp_in;

  ap_uint<1> initial_fill = 0;
  unsigned stride = 0;
  unsigned pointer = 0;
  unsigned h = 0;

  cout << "here" << endl;
  fflush(stdout);

  for (unsigned rep = 0; rep < reps * Din_H; rep++) {
    for (unsigned w = 0; w < Din_W; w++) {
#pragma HLS PISIMDLINE II = 1
      temp_in = in.read();

      unsigned line_buffer_pointer = pointer + w;
      if (line_buffer_pointer >= line_buffer_size) {
        line_buffer_pointer = line_buffer_pointer - line_buffer_size;
      }
      for (int i = 0; i < SIMDgroupNum; i++) {
#pragma HLS unroll
        ap_uint<SIMDbit> data = temp_in(i * SIMDbit + SIMDbit - 1, i * SIMDbit);
        line_buffer[i][line_buffer_pointer] = data;
      }
    }

    stride += 1;
    pointer += Din_W;
    if (pointer >= line_buffer_size) {
      pointer = pointer - line_buffer_size;
      initial_fill = 1;
#ifdef SWU_DEBUG
      cout << "initial_fill set to 1!" << endl;
#endif
    }

    if (initial_fill == 1 && stride >= S) {
      stride = 0;
      for (int p = 0; p < PENUM; p++) {
        unsigned s = 0;
        unsigned r = 0;
        unsigned c = 0;
        unsigned ch = 0;
        const unsigned iter_num = Din_W * K * SIMDgroupNum / 2;
        for (unsigned i = 0; i < iter_num; i++) {
#pragma HLS PIPELINE II = 1
          unsigned read_address = pointer + r * Din_W + c;

          if (read_address >= line_buffer_size)
            read_address = read_address - line_buffer_size;
#ifdef SWU_DEBUG
          cout << "read_address: " << read_address << endl;
#endif
          cout << "r,c,ch" << r << "," << c << "," << ch << endl;

          ap_uint<SIMDbit * 2> temp_out = (line_buffer[ch][read_address + 1],
                                           line_buffer[ch][read_address]);
          out.write(temp_out);

          if (ch == SIMDgroupNum - 1) {
            ch = 0;
            if (r == K - 1) {
              r = 0;
              c += 2;
            } else
              r++;
          } else
            ch++;
        }
      }
    }
  }
}

template <unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT, unsigned SIMD>
void simd_update_m(ap_int<PROD_BIT * 2> m_lo[SIMD],
                   ap_int<PROD_BIT * 2> m_hi[SIMD],
                   ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD],
                   ap_int<PROD_BIT + IN_BIT> ipack[SIMD], bool clear) {
#pragma HLS inline
#pragma HLS pipeline II = 1
#pragma HLS array_partition variable = wpack complete
#pragma HLS array_partition variable = ipack complete
#pragma HLS array_partition variable = m_lo complete
#pragma HLS array_partition variable = m_hi complete
  for (int i = 0; i < SIMD; i++) {
    ap_int<PROD_BIT * 2> res = clear ? (ap_int<PROD_BIT * 2>)0 : m_hi[i];
    ap_int<PROD_BIT * 4> m = wpack[i] * ipack[i] + res;
    m_lo[i] = m(PROD_BIT * 2 - 1, 0);
    m_hi[i] = m(PROD_BIT * 4 - 1, PROD_BIT * 2) + m[PROD_BIT * 2 - 1];
  }
}

template <unsigned PROD_BIT, unsigned M_BIT, unsigned SIMD>
void simd_sum(ap_int<PROD_BIT * 2> m_lo[SIMD], ap_int<M_BIT> &sumHi,
              ap_int<M_BIT> &sumLo, bool o_clear) {
  ap_int<PROD_BIT> X[SIMD];
  ap_int<PROD_BIT> Y[SIMD];
#pragma HLS array_partition variable = m
#pragma HLS array_partition variable = X
#pragma HLS array_partition variable = Y

  for (int i = 0; i < SIMD; i++) {
    X[i] = m_lo[i](PROD_BIT - 1, 0);
    Y[i] =
        m_lo[i](PROD_BIT * 2 - 1, PROD_BIT) + (ap_uint<1>)m_lo[i][PROD_BIT - 1];
  }
  ap_int<M_BIT> retHi = 0, retLo = 0;

  for (int i = 0; i < SIMD; i++) {

    retHi += Y[i];
    retLo += X[i];
  }
  if (o_clear) {
    sumHi = retHi;
    sumLo = retLo;
  } else {
    sumHi += retHi;
    sumLo += retLo;
  }
}

template <unsigned IN_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_input_data(ap_uint<IN_BIT * SIMD> A, ap_uint<IN_BIT * SIMD> B,
                     ap_uint<PROD_BIT + IN_BIT> ipack[SIMD]) {
#pragma HLS array_partition variable = ipack

  for (int i = 0; i < SIMD; i++) {
    ipack[i] =
        (A(i * IN_BIT + IN_BIT - 1, i * IN_BIT), (ap_uint<PROD_BIT - IN_BIT>)0,
         B(i * IN_BIT + IN_BIT - 1, i * IN_BIT));
  }
}

template <unsigned W_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_weight_data(ap_uint<W_BIT * SIMD> w2, ap_uint<W_BIT * SIMD> w1,
                      ap_uint<W_BIT * SIMD> w0,
                      ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD]) {
#pragma HLS array_partition variable = wpack

  for (int i = 0; i < SIMD; i++) {
    ap_int<W_BIT> w2_seg = w2(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w1_seg = w1(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w0_seg = w0(i * W_BIT + W_BIT - 1, i * W_BIT);
    wpack[i] =
        (w0_seg * (1 << (PROD_BIT * 2))) + (w1_seg * (1 << PROD_BIT)) + w2_seg;
  }
}

template <unsigned W_BIT, unsigned IN_BIT, unsigned SIMD, unsigned PROD_BIT>
void simd_MAC_normal(ap_int<W_BIT * SIMD> w0, ap_int<W_BIT * SIMD> w1,
                     ap_int<W_BIT * SIMD> w2, ap_uint<IN_BIT * SIMD> i0,
                     ap_uint<IN_BIT * SIMD> i1, ap_int<PROD_BIT + 5> &partial0,
                     ap_int<PROD_BIT + 5> &partial1,
                     ap_int<PROD_BIT + 5> &partial2,
                     ap_int<PROD_BIT + 5> &partial3) {
  ap_int<PROD_BIT + 5> r0, r1, r2, r3;
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;
  for (int i = 0; i < SIMD; i++) {
    ap_int<W_BIT> w0_seg = w0((i + 1) * W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w1_seg = w1((i + 1) * W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w2_seg = w2((i + 1) * W_BIT - 1, i * W_BIT);
    ap_uint<IN_BIT> x0_seg = i0((i + 1) * IN_BIT - 1, i * IN_BIT);
    ap_uint<IN_BIT> x1_seg = i1((i + 1) * IN_BIT - 1, i * IN_BIT);

    r0 += x0_seg * w2_seg;
    r1 += x0_seg * w0_seg + x1_seg * w1_seg;
    r2 += x0_seg * w1_seg + x1_seg * w2_seg;
    r3 += x1_seg * w0_seg;
  }
  partial0 = r0;
  partial1 = r1;
  partial2 = r2;
  partial3 = r3;
}

template <unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT, unsigned SIMD,
          unsigned CASCADE_NUM>
void simd_MAC(ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD],
              ap_uint<PROD_BIT + IN_BIT> ipack[SIMD],
              ap_int<PROD_BIT + 5> &partial0, ap_int<PROD_BIT + 5> &partial1,
              ap_int<PROD_BIT + 5> &partial2, ap_int<PROD_BIT + 5> &partial3) {

  ap_int<PROD_BIT + 5> r0, r1, r2, r3;
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;
  for (int i = 0; i < SIMD; i += CASCADE_NUM) {
    ap_int<PROD_BIT * 4> m = 0;
    for (int cs = 0; cs < CASCADE_NUM; cs++) {
      m += wpack[i + cs] * ipack[i + cs];
    }

    ap_int<PROD_BIT> p0 = m(PROD_BIT - 1, 0);
    ap_int<PROD_BIT> p1 = m(PROD_BIT * 2 - 1, PROD_BIT) + m[PROD_BIT - 1];
    ap_int<PROD_BIT> p2 =
        m(PROD_BIT * 3 - 1, PROD_BIT * 2) + m[PROD_BIT * 2 - 1];
    ap_int<PROD_BIT> p3 =
        m(PROD_BIT * 4 - 1, PROD_BIT * 3) + m[PROD_BIT * 3 - 1];

    r0 += p0;
    r1 += p1;
    r2 += p2;
    r3 += p3;
  }
  partial0 = r0;
  partial1 = r1;
  partial2 = r2;
  partial3 = r3;
}

template <unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT, unsigned SIMD>
void simd_MAC_compare(ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD],
                      ap_uint<PROD_BIT + IN_BIT> ipack[SIMD],
                      ap_int<W_BIT * SIMD> w0, ap_int<W_BIT * SIMD> w1,
                      ap_int<W_BIT * SIMD> w2, ap_uint<IN_BIT * SIMD> i0,
                      ap_uint<IN_BIT * SIMD> i1, ap_int<PROD_BIT + 5> &partial0,
                      ap_int<PROD_BIT + 5> &partial1,
                      ap_int<PROD_BIT + 5> &partial2,
                      ap_int<PROD_BIT + 5> &partial3) {

  ap_int<PROD_BIT + 5> r0, r1, r2, r3;
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;
  for (int i = 0; i < SIMD; i++) {

    ap_int<PROD_BIT * 4> m = wpack[i] * ipack[i];
    ap_int<PROD_BIT> p0 = m(PROD_BIT - 1, 0);
    ap_int<PROD_BIT> p1 = m(PROD_BIT * 2 - 1, PROD_BIT) + m[PROD_BIT - 1];
    ap_int<PROD_BIT> p2 =
        m(PROD_BIT * 3 - 1, PROD_BIT * 2) + m[PROD_BIT * 2 - 1];
    ap_int<PROD_BIT> p3 =
        m(PROD_BIT * 4 - 1, PROD_BIT * 3) + m[PROD_BIT * 3 - 1];

    ap_int<W_BIT> w0_seg = w0((i + 1) * W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w1_seg = w1((i + 1) * W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w2_seg = w2((i + 1) * W_BIT - 1, i * W_BIT);
    ap_uint<IN_BIT> x0_seg = i0((i + 1) * IN_BIT - 1, i * IN_BIT);
    ap_uint<IN_BIT> x1_seg = i1((i + 1) * IN_BIT - 1, i * IN_BIT);
    cout << "Weight :" << w0_seg << "," << w1_seg << "," << w2_seg << endl;
    cout << "Input :" << x0_seg << "," << x1_seg << endl;
    cout << "Wpack :" << wpack[i] << endl;
    cout << "Ipack :" << ipack[i] << endl;

    cout << r0 << "," << p0 << "," << x0_seg * w2_seg << endl;
    cout << r1 << "," << p1 << "," << x0_seg * w0_seg + x1_seg * w1_seg << endl;
    cout << r2 << "," << p2 << "," << x0_seg * w1_seg + x1_seg * w2_seg << endl;
    cout << r3 << "," << p3 << "," << x1_seg * w0_seg << endl;

    r0 += p0;
    r1 += p1;
    r2 += p2;
    r3 += p3;
  }
  partial0 = r0;
  partial1 = r1;
  partial2 = r2;
  partial3 = r3;
}

template <unsigned K, unsigned IN_BIT, unsigned IN_CH, unsigned OUT_BIT,
          unsigned OUT_W, unsigned OUT_H, unsigned OUT_CH, unsigned W_BIT,
          unsigned GUARD_BIT, unsigned M_BIT, unsigned INC_BIT,
          unsigned BIAS_BIT, unsigned SIMD, unsigned PE, unsigned L_SHIFT>
void convDSPOpt(
    stream<ap_uint<SIMD * IN_BIT * 2>> &vec,
    const ap_uint<SIMD * W_BIT> weights[PE][3][K * IN_CH / SIMD * OUT_CH / PE],
    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
    // stream<ap_uint<PE * OUT_BIT * 2>> &out,
    stream<ap_uint<PE * M_BIT * 2>> &out, const unsigned reps = 1) {

  static_assert(SIMD % 8 == 0, "SIMD mod 8 !=0");
  const unsigned PENUM = OUT_CH / PE;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned PROD_BIT = W_BIT + IN_BIT + GUARD_BIT;
  const unsigned WPACK_BIT = W_BIT * 3 + IN_BIT * 2 + GUARD_BIT * 2;
  const unsigned IPACK_BIT = IN_BIT * 2 + W_BIT + GUARD_BIT * 1;
  const unsigned INFOLD = K * SIMDNUM;

#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 2

  //   ap_int<PROD_BIT * 2> m_hi[INFOLD][PE][SIMD];
  // #pragma HLS ARRAY_PARTITION variable = m_hi complete dim = 2
  //   ap_int<PROD_BIT * 2> m_lo[PE][SIMD];
  // #pragma HLS ARRAY_PARTITION variable = m_lo complete dim = 1

  ap_int<WPACK_BIT> wpacks[PE][SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 2

  ap_uint<IPACK_BIT> ipack[SIMD];
#pragma HLS ARRAY_PARTITION variable = m_lo complete dim = 1

  // ap_uint<12> weightAddr = 0;
  ap_int<PROD_BIT + 4> firPartialRes0[PE];
  ap_int<PROD_BIT + 4> firPartialRes1[PE];

  ap_int<M_BIT> outPartialArr0[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr0 complete dim = 1
  ap_int<M_BIT> outPartialArr1[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr1 complete dim = 1

  for (unsigned int h = 0; h < OUT_H; h++) {
    for (unsigned int peIdx = 0; peIdx < PENUM; peIdx++) {
      for (unsigned int w = 0; w < OUT_W + K - 1; w += 2) {
        for (unsigned int infoldIdx = 0; infoldIdx < INFOLD; infoldIdx++) {
#pragma HLS DEPENDENCE variable = m_hi inter false
#pragma HLS DEPENDENCE variable = m_hi intra false

#pragma HLS pipeline
          bool m_clear = (w == 0);
          bool o_clear = (infoldIdx == 0);
          bool o_out = (infoldIdx == INFOLD - 1 && w != 0);
          ap_uint<SIMD * IN_BIT> data1, data0;
          (data1, data0) = vec.read();
          pack_input_data<IN_BIT, SIMD, PROD_BIT>(data1, data0, ipack);
          for (unsigned p = 0; p < PE; p++) {
            pack_weight_data<W_BIT, SIMD, PROD_BIT>(
                weights[p][2][peIdx * INFOLD + infoldIdx],
                weights[p][1][peIdx * INFOLD + infoldIdx],
                weights[p][0][peIdx * INFOLD + infoldIdx], wpacks[p]);
          }

          ap_int<PROD_BIT + 5> firPartial0[PE];
          ap_int<PROD_BIT + 5> firPartial1[PE];
          ap_int<PROD_BIT + 5> firPartial2[PE];
          ap_int<PROD_BIT + 5> firPartial3[PE];

          for (int p = 0; p < PE; p++) {
            cout << "FIR result compare " << endl;

            ap_int<PROD_BIT + 5> testPartial0;
            ap_int<PROD_BIT + 5> testPartial1;
            ap_int<PROD_BIT + 5> testPartial2;
            ap_int<PROD_BIT + 5> testPartial3;
            // simd_MAC<W_BIT, IN_BIT, PROD_BIT, SIMD, 4>(
            //     wpacks[p], ipack, firPartial0[p], firPartial1[p],
            //     firPartial2[p], firPartial3[p]);
            simd_MAC_compare<W_BIT, IN_BIT, PROD_BIT, SIMD>(
                wpacks[p], ipack, weights[p][0][peIdx * INFOLD + infoldIdx],
                weights[p][1][peIdx * INFOLD + infoldIdx],
                weights[p][2][peIdx * INFOLD + infoldIdx], data0, data1,
                testPartial0, testPartial1, testPartial2, testPartial3);

            getchar();

            if (o_clear) {
              outPartialArr0[p] = firPartial0[p] + firPartialRes0[p];
              outPartialArr1[p] = firPartial1[p] + firPartialRes1[p];
            } else {
              outPartialArr0[p] += firPartial0[p] + firPartialRes0[p];
              outPartialArr1[p] += firPartial1[p] + firPartialRes1[p];
            }
            firPartialRes0[p] = firPartial2[p];
            firPartialRes1[p] = firPartial3[p];
            // cout << outPartialArr1[p] << "," << outPartialArr0[p] << ",";
          }
          ap_int<M_BIT * PE> oData0;
          ap_int<M_BIT * PE> oData1;

          if (o_out) {
            ap_uint<PE * M_BIT> out_buf0;
            ap_uint<PE * M_BIT> out_buf1;
            for (int p = 0; p < PE; p++) {
              out_buf0(p * M_BIT + M_BIT - 1, p * M_BIT) = outPartialArr0[p];
              out_buf1(p * M_BIT + M_BIT - 1, p * M_BIT) = outPartialArr1[p];
            }
            out.write((out_buf1, out_buf0));
            // weightAddr = 0;
          }
        }
      }
    }
  }
}
/**
 * 矩阵向量计算单元
 * 同时进行量化激活处理
 */

// weights-> PE->SIMD->wr->col

/**
 * 卷积计算单元 同时计算bn_层与激活层
 * 在矩阵向量计算后立即计算得到激活输出值
 * 只计算 3x3 的卷积 K = 3, P = 1 S = 1
 * 输入数据宽度 为 IN_STREAM_BIT
 * 输出数据宽度为 PE * OUT_BIT
 */
template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,

          unsigned OUT_CH,
          unsigned OUT_BIT, // 量化激活后的位宽

          unsigned W_BIT, unsigned M_BIT, unsigned INC_BIT, unsigned BIAS_BIT,

          unsigned SIMD, unsigned PE, unsigned L_SHIFT>
void conv3x3_bn_act_DSPopt(
    stream<ap_uint<IN_BIT * IN_CH>> &in,
    const ap_uint<SIMD * W_BIT> weights[PE][3]
                                       [((IN_CH * 3) / SIMD) * (OUT_CH / PE)],
    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
    stream<ap_uint<OUT_BIT * OUT_CH>> &out, const unsigned reps = 1) {
#pragma HLS DATAFLOW

  const unsigned INTER_ROW = IN_ROW + 2;
  const unsigned INTER_COL = IN_COL + 2;
  // 暂时认为输入 输出维度不变
  const unsigned OUT_ROW = IN_ROW;
  const unsigned OUT_COL = IN_COL;

  // stream<ap_uint<IN_CH*IN_BIT> > in_adj("in_adj");
  // StreamingDataWidthConverter_Batch<IN_STREAM_BIT, IN_CH*IN_BIT>(in,
  // in_adj, reps); pading
  stream<ap_uint<IN_CH * IN_BIT>> padding_out("samepad_out");
  padding<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);

  stream<ap_uint<SIMD * IN_BIT * 2>> swu_reorder_out("swu_reorder_out");
  SWU_reordered<3, 1, INTER_ROW, INTER_COL, IN_CH, SIMD, IN_BIT, OUT_CH / PE>(
      padding_out, swu_reorder_out, reps);

  cout << "size " << swu_reorder_out.size() << endl;
  // print_SWU_stream_through<3, IN_ROW, IN_COL, IN_CH, SIMD, IN_BIT, OUT_CH /
  // PE>(
  //     swu_reorder_out, "swu_reorder_out.txt");
  stream<ap_uint<PE * M_BIT * 2>> mvau_out("mvau_out");
  convDSPOpt<3, IN_BIT, IN_CH, OUT_BIT, OUT_COL, OUT_ROW, OUT_CH, W_BIT, 3,
             M_BIT, INC_BIT, BIAS_BIT, SIMD, PE, L_SHIFT>(
      swu_reorder_out, weights, inc, bias, mvau_out);

  print_mavu_DSPopt_stream_through<OUT_ROW, OUT_COL, OUT_CH, PE, M_BIT>(
      mvau_out, "output.txt");
  // SWU<3, 1, INTER_ROW, INTER_COL, IN_CH, IN_BIT>(padding_out, swu_out, reps);
  // // 位宽调整
  // stream<ap_uint<SIMD * IN_BIT>> adj_out("adj_out");
  // StreamingDataWidthConverter_Batch<IN_CH * IN_BIT, SIMD * IN_BIT,
  //                                   9 * OUT_ROW * OUT_COL>(swu_out, adj_out,
  //                                                          reps);

  // cout << "adj_out size " << adj_out.size() << endl;
  // 矩阵向量计算
  // stream<ap_uint<PE * OUT_BIT>> mvau_out("mvau_out");
  // matrix_vector_act_unit<IN_CH * 3 * 3, OUT_CH, IN_BIT, OUT_BIT, W_BIT,
  // M_BIT,
  //                        INC_BIT, BIAS_BIT, SIMD, PE, L_SHIFT,
  //                        OUT_ROW * OUT_COL>(adj_out, weights, inc, bias,
  //                                           mvau_out, reps);
  // // cout << "mvau_out size " << mvau_out.size() << endl;
  // StreamingDataWidthConverter_Batch<PE*OUT_BIT, OUT_CH*OUT_BIT, OUT_ROW
  // * OUT_COL * OUT_CH / PE>(mvau_out, out, reps);
}

#endif