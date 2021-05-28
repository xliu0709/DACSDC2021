
#pragma once
#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "sliding_window_unit.h"
#include "stream_tools.h"

template <unsigned IN_BIT, unsigned PE>
ap_uint<IN_BIT * PE> max2_PE(ap_uint<IN_BIT * PE> data0,
                             ap_uint<IN_BIT * PE> data1) {
  ap_uint<IN_BIT * PE> ret;

  for (int i = 0; i < PE; i++) {
    ap_uint<IN_BIT> d0 = data0(IN_BIT * (i + 1) - 1, IN_BIT);
    ap_uint<IN_BIT> d1 = data1(IN_BIT * (i + 1) - 1, IN_BIT);
    ap_uint<IN_BIT> dret = d1 > d0 ? d1 : d0;
    ret(IN_BIT * (i + 1) - 1, IN_BIT) = dret;
  }
  return ret;
}

template <unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned IN_BIT,
          unsigned PE>
void max_pool2x2(stream<ap_uint<PE * IN_BIT * 2>> &vec,
                 stream<ap_uint<PE * IN_BIT>> &out, const unsigned reps = 1) {

  ap_uint<PE * IN_BIT> row_store[IN_W / 2 * IN_CH / PE];

  bool load_flag;
  ap_uint<IN_BIT * PE> dataOut0;
  ap_uint<IN_BIT * PE> dataOut1;
  for (int h = 0; h < IN_H; h++)
    for (int peIdx = 0; peIdx < IN_CH / PE; peIdx++)
      for (int w = 0; w < IN_W / 2; w++) {
#pragma HLS pipeline II = 1
        ap_uint<IN_BIT * PE> data0;
        ap_uint<IN_BIT * PE> data1;
        (data1, data0) = vec.read();
        ap_uint<IN_BIT *PE> dataMax2 = max2_PE<IN_BIT, PE>(data0, data1);
        int addr = w * (IN_CH / PE) + peIdx;
        if (h % 2) {
          ap_uint<IN_BIT *PE> dataRes = row_store[addr];
          dataOut0 = max2_PE<IN_BIT, PE>(dataMax2, dataRes);

        } else {
          row_store[addr] = dataMax2;
        }
        if (w % 2 && h % 2) {
          dataOut1 = dataOut0;
        } else {
          out.write((dataOut1, dataOut0));
        }
      }
}