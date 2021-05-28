#ifndef __DEBUG_HPP__
#define __DEBUG_HPP__

#include <ap_int.h>
#include <fstream>
#include <hls_stream.h>
#include <string>
#include <iomanip>


using namespace std;
template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned OUT_BIT>
void print_FM_stream(hls::stream<ap_uint<OUT_BIT * OUT_CH>> &in,
                     string filename) {
  ofstream f(filename);

  for (int r = 0; r < OUT_ROW; r++)
    for (int c = 0; c < OUT_COL; c++) {
      f.width(10);
      f << '[' << r << "," << c << "]";
      ap_uint<OUT_BIT *OUT_CH> data = in.read();

      for (int d = 0; d < OUT_CH; d++) {
        ap_int<OUT_BIT> wdata = data(d * OUT_BIT + OUT_BIT - 1, d * OUT_BIT);
        f << wdata;
      }
      f << endl;
    }
  f.close();
}

template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned OUT_BIT>
void print_FM_stream_through(hls::stream<ap_uint<OUT_BIT * OUT_CH>> &out,
                             string filename) {
  ofstream f(filename);

  for (int r = 0; r < OUT_ROW; r++)
    for (int c = 0; c < OUT_COL; c++) {
      f << '[' << setw(3) << r << "," << setw(3) << c << "]";
      ap_uint<OUT_BIT *OUT_CH> data = out.read();
      out.write(data);
      for (int d = 0; d < OUT_CH * OUT_BIT / 8; d++) {
        ap_uint<8> wdata = data(d * 8 + 7, d * 8);
        f << setw(4) << wdata;
      }
      f << endl;
    }
  f.close();
}

template <unsigned K, unsigned ROW, unsigned COL, unsigned CH, unsigned SIMD,
          unsigned BIT, unsigned PENUM>
void print_SWU_stream_through(hls::stream<ap_uint<BIT * SIMD * 2>> &out,
                              string filename) {
  ofstream f(filename);
  for (int r = 0; r < ROW; r++)
    for (int peIdx = 0; peIdx < PENUM; peIdx++)
      for (int c = 0; c < COL + K - 1; c += 2) {
        for (int kh = 0; kh < K; kh++)
          for (int cs = 0; cs < CH / SIMD; cs++) {
            f << '[' << setw(3) << r + kh << "," << setw(3) << c << ","
              << setw(3) << cs << "]";
            ap_uint<BIT *SIMD * 2> data = out.read();
            // out.write(data);
            for (int d = 0; d < BIT * SIMD * 2 / 8; d++) {
              ap_uint<8> wdata = data(d * 8 + 7, d * 8);
              f << setw(4) << wdata;
            }
            f << endl;
          }
      }
  f.close();
}

template <unsigned ROW, unsigned COL, unsigned CH, unsigned PE, unsigned BIT>
void print_mavu_DSPopt_stream_through(hls::stream<ap_uint<BIT * PE * 2>> &out,
                                      string filename) {
  ofstream f(filename);
  ap_uint<BIT * PE> buffer[CH / PE][COL];

  for (int r = 0; r < ROW; r++) {
    for (int peIdx = 0; peIdx < CH / PE; peIdx++) {
      for (int c = 0; c < COL; c += 2) {
        ap_uint<BIT *PE * 2> data = out.read();
        buffer[peIdx][c] = data(BIT * PE - 1, 0);
        buffer[peIdx][c + 1] = data(BIT * PE * 2 - 1, BIT * PE);
      }
    }
    for (int c = 0; c < COL; c++) {
      f << "[" << setw(4) << r << "," << setw(4) << c << "]";
      for (int peIdx = 0; peIdx < CH / PE; peIdx++) {
        for (int p = 0; p < PE; p++) {
          unsigned data = buffer[peIdx][c].range(p * BIT + BIT - 1, p * BIT);
          f << setw(8) << hex << data << ",";
        }
      }
      f << endl;
    }
  }
  f.close();
}
template <unsigned ROW, unsigned COL, unsigned CH, unsigned PE, unsigned BIT>
void print_mavu_stream_through(hls::stream<ap_uint<BIT * CH>> &out,
                               string filename) {
  ofstream f(filename);

  for (int r = 0; r < ROW; r++) {
    for (int c = 0; c < COL; c++) {
      f << "[" << setw(4) << r << "," << setw(4) << c << "]";
      ap_uint<PE *CH> outdata = out.read();
      for (int c = 0; c < CH; c++) {
        unsigned data = outdata.range(c * BIT + BIT - 1, c * BIT);
        f << setw(8) << hex << data << ",";
      }
      f << endl;
    }
    f.close();
  }
}
#endif
