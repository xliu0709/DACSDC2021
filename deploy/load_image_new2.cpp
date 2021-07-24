#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <malloc.h>
#include <stdint.h>
#include <iterator>
using namespace cv;
using namespace std;

#define NUM_THREAD 4

#define ROW_SO 360
#define COL_SO 640
#define CH_SO 3

typedef struct {
    int thr;
    char **paths;
    uint8_t *matrix;
    int start;
    int end;

    int batch_size;
} ThrPara;

void mat_to_arr(Mat & image, uint8_t* arr)
{
    uint8_t* data = image.data;
    uint8_t* pSubBuffer= arr;
    for (int j = 0; j < ROW_SO*COL_SO*CH_SO/256; j ++) {
        memcpy(pSubBuffer, data, 256);
        data+=256;
        pSubBuffer+=256;
    }
}

void *load_image_pthread(void * thread_para) {
    ThrPara para = *((ThrPara *)thread_para);
    uint8_t * arr = para.matrix+para.start*COL_SO * ROW_SO * CH_SO;
    
    Mat image(ROW_SO, COL_SO, CV_8UC3, arr);
    uint8_t buffer[COL_SO * ROW_SO* CH_SO];
    for (int i=para.start; i < para.end; i ++) {
        // printf("%s \n", para.paths[i]);
       
        FILE *fp = fopen(para.paths[i], "rb");

        int bufferLength= fread(buffer, 1, COL_SO * ROW_SO * CH_SO, fp);

        imdecode(cv::Mat(1, bufferLength, CV_8UC1, buffer), 1, &image);
    
        image.data+=COL_SO * ROW_SO * CH_SO;
        fclose(fp);
       
       
//         mat_to_arr(image, arr);
    }

}
 
void load_image_cpp(char **paths, uint8_t *matrix, int batch_size, int row, int col, int ch) {
    pthread_t pthread[NUM_THREAD];
    ThrPara node[NUM_THREAD];
    pthread_attr_t attr; // 定义线程属性
    void ** status;
    int index[NUM_THREAD];
    // 初始化并设置线程为可连接
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    float step = batch_size * 1.0 / NUM_THREAD;
    for (int i=0; i<NUM_THREAD; i++)
    {
        node[i].thr = i;
        node[i].start = (int) (step * i + 0.01);
        if (i < NUM_THREAD - 1) {
            node[i].end = (int)(step * (i + 1) + 0.01);
        } else {
            node[i].end = batch_size;
        }
        node[i].paths = paths;
        node[i].matrix = matrix;
        node[i].batch_size = batch_size;
        pthread_create(&pthread[i], NULL, load_image_pthread, (void *)&node[i]);
    }

    // 删除属性，并等待其他线程
    pthread_attr_destroy(&attr);
    for (int i=0; i<NUM_THREAD; i++)
    {
        int res = pthread_join(pthread[i], status);
        if (0 != res)
        {
            exit(-1);
        }
        
    }
}

extern "C" {
    void load_image(char **paths, uint8_t *matrix, int batch_size, int row, int col, int ch) {
        load_image_cpp(paths, matrix, batch_size, row, col, ch);
    }
}


// int main()
// {
//     int batch_size = 100;
//     uint8_t * arr = (uint8_t *) malloc(batch_size * 360 * 640 * 3);
//     char * paths[batch_size];
//     for (int i=0; i < batch_size; i ++) {
//         paths[i] = "0.jpg";
//     }

//     load_image(paths, arr, batch_size, 360, 640, 3);
//     cout << " end " << endl;
//     // for (int i=0; i < 100; i ++) {
//     //     Mat image = imread("0.jpg");
//     //     int nWidth = image.cols;
//     //     int nHeight = image.rows;
//     //     int nBandNum = image.channels();    

        
//     //     mat_to_arr(image, arr);
        
//     //     cout << nWidth << "  " << nHeight << " " << nBandNum << endl;

//     // }
//     return 0;
// }
