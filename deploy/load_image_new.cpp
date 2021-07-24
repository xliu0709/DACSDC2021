#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <malloc.h>
#include <stdint.h>
using namespace cv;
using namespace std;

#define NUM_THREAD 4
#define ROW_SO 360
#define COL_SO 640
#define CH_SO 3

pthread_mutex_t lock_ping[NUM_THREAD];
pthread_mutex_t lock_pong[NUM_THREAD];

char** path_global; 

Mat image_ping[NUM_THREAD];

Mat image_pong[NUM_THREAD];



typedef struct {
    int thr;
    int idx;
    bool pingpong;
} ThrParaDecode;

typedef struct {
    uint8_t *mem;
    int thr;
    int idx;
    bool pingpong;
} ThrParaCopy;

inline void mat_to_arr(Mat & image, uint8_t* arr)
{
    // int nWidth = image.cols;
    // int nHeight = image.rows;
    // int nBandNum = image.channels();
    // int nBPB = 1;
    // size_t line_size = COL_SO * CH_SO、;
    uint8_t* data = image.data;
    uint8_t* pSubBuffer= arr;
    for (int j = 0; j < ROW_SO*COL_SO*CH_SO/16; j ++) {
        memcpy(pSubBuffer, data, 16);
        data+=16;
        pSubBuffer+=16;
    }
}

 
void *decode_image_phread(void * thread_para){
    ThrParaDecode para = *((ThrParaDecode *)thread_para);
    if(para.pingpong)
    {
        image_ping[para.thr] = imread(path_global[para.idx]);
    }
    else
    {
        image_pong[para.thr] = imread(path_global[para.idx]);
    }
}

void *copy_image_phread(void * thread_para){
    ThrParaCopy para = *((ThrParaCopy *)thread_para);
    uint8_t* mem=para.mem+para.idx*ROW_SO*COL_SO*CH_SO;
    
    if(para.pingpong)
    {
        mat_to_arr(image_ping[para.thr],mem);
//         memcpy(image_ping[para.thr].data, mem, ROW_SO*COL_SO*CH_SO);
        
    }
    else
    {
         mat_to_arr(image_pong[para.thr],mem);
//         memcpy(image_pong[para.thr].data, mem, ROW_SO*COL_SO*CH_SO);
    }
}



void load_image_cpp(char **paths, uint8_t *matrix, int batch_size, int row, int col, int ch) {
    pthread_t pthread_decode[NUM_THREAD];
    pthread_t pthread_copy[NUM_THREAD];

    ThrParaDecode node_decode[NUM_THREAD];
    ThrParaCopy node_copy[NUM_THREAD];
    
    path_global = paths;

    void ** status;
  



    for(int i=0;i<NUM_THREAD;i++)
    {
        node_decode[i].thr=i;
        node_decode[i].idx=i;
        node_decode[i].pingpong=true;
    }
    for(int i=0;i<NUM_THREAD;i++)
    {
        if(node_decode[i].idx<batch_size)
        {
            pthread_create(&pthread_decode[i], NULL, decode_image_phread, (void *)&node_decode[i]);
        }
    }
    for(int i=0;i<NUM_THREAD;i++)
    {
        node_copy[i].pingpong=true;
        node_copy[i].thr=i;
        node_copy[i].idx=i;
    }
    for(int i=0;i<NUM_THREAD;i++)
    {
        pthread_join(pthread_decode[i], status);
        node_decode[i].idx+=NUM_THREAD;
        node_decode[i].pingpong=!node_decode[i].pingpong;
    }
    
    while(node_copy[0].idx<batch_size)
    {
        for(int i=0;i<NUM_THREAD;i++)
        {
            if(node_decode[i].idx<batch_size)
            {
                pthread_create(&pthread_decode[i], NULL, decode_image_phread, (void *)&node_decode[i]);
                
            }
            if(node_copy[i].idx<batch_size)
            {
                pthread_create(&pthread_copy[i], NULL, copy_image_phread, (void *)&node_copy[i]);
            }
        }
//         cout<<node_copy[0].idx<<":";
//         fflush(stdout);

        for(int i=0;i<NUM_THREAD;i++)
        {
            if(node_decode[i].idx<batch_size)
            {
                pthread_join(pthread_decode[i], status);
                node_decode[i].idx+=NUM_THREAD;
                node_decode[i].pingpong=!node_decode[i].pingpong;
            }
            if(node_copy[i].idx<batch_size)
            {

                pthread_join(pthread_copy[i], status);
//                 cout<<i<<" ";
//                 fflush(stdout);
                node_copy[i].idx+=NUM_THREAD;
                node_copy[i].pingpong=!node_copy[i].pingpong;
            }
        }
//         cout<<endl;
    }
    


    
    // 删除属性，并等待其他线程
    // pthread_attr_destroy(&attr);
    // for (int i=0; i<NUM_THREAD; i++)
    // {
    //     pthread_join(pthread_decode[i], status);
    //     pthread_join(pthread_copy[i], status);
    //     pthread_mutex_destroy(&lock_ping[i]);
    //     pthread_mutex_destroy(&lock_pong[i]);
    // }
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
