#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <malloc.h>
#include <stdint.h>
#include <iterator>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstdint>
#include <cmath>
#include <signal.h> 

using namespace cv;
using namespace std;

#define NUM_THREAD 16
#define ROW_SO 360
#define COL_SO 640
#define CH_SO 3


#define FIRSTLASTBITWIDTH 8
#define GRIDROW 10
#define GRIDCOL 20
#define IMAGE_RAW_ROW  360
#define IMAGE_RAW_COL  640
#define IMAGE_ROW  160
#define IMAGE_COL  320


#define X_SCALE   (IMAGE_RAW_COL / IMAGE_COL)
#define Y_SCALE   (IMAGE_RAW_ROW / IMAGE_ROW)


inline float sigmoid(float x){
    return 1/(1 +expf(-x) );  
}
    

void yolo_cpp(int32_t *vec, int batch_n, int32_t* rst)
{
    
       for(int bc=0;bc<batch_n;bc++)
    {
        int32_t (*castarr)[6][6]= (int32_t (*)[6][6] ) vec;


        int max_idx=0;
        int max_sum=INT32_MIN;

        for(int i=0;i<GRIDROW*GRIDCOL;i++)
        {
            int sum=castarr[i][0][4]+castarr[i][1][4]+castarr[i][2][4]+castarr[i][3][4]+castarr[i][4][4]+castarr[i][5][4];
            if(sum>max_sum)
            {
                sum=max_sum;
                max_idx=i;
            }
        }

        float xy[2]={0,0};
        float wh[2]={0,0};

        
        for(int i=0;i<6;i++)
        {
            xy[0]+=sigmoid(castarr[max_idx][i][0]);
            xy[1]+=sigmoid(castarr[max_idx][i][1]);
            wh[0]+=expf(castarr[max_idx][i][2]);
            wh[1]+=expf(castarr[max_idx][i][3]);
        }
        xy[0]=xy[0]/6;
        xy[1]=xy[1]/6;
        wh[0]=wh[0]/6;
        wh[1]=wh[1]/6;

        xy[0] *= X_SCALE;
        xy[1] *= Y_SCALE;
        wh[0] *= X_SCALE;
        wh[1] *= Y_SCALE;

        float xmin = xy[0] - wh[0] / 2;
        float xmax = xy[0] + wh[0] / 2;
        float ymin = xy[1] - wh[1] / 2;
        float ymax = xy[1] + wh[1] / 2;

        rst[0]=xmin;
        rst[1]=xmax;
        rst[2]=ymin;
        rst[3]=ymax;
        rst+=4;
        vec+= GRIDCOL*GRIDROW*36;
    }
}



uint8_t* buffer_ping[NUM_THREAD];
uint8_t* buffer_pong[NUM_THREAD];


typedef struct {
    int thr;
    int start;
    int end;
    char **paths;
    uint8_t *matrix;
} ThrPara;





pthread_t read_pthread[NUM_THREAD];
pthread_t decode_pthread[NUM_THREAD];

void *read_image_pthread(void *thread_para){
    ThrPara para = *((ThrPara *)thread_para);
    
    uint8_t* buffer0=buffer_ping[para.thr];
    uint8_t* buffer1=buffer_pong[para.thr];
    
    
    sigset_t set1,set2;  
    int sig;
    sigemptyset(&set1);                                                             
    sigaddset(&set1, SIGUSR1);
    sigemptyset(&set2);                                                             
    sigaddset(&set2, SIGUSR2);
    
    bool pingpong=true;

    
    for (int i=para.start; i < para.end; i ++) {
        // printf("%s \n", para.paths[i]);
    
        if(pingpong)
        {
            sigwait(&set1,&sig);
            FILE *fp = fopen(para.paths[i], "rb");
            fread(buffer0, 1, COL_SO * ROW_SO * CH_SO, fp);
            fclose(fp);
            pthread_kill( decode_pthread[para.thr], SIGUSR1);
            
        }
        else
        {
            sigwait(&set2,&sig);
            FILE *fp = fopen(para.paths[i], "rb");
            fread(buffer1, 1, COL_SO * ROW_SO * CH_SO, fp);
            fclose(fp);
            pthread_kill( decode_pthread[para.thr], SIGUSR2);   
        }
        pingpong=!pingpong;
    }
}


void *decode_image_pthread(void * thread_para) {
    ThrPara para = *((ThrPara *)thread_para);
    uint8_t * arr = para.matrix+para.start*COL_SO * ROW_SO * CH_SO;
    uint8_t* buffer0=buffer_ping[para.thr];
    uint8_t* buffer1=buffer_pong[para.thr];
    
    sigset_t set1,set2;  
    int sig;
    sigemptyset(&set1);                                                             
    sigaddset(&set1, SIGUSR1);
    sigemptyset(&set2);                                                             
    sigaddset(&set2, SIGUSR2);
    
    Mat image(ROW_SO, COL_SO, CV_8UC3, arr);
    bool pingpong=true;
    for (int i=para.start; i < para.end; i ++) {
        if(pingpong)
        {
            sigwait(&set1,&sig);
            imdecode(cv::Mat(1, COL_SO * ROW_SO * CH_SO, CV_8UC1, buffer0), 1, &image);
            image.data += COL_SO * ROW_SO * CH_SO;
            pthread_kill( read_pthread[para.thr], SIGUSR1);
        }
        else
        {
            sigwait(&set2,&sig);
            imdecode(cv::Mat(1, COL_SO * ROW_SO * CH_SO, CV_8UC1, buffer1), 1, &image);
            image.data += COL_SO * ROW_SO * CH_SO;
            pthread_kill( read_pthread[para.thr], SIGUSR2);
        }
        pingpong=!pingpong;
    }

}

void my_handler(int signal) {
}
               

void load_image_cpp(char **paths, uint8_t *matrix, int batch_size, int row, int col, int ch) {

    
    
    
    ThrPara node[NUM_THREAD];
    pthread_attr_t attr; // 定义线程属性
    void ** status;
    
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    float step = batch_size * 1.0 / NUM_THREAD;


    signal(SIGUSR1,my_handler);
    signal(SIGUSR2,my_handler);
                                                    
    sigset_t    signal_mask;
    sigemptyset(&signal_mask);
    sigaddset(&signal_mask, SIGUSR1);
    sigaddset(&signal_mask, SIGUSR2);
    pthread_sigmask (SIG_BLOCK, &signal_mask, NULL);
    
    
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
        buffer_ping[i]=(uint8_t*) malloc(COL_SO * ROW_SO* CH_SO);
        buffer_pong[i]=(uint8_t*) malloc(COL_SO * ROW_SO* CH_SO);
        pthread_create(&read_pthread[i], NULL, read_image_pthread, (void *)&node[i]);
        pthread_create(&decode_pthread[i], NULL, decode_image_pthread, (void *)&node[i]);
        pthread_kill( read_pthread[i], SIGUSR1);  
        pthread_kill( read_pthread[i], SIGUSR2);  
    }

    // 删除属性，并等待其他线程
    pthread_attr_destroy(&attr);
    for (int i=0; i<NUM_THREAD; i++)
    {
        pthread_join(read_pthread[i], status);
        pthread_join(decode_pthread[i], status);

        free(buffer_ping[i]);
        free(buffer_pong[i]);
    }
}

extern "C" {
    void load_image(char **paths, uint8_t *matrix, int batch_size, int row, int col, int ch) {
        load_image_cpp(paths, matrix, batch_size, row, col, ch);
    }
    void yolo(int32_t *vec, int batch_n, float* rst){
        yolo(vec, batch_n, rst);
    }
    
}
