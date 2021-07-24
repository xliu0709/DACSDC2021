import sys
import os

# sys.path.append(os.path.abspath("../common"))

import math
import time
import numpy as np
import cv2
import pynq
import dac_sdc
import ctypes

team_name = 'iSmart'
team = dac_sdc.Team(team_name, batch_size = 20)




# #################################

# last_bais = np.load('last_bias.npy')
# last_bais = last_bais.reshape((6, 6))[:, :4]
# os.system('g++ -shared -O2 ./load_image_new.cpp -o load_image_new.so -fPIC $(pkg-config opencv --cflags --libs) -lpthread')
cfuns = ctypes.cdll.LoadLibrary("./load_image_new.so")

overlay = pynq.Overlay(team.get_bitstream_path())
dma = overlay.axi_dma_0
xlnk = pynq.Xlnk()
nn_ctrl = overlay.ultra_net_0
print('got nn accelerator!')




#######################
BATCH_SIZE = team.batch_size
IMAGE_RAW_ROW = 360
IMAGE_RAW_COL = 640
IMAGE_ROW = 160
IMAGE_COL = 320
GRID_ROw = 10
GRID_COL = 20
X_SCALE = IMAGE_RAW_COL / IMAGE_COL
Y_SCALE = IMAGE_RAW_ROW / IMAGE_ROW


in_buffer0 = xlnk.cma_array(shape=(BATCH_SIZE, IMAGE_RAW_ROW, IMAGE_RAW_COL, 3), dtype=np.uint8, cacheable = 1)
in_buffer1 = xlnk.cma_array(shape=(BATCH_SIZE, IMAGE_RAW_ROW, IMAGE_RAW_COL, 3), dtype=np.uint8, cacheable = 1)
in_buffers = [in_buffer0, in_buffer1]
out_buffer0 = xlnk.cma_array(shape=(BATCH_SIZE, GRID_ROw, GRID_COL, 6, 6), dtype=np.int32, cacheable = 1)
out_buffer1 = xlnk.cma_array(shape=(BATCH_SIZE, GRID_ROw, GRID_COL, 6, 6), dtype=np.int32, cacheable = 1)
out_buffers = [out_buffer0, out_buffer1]


# use c code load image
def load_image(image_paths, buff):
    paths = [str(path) for path in image_paths]
    tmp = np.asarray(buff)
    dataptr = tmp.ctypes.data_as(ctypes.c_char_p)
    paths_p_list = [ctypes.c_char_p(bytes(str_, 'utf-8')) for str_ in paths]
    paths_c = (ctypes.c_char_p*len(paths_p_list))(*paths_p_list)
    cfuns.load_image(paths_c, dataptr, len(paths), IMAGE_ROW, IMAGE_COL, 3)
    
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def yolo(out_buffer, batch_n, div, result=None):
    res_np = np.array(out_buffer[:batch_n]).reshape(batch_n, -1, 6, 6)
    conf = res_np[...,4].sum(axis=2)
    max_index = conf.argmax(1)
    
    grid_x = max_index % GRID_COL
    grid_y = max_index // GRID_COL
    
    boxs = np.zeros((batch_n, 6, 4))
    for i in range(batch_n):
        boxs[i, :, :] = res_np[i, max_index[i], :, :4] / div
    xy = sigmoid(boxs[..., :2]).mean(axis=1)
    wh = np.exp(boxs[..., 2:4]).mean(axis=1)
    
    xy[:, 0] += grid_x
    xy[:, 1] += grid_y

    xy *= 16
    wh *= 20

    xy[:, 0] *= X_SCALE
    xy[:, 1] *= Y_SCALE
    wh[:, 0] *= X_SCALE
    wh[:, 1] *= Y_SCALE
    xmin = xy[:, 0] - wh[:, 0] / 2
    xmax = xy[:, 0] + wh[:, 0] / 2
    ymin = xy[:, 1] - wh[:, 1] / 2
    ymax = xy[:, 1] + wh[:, 1] / 2
    
    for i in range(batch_n):
        temp = [int(xmin[i]), int(xmax[i]), int(ymin[i]), int(ymax[i])]
        result.append(temp)
#         result.append([0,0,0,0])
    
which_buffer = 0
first_batch = True
net_cnt = 0
last_batch_size = BATCH_SIZE

def net(img_paths, result):
    global first_batch
    global which_buffer    
    global net_cnt
    global last_batch_size
    # buffer first batch
    if first_batch == True:
        first_batch = False
        which_buffer = 0
        load_image(img_paths, in_buffers[which_buffer])
#         print("load_image done")
        return
    # count
    net_cnt += 1
    nn_ctrl.write(0x0, 0) # Reset
    nn_ctrl.write(0x10, in_buffers[which_buffer].shape[0])
    nn_ctrl.write(0x0, 1) # Deassert reset
    
#     print("starting FPGA")
    dma.recvchannel.transfer(out_buffers[which_buffer])
    dma.sendchannel.transfer(in_buffers[which_buffer])
#     print("ack done")
    
    # switch buffer
    if which_buffer == 0:
        which_buffer = 1
    else:
        which_buffer = 0
    # buffer next batch

    if img_paths is not None:
        load_image(img_paths, in_buffers[which_buffer])
#     print("load_image done")
    
    # yolo 
    if net_cnt > 1:
        yolo(out_buffers[which_buffer], BATCH_SIZE, 7 * 15, result)
    
    if img_paths is not None and len(img_paths) != BATCH_SIZE:
        last_batch_size = len(img_paths)
#     print("ending FPGA")
    dma.sendchannel.wait()
    dma.recvchannel.wait()
#     print("ack done")
    # last batch 
    if img_paths is None:
        yolo(out_buffers[(which_buffer + 1) % 2], last_batch_size, 7 * 15, result) 
        
################################Inference##################################
interval_time = 0
total_time = 0
total_energy = 0
result = list()
team.reset_batch_count()

rails = pynq.get_rails()

start = time.time()    
recorder = pynq.DataRecorder(rails["5V"].power)
with recorder.record(0.05): 
    while True:
        image_paths = team.get_next_batch()
        net(image_paths, result)

#         print('pro_image_cnt', len(result))
        # end
        if image_paths is None:
            break

end = time.time()
t = end - start
    
# Energy measurements    
energy = recorder.frame["5V_power"].mean() * t    
# energy = 0

total_time = t
total_energy = energy
print("Total time:", total_time, "seconds")
print("Total energy:", total_energy, "J")
print('images nums: {} .'.format(len(result)))
print('fps: {} .'.format(len(result) / total_time))



##############################################################################
team.save_results_xml(result, total_time, energy)
xlnk.xlnk_reset()




##############################################################################

def computeIoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    if( xB < xA or yB < yA ):
        return 0.0

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou



#f_PL = open("iSmart2.txt", "r")
f_PL = open("../deploy/bbox_PL_1000_sample.txt", "r")
#f_PL = open("bbox_PL_1000.txt", "r")
#f_PL = open("bbox_PL_1000_sample_64_py2.txt", "r")
#f_PL = open("bbox_golden_c_1000_sample_512_py3.txt", "r")
#f_PL = open("bbox_golden_c_1000.txt", "r")
f_GT = open("../deploy2/bbox_GT_1000_sample.txt", "r")
#f_GT = open("bbox_GT_1000.txt", "r")

#f_PL = open("bbox_pytorch.txt", "r")
#f_PL = open("bbox_golden_C.txt", "r")

bbox_PL = []
bbox_GT = []


for line in f_PL:
    if not line.strip():
        continue
    if(line.endswith("jpg\n") or line.endswith("xml\n") or line.endswith("bin\n")):
        continue
        
    x1, x2, y1, y2 = line.split()
    bbox_PL.append([int(x1), int(y1), int(x2), int(y2)])
    


for line in f_GT:
    if not line.strip():
        continue
    if(line.endswith("jpg\n") or line.endswith("xml\n") or line.endswith("bin\n")):
        continue
        
    x1, x2, y1, y2 = line.split()
    bbox_GT.append([int(x1), int(y1), int(x2), int(y2)])
    

#if( len(bbox_PL) != len(bbox_GT) ):
#    print("ERROR! Ground truth and PL output do not match!")
    
cnt = len(bbox_PL)
IoU_avg = 0
for i in range(cnt):
    boxA = bbox_PL[i]
    boxB = bbox_GT[i]
    
#     print('\n')
#     print(i)
#     print(boxA)
#     print(boxB)
    
    IoU = computeIoU(boxA, boxB)
#     print(IoU)
    
    IoU_avg = IoU_avg + IoU

print("\nAvg IOU:")
print(IoU_avg/cnt)



