import sys
import os

class Team:
    def __init__(self, team_name, batch_size):
        self.batch_size=batch_size
        IMG_DIR = '/home/xilinx/jupyter_notebooks/dac_sdc_2020/images2/'

        names_temp = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
        names_temp.sort(key= lambda x:int(x[:-4]))
            
        for i,k in enumerate(names_temp):
            names_temp[i]=IMG_DIR+k
            
        self.names_temp=[]
        
        for i in range(1):
            self.names_temp+=names_temp;
#         self.names_temp+=names_temp[0:500]
        self.start_idx=-batch_size;

    def get_next_batch(self):
        self.start_idx+=self.batch_size;
        if(self.start_idx>= len(self.names_temp) ):
            self.start_idx=-self.batch_size
            return None
        elif( self.start_idx+self.batch_size<=len(self.names_temp) ):
            return self.names_temp[self.start_idx:self.start_idx+self.batch_size]
        else:
            return self.names_temp[self.start_idx:]
        
    def reset_batch_count(self):
        self.start_idx=-self.batch_size
    def get_bitstream_path(self):
        return "./fastNet8bit.bit"
    def save_results_xml(self,result_rectangle, total_time, energy):
        f_out = open('./bbox_PL_1000_sample.txt', 'w')
        cnt = 0


        for box in result_rectangle:
            x1 = box[0]
            x2 = box[1]
            y1 = box[2]
            y2 = box[3]
            coord = str(x1) + ' ' + str(x2) + ' ' + str(y1) + ' ' + str(y2)

            #name = names[cnt]
            cnt = cnt + 1
            #f_out.write(name + '\n')
            f_out.write(coord + '\n')

        f_out.close()
        print("\nAll results stored in bbox_PL_1000_sample.txt")