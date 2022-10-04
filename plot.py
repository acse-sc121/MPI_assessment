import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#define process
p = 4

def find_size(p):
    """
    Divide the total process into rows*cols
    """
    min_gap = p
    top =  math.floor(math.sqrt(p) + 1)
    for i in range(1, top+1):
        if (p % i == 0):
            gap = math.floor(abs(p / i - i))
            if (gap < min_gap):
                min_gap = gap
                rows = i
                columns = math.floor(p / i)
    return rows, columns
rows, cols = find_size(p)

#This python file is placed at the upper level of out folder
dir = "./Debug/out"
num_files = 0
for path in os.listdir(dir):
    if os.path.isfile(os.path.join(dir, path)):
        num_files += 1
print("There are %d files in the folder!"%num_files)
num = math.floor(num_files/p)

for cnt in range(num):
    square = [rows, cols]
    for i in range(rows):
        square[i] = []
        for j in range(cols):
            line = pd.read_table("./Debug/out/id_%d_output_%d.dat" % (j + i * cols, cnt), sep = '\t', header = None)
            #Add data in each row
            square[i].append(line)
            line.drop(line.columns[len(line.columns)-1], axis=1, inplace=True)
        
    images = pd.DataFrame()
    #Combine into images
    for i in range(rows):
        pd_row = pd.DataFrame()
        for j in range(cols):
            pd_row = pd.concat([pd_row,square[i][j]], axis=1, ignore_index=True)
        images = pd.concat([images, pd_row], ignore_index=True)
    #Write the combined data into files
    #Need to create a new folder to save the outputs
    images.to_csv("./Debug/out_to/output_%d.dat" % (cnt), sep='\t', index=0, header=0)
    
#Create GIF
fig = plt.figure()

img_list = []

for i in range(num):
    output =  pd.read_table("./Debug/out_to/output_%d.dat" % (i), sep = '\t', header=None)
    data = output.values
    img_list.append(data)
    
img_num = len(img_list)

im = plt.imshow(img_list[0], animated=True)

def outfig(frame,*args):
    im.set_array(img_list[frame])
    return im

gif = animation.FuncAnimation(fig, outfig, frames=img_num, interval=50, blit=True)
gif.save('result.gif', writer='pillow')
plt.show()
