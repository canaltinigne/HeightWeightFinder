import matplotlib
import random
import json
import numpy as np
import cv2

def create_colors(n):
    
    cmap = matplotlib.cm.get_cmap('gist_ncar')
    colors = []

    for i in range(n):
        colors.append(cmap(random.random()))
        
    return colors

def draw_skeleton(mask_img, joint_pos, colors):
        
    neighbors = {
        0: [1,14,15], 
        1: [2,5,8,11], 
        2: [3], 
        3: [4], 
        5: [6], 
        6: [7], 
        8: [9], 
        9: [10], 
        11: [12],
        12: [13], 
        14: [16], 
        15: [17]
    }
    
    for pos in joint_pos:
        cl = 0

        for point in neighbors:
            if pos[point] != (0,0):
                for neighbor in neighbors[point]:
                    if pos[neighbor] != (0,0):
                        cv2.line(mask_img, pos[point][::-1], pos[neighbor][::-1], colors[cl], 2)
                        cl += 1

    return mask_img