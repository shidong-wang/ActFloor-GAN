import numpy as np
import cv2 as cv
import os
import copy
from PIL import Image
from scipy.ndimage import label
import shutil

data_size = 256

class WallAbstract():
    def __init__(self, input_map):
        self.boundary_map = input_map[:,:,0]
        self.interior_wall_map = input_map[:,:,1]
        self.label_map = input_map[:,:,2]
        self.inside_map = input_map[:,:,3]

        self.exterior_boundary = []
        self.interior_walls = []

        self.house_min_h = data_size
        self.house_max_h = 0
        self.house_min_w = data_size
        self.house_max_w = 0

        for h in range(data_size):  
            for w in range(data_size):
                if self.boundary_map[h, w] > 0:
                    self.house_min_h = h if(self.house_min_h > h) else self.house_min_h
                    self.house_max_h = h if(self.house_max_h < h) else self.house_max_h
                    self.house_min_w = w if(self.house_min_w > w) else self.house_min_w
                    self.house_max_w = w if(self.house_max_w < w) else self.house_max_w

        self.house_min_h -= 10
        self.house_max_h += 10
        self.house_min_w -= 10
        self.house_max_w += 10

    def scan_line(self, wall_map, location, direction=0):
        lines = []
        w_min = 0
        w_max = 0
        h_min = 0
        h_max = 0
        
        if location > 255:
            print("error")
            return lines

        if direction == 0:
            for w in range(self.house_min_w, self.house_max_w):
                if location > 255 or w-1 < 0 or w+1 > 255:
                    continue
                if wall_map[location, w-1] == 0 and wall_map[location, w] == 127:
                    w_min = w
                if wall_map[location, w] == 127 and wall_map[location, w+1] == 0:
                    w_max = w
                    lines.append((w_min, w_max))
        else:
            for h in range(self.house_min_h, self.house_max_h):
                if location > 255 or h-1 < 0 or h+1 > 255:
                    continue
                if wall_map[h-1, location] == 0 and wall_map[h, location] == 127:
                    h_min = h
                if wall_map[h, location] == 127 and wall_map[h+1, location] == 0:
                    h_max = h
                    lines.append((h_min, h_max))

        return lines

    def contact_length(self, line1, line2):
        length1 = line1[1] - line1[0] + 1
        length2 = line2[1] - line2[0] + 1
        length = max(line1[1], line2[1]) - min(line1[0], line2[0]) + 1
        contact_length = length1 + length2 - length

        return contact_length

    def find_contact_line(self, wall_map, location, input_line, direction=0):
        candidate_lines = self.scan_line(wall_map, location, direction=direction)
        contact_length = 0
        max_contact_line = (-1, -2)
        for line in candidate_lines:
            new_contact_length = self.contact_length(line, input_line)
            if new_contact_length > contact_length:
                max_contact_line = line
                contact_length = new_contact_length

        return max_contact_line 

    def find_contact_line2(self, wall_map, location, input_line, direction=0):
        contact_lines = []
        candidate_lines = self.scan_line(wall_map, location, direction=direction)
        for line in candidate_lines:
            if self.contact_length(line, input_line) > 0:
                contact_lines.append(line)
        return contact_lines 

    def wall_abstract(self, wall_map):
        walls = []
        while True:
            wall_block = []
            for h in range(self.house_min_h, self.house_max_h):
                lines = self.scan_line(wall_map, h, direction=0)
                if len(lines) > 0:
                    wall_block.append((h, lines[0]))
                    for w in range(lines[0][0], lines[0][1]+1):
                        wall_map[h, w] = 0
                    break

            if len(wall_block) == 0:
                break    

            i = 0
            while i < len(wall_block):
                h, input_line = wall_block[i][0], wall_block[i][1]
                contact_line_up = self.find_contact_line(wall_map, h-1, input_line)
                if contact_line_up[0] > 0:
                    wall_block.append((h-1, contact_line_up))
                    for w in range(contact_line_up[0], contact_line_up[1]+1):
                        wall_map[h-1, w] = 0
                contact_line_down = self.find_contact_line(wall_map, h+1, input_line)
                if contact_line_down[0] > 0:
                    wall_block.append((h+1, contact_line_down))
                    for w in range(contact_line_down[0], contact_line_down[1]+1):
                        wall_map[h+1, w] = 0
                i += 1

            merge_min_h = data_size
            merge_max_h = 0            
            merge_min_w = data_size
            merge_max_w = 0
            
            for block in wall_block:
                merge_min_h = block[0] if(merge_min_h > block[0]) else merge_min_h
                merge_max_h = block[0] if(merge_max_h < block[0]) else merge_max_h
                merge_min_w = block[1][0] if(merge_min_w > block[1][0]) else merge_min_w
                merge_max_w = block[1][1] if(merge_max_w < block[1][1]) else merge_max_w 
            merge_height = merge_max_h - merge_min_h + 1
            merge_width = merge_max_w - merge_min_w + 1  

            if merge_height != 1 and merge_width != 1 and merge_height*merge_width > 3*6:
                if merge_height > merge_width:                        
                    merge_mid_w = (merge_min_w + merge_max_w) // 2
                    merge_min_w =  merge_mid_w - 1
                    merge_max_w =  merge_mid_w + 1
                else:                        
                    merge_mid_h = (merge_min_h + merge_max_h) // 2
                    merge_min_h =  merge_mid_h - 1
                    merge_max_h =  merge_mid_h + 1

                merge_height = merge_max_h - merge_min_h + 1
                merge_width = merge_max_w - merge_min_w + 1                        
                if merge_height > merge_width:
                    walls.append([merge_min_h, merge_max_h, merge_min_w, merge_max_w, 1]) 
                else:    
                    walls.append([merge_min_h, merge_max_h, merge_min_w, merge_max_w, 0])

        return walls      

    def interior_wall_padding(self, tolerance=6):
        new_interior_wall_map = self.interior_wall_map.copy()
        for h in range(self.house_min_h, self.house_max_h):
            lines = self.scan_line(self.interior_wall_map, h, direction=0)
            if len(lines) > 1:
                for i in range(len(lines)-1):
                    line_0 = lines[i]
                    line_1 = lines[i+1]
                    if line_1[0] - line_0[1] - 1 < tolerance:
                        for w in range(line_0[1]+1, line_1[0]):
                            new_interior_wall_map[h, w] = 127

        for w in range(self.house_min_w, self.house_max_w):
            lines = self.scan_line(self.interior_wall_map, w, direction=1)
            if len(lines) > 1:
                for i in range(len(lines)-1):
                    line_0 = lines[i]
                    line_1 = lines[i+1]
                    if line_1[0] - line_0[1] - 1 < tolerance:
                        for h in range(line_0[1]+1, line_1[0]):
                            new_interior_wall_map[h, w] = 127

        for h in range(self.house_min_h, self.house_max_h):
            lines = self.scan_line(new_interior_wall_map, h, direction=0)
            for line in lines:
                contact_line_next = self.find_contact_line(new_interior_wall_map, h+1, line, direction=0)
                length_line = line[1] - line[0] + 1
                length_contact_line_next = contact_line_next[1] - contact_line_next[0] + 1
                if (length_contact_line_next > length_line > 0.5*length_contact_line_next) or \
                    (length_line > length_contact_line_next > 0.5*length_line):
                    min_w = min(line[0], contact_line_next[0])
                    max_w = max(line[1], contact_line_next[1])
                    for w in range(min_w, max_w+1):
                        new_interior_wall_map[h, w] = 127
                        new_interior_wall_map[h+1, w] = 127

        for w in range(self.house_min_w, self.house_max_w):
            lines = self.scan_line(new_interior_wall_map, w, direction=1)
            for line in lines:
                contact_line_next = self.find_contact_line(new_interior_wall_map, w+1, line, direction=1)
                length_line = line[1] - line[0] + 1
                length_contact_line_next = contact_line_next[1] - contact_line_next[0] + 1
                if (length_contact_line_next > length_line > 0.5*length_contact_line_next) or \
                    (length_line > length_contact_line_next > 0.5*length_line):
                    min_h = min(line[0], contact_line_next[0])
                    max_h = max(line[1], contact_line_next[1])
                    for h in range(min_h, max_h+1):
                        new_interior_wall_map[h, w] = 127
                        new_interior_wall_map[h, w+1] = 127
                        
        self.interior_wall_map = new_interior_wall_map

    def interior_wall_decomposition(self):
        for w in range(self.house_min_w, self.house_max_w):
            lines = self.scan_line(self.interior_wall_map, w, direction=1)
            for line in lines:
                contact_line_nexts = self.find_contact_line2(self.interior_wall_map, w+1, line, direction=1)
                for contact_line_next in contact_line_nexts:
                    length_line = line[1] - line[0] + 1
                    length_contact_line_next = contact_line_next[1] - contact_line_next[0] + 1
                    if 0 < length_line < 0.5*length_contact_line_next:
                        for h in range(line[0], line[1]+1):
                            self.interior_wall_map[h, w] = 0
                    if 0 < length_contact_line_next < 0.5*length_line:
                        for h in range(line[0], line[1]+1):
                            self.interior_wall_map[h, w+1] = 0

    def merge(self, wall1, wall2):
        merge_min_h = min(wall1[0], wall2[0])
        merge_max_h = max(wall1[1], wall2[1])
        merge_min_w = min(wall1[2], wall2[2])
        merge_max_w = max(wall1[3], wall2[3])
        merge_height = merge_max_h - merge_min_h + 1
        merge_width = merge_max_w - merge_min_w + 1 
        if merge_height > merge_width:                        
            merge_mid_w = (merge_min_w + merge_max_w) // 2
            merge_min_w =  merge_mid_w - 1
            merge_max_w =  merge_mid_w + 1
            merge_wall = [merge_min_h, merge_max_h, merge_min_w, merge_max_w, 1]
        else:                        
            merge_mid_h = (merge_min_h + merge_max_h) // 2
            merge_min_h =  merge_mid_h - 1
            merge_max_h =  merge_mid_h + 1
            merge_wall = [merge_min_h, merge_max_h, merge_min_w, merge_max_w, 0]

        return merge_wall

    def interior_wall_merge(self, tolerance=5):
        i = 0
        while i < len(self.interior_walls):
            j = i + 1
            while j < len(self.interior_walls):
                flag = False           
                wall1 = self.interior_walls[i]
                wall2 = self.interior_walls[j] 
                height_wall1 = wall1[1] - wall1[0] + 1
                width_wall1 = wall1[3] - wall1[2] + 1
                height_wall2 = wall2[1] - wall2[0] + 1
                width_wall2 = wall2[3] - wall2[2] + 1  
                min_h = min(wall1[0], wall2[0])
                max_h = max(wall1[1], wall2[1])
                min_w = min(wall1[2], wall2[2])
                max_w = max(wall1[3], wall2[3])
                height = max_h - min_h + 1
                width = max_w - min_w + 1

                if wall1[4] == wall2[4] and height < height_wall1 + height_wall2 + tolerance and \
                    width < width_wall1 + width_wall2 + tolerance:
                    self.interior_walls[i] = self.merge(wall1, wall2)
                    self.interior_walls.pop(j)                    
                    flag = True                
                if flag is not True:
                    j = j + 1
            i = i + 1

    def interior_wall_adjustment(self):
        # wall-boundary alignment
        i = 0
        while i < len(self.interior_walls):
            j = 0
            while j < len(self.exterior_boundary):
                wall = self.interior_walls[i]
                boundary = self.exterior_boundary[j]
                boundary_pre = self.exterior_boundary[j-1]
                if wall[0] - 6 < boundary[0] < wall[1] + 6 and wall[2] - 6 < boundary[1] < wall[3] + 6: 
                    if wall[4] == 0:
                        if (boundary_pre[2] == 0 and boundary[2] == 3) or (boundary_pre[2] == 1 and boundary[2] == 0):
                            self.interior_walls[i][0] = boundary[0] - 3
                            self.interior_walls[i][1] = boundary[0] - 1
                        if (boundary_pre[2] == 2 and boundary[2] == 1) or (boundary_pre[2] == 3 and boundary[2] == 2):
                            self.interior_walls[i][1] = boundary[0] + 2
                            self.interior_walls[i][0] = boundary[0]
                    if wall[4] == 1:
                        if (boundary_pre[2] == 0 and boundary[2] == 3) or (boundary_pre[2] == 3 and boundary[2] == 2):
                            self.interior_walls[i][2] = boundary[1] - 3
                            self.interior_walls[i][3] = boundary[1] - 1
                        if (boundary_pre[2] == 1 and boundary[2] == 0) or (boundary_pre[2] == 2 and boundary[2] == 1):
                            self.interior_walls[i][3] = boundary[1] + 2
                            self.interior_walls[i][2] = boundary[1]
                j = j + 1
            i = i + 1

        # wall alignment
        i = 0
        while i < len(self.interior_walls):
            j = 0
            while j < len(self.interior_walls): 
                wall1 = self.interior_walls[i]
                wall2 = self.interior_walls[j] 
                height_wall1 = wall1[1] - wall1[0] + 1
                width_wall1 = wall1[3] - wall1[2] + 1
                height_wall2 = wall2[1] - wall2[0] + 1
                width_wall2 = wall2[3] - wall2[2] + 1  
                min_h = min(wall1[0], wall2[0])
                max_h = max(wall1[1], wall2[1])
                min_w = min(wall1[2], wall2[2])
                max_w = max(wall1[3], wall2[3])
                height = max_h - min_h + 1
                width = max_w - min_w + 1

                if wall1[4] == 0 and wall2[4] == 0:
                    if height <= height_wall1 + height_wall2 and width <= width_wall1 + width_wall2 + 9:
                        if width_wall1 < width_wall2:
                            self.interior_walls[i][0] = self.interior_walls[j][0]
                            self.interior_walls[i][1] = self.interior_walls[j][1]
                        else:
                            self.interior_walls[j][0] = self.interior_walls[i][0]
                            self.interior_walls[j][1] = self.interior_walls[i][1]

                if wall1[4] == 1 and wall2[4] == 1:
                    if height <= height_wall1 + height_wall2 + 9 and width <= width_wall1 + width_wall2:          
                        if height_wall1 < height_wall2:
                            self.interior_walls[i][2] = self.interior_walls[j][2]
                            self.interior_walls[i][3] = self.interior_walls[j][3]
                        else:
                            self.interior_walls[j][2] = self.interior_walls[i][2]
                            self.interior_walls[j][3] = self.interior_walls[i][3]
                j = j + 1
            i = i + 1

        # wall extension
        i = 0
        while i < len(self.interior_walls):
            j = 0
            while j < len(self.interior_walls): 
                wall1 = self.interior_walls[i]
                wall2 = self.interior_walls[j]   

                if wall2[4] == 0 and wall1[4] == 1:
                    if wall1[2] >= wall2[2] and wall1[3] <= wall2[3]:
                        base_line = (wall2[0] + wall2[1]) // 2
                        distance_up = abs(wall1[0] - base_line)
                        distance_down = abs(wall1[1] - base_line)
                        if min(distance_up, distance_down) < 9 + 3:
                            if distance_up > distance_down:
                                self.interior_walls[i][1] = wall2[0]-1
                            else:
                                self.interior_walls[i][0] = wall2[1]+1
                    
                if wall1[4] == 0 and wall2[4] == 1:
                    if wall1[0] >= wall2[0] and wall1[1] <= wall2[1]:
                        base_line = (wall2[2] + wall2[3]) // 2
                        distance_left = abs(wall1[2] - base_line)
                        distance_right = abs(wall1[3] - base_line)
                        if min(distance_left, distance_right) < 9 + 3:
                            if distance_left > distance_right:
                                self.interior_walls[i][3] = wall2[2]-1
                            else:
                                self.interior_walls[i][2] = wall2[3]+1
                j = j + 1
            i = i + 1

        # corner alignment
        i = 0
        while i < len(self.interior_walls):
            j = 0
            while j < len(self.interior_walls):  
                wall1 = self.interior_walls[i]
                wall2 = self.interior_walls[j]  

                if wall1[4] == 1 and wall2[4] == 0:                    
                    if wall1[2] == wall2[3]+1 or wall2[2] == wall1[3]+1:
                        if  wall2[0] - 6 < wall1[0] < wall2[1] + 6:
                            self.interior_walls[i][0] = wall2[0]
                        if  wall2[0] - 6 < wall1[1] < wall2[1] + 6:
                            self.interior_walls[i][1] = wall2[1]

                if wall1[4] == 0 and wall2[4] == 1:                    
                    if wall1[0] == wall2[1]+1 or wall2[0] == wall1[1]+1:
                        if  wall2[2] - 6 < wall1[2] < wall2[3] + 6:
                            self.interior_walls[i][2] = wall2[2]
                        if  wall2[2] - 6 < wall1[3] < wall2[3] + 6:
                            self.interior_walls[i][3] = wall2[3]
                j = j + 1
            i = i + 1 

        i = 0
        while i < len(self.interior_walls):
            j = 0 
            while j < len(self.interior_walls): 
                wall1 = self.interior_walls[i]
                wall2 = self.interior_walls[j]

                if wall1[4] == 1 and wall2[4] == 0:                    
                    if wall2[2] - 9 <= wall1[2] <= wall2[2]:
                        if wall2[0] <= wall1[0] <= wall2[1] + 9:
                            self.interior_walls[i][0] = wall2[1] + 1
                            self.interior_walls[j][2] = wall1[2]
                        if wall2[0] - 9 <= wall1[1] <= wall2[1]:
                            self.interior_walls[i][1] = wall2[0] - 1
                            self.interior_walls[j][2] = wall1[2]
                    if wall2[3] <= wall1[3] <= wall2[3] + 9:
                        if wall2[0] <= wall1[0] <= wall2[1] + 9:
                            self.interior_walls[i][0] = wall2[1] + 1
                            self.interior_walls[j][3] = wall1[3]
                        if wall2[0] - 9 <= wall1[1] <= wall2[1]:
                            self.interior_walls[i][1] = wall2[0] - 1
                            self.interior_walls[j][3] = wall1[3]

                if wall1[4] == 0 and wall2[4] == 1:                    
                    if wall2[0] - 9 <= wall1[0] <= wall2[0]:
                        if wall2[2] <= wall1[2] <= wall2[3] + 9:
                            self.interior_walls[i][2] = wall2[2]
                            self.interior_walls[j][0] = wall1[1] + 1
                        if wall2[2] - 9 <= wall1[3] <= wall2[3]:
                            self.interior_walls[i][3] = wall2[3]
                            self.interior_walls[j][0] = wall1[1] + 1
                    if wall2[1] <= wall1[1] <= wall2[1] + 9:
                        if wall2[2] <= wall1[2] <= wall2[3] + 9:
                            self.interior_walls[i][2] = wall2[2]
                            self.interior_walls[j][1] = wall1[0] - 1
                        if wall2[2] - 9 <= wall1[3] <= wall2[3]:
                            self.interior_walls[i][3] = wall2[3]
                            self.interior_walls[j][1] = wall1[0] - 1
                j = j + 1
            i = i + 1 

    def is_near_boundary(self, wall, dir, boundary, tolerance=15):

        if wall[4] == 0:
            if dir == 2:       
                count_sum = 0
                if self.interior_wall_map[wall[0] - 1][wall[2]] == 0 and self.inside_map[wall[0] - 1][wall[2]] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[0] + 1][wall[2] - 1] == 0 and self.inside_map[wall[0] + 1][wall[2] - 1] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[1] + 1][wall[2]] == 0 and self.inside_map[wall[1] + 1][wall[2]] == 255:
                    count_sum += 1
                if count_sum == 3 and 0 < wall[2] - boundary < tolerance:
                    return True
                else:
                    return False
            if dir == 3:
                count_sum = 0
                if self.interior_wall_map[wall[0] - 1][wall[3]] == 0 and self.inside_map[wall[0] - 1][wall[3]] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[0] + 1][wall[3] + 1] == 0 and self.inside_map[wall[0] + 1][wall[3] + 1] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[1] + 1][wall[3]] == 0 and self.inside_map[wall[1] + 1][wall[3]] == 255:
                    count_sum += 1
                if count_sum == 3 and 0 < boundary - wall[3] < tolerance:
                    return True
                else:
                    return False
        if wall[4] == 1:
            if dir == 0:
                count_sum = 0
                if self.interior_wall_map[wall[0]][wall[2] - 1] == 0 and self.inside_map[wall[0]][wall[2] - 1] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[0] - 1][wall[2] + 1] == 0 and self.inside_map[wall[0] - 1][wall[2] + 1] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[0]][wall[3] + 1] == 0 and self.inside_map[wall[0]][wall[3] + 1] == 255:
                    count_sum += 1
                if count_sum == 3 and 0 < wall[0] - boundary < tolerance:
                    return True
                else:
                    return False
            if dir == 1:
                count_sum = 0
                if self.interior_wall_map[wall[1]][wall[2] - 1] == 0 and self.inside_map[wall[1]][wall[2] - 1] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[1] + 1][wall[2] + 1] == 0 and self.inside_map[wall[1] + 1][wall[2] + 1] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[1]][wall[3] + 1] == 0 and self.inside_map[wall[1]][wall[3] + 1] == 255:
                    count_sum += 1
                if count_sum == 3 and 0 < boundary - wall[1] < tolerance:
                    return True
                else:
                    return False

    def is_break_wall(self, wall, dir):
        if wall[4] == 0:
            if dir == 2:
                flag = True
                for delta_h in range(-1, 2):
                    for delta_w in range(-1, 2):
                        if self.interior_wall_map[wall[0] - 2 + delta_h][wall[2] + 1 + delta_w] == 127 or self.inside_map[wall[0] - 2 + delta_h][wall[2] + 1 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[0] + 1 + delta_h][wall[2] - 2 + delta_w] == 127 or self.inside_map[wall[0] + 1 + delta_h][wall[2] - 2 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[1] + 2 + delta_h][wall[2] + 1 + delta_w] == 127 or self.inside_map[wall[1] + 2 + delta_h][wall[2] + 1 + delta_w] == 0:
                            flag = False
                return flag
            if dir == 3:
                flag = True
                for delta_h in range(-1, 2):
                    for delta_w in range(-1, 2):
                        if self.interior_wall_map[wall[0] - 2 + delta_h][wall[3] + 1 + delta_w] == 127 or self.inside_map[wall[0] - 2 + delta_h][wall[3] + 1 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[0] + 1 + delta_h][wall[3] + 2 + delta_w] == 127 or self.inside_map[wall[0] + 1 + delta_h][wall[3] + 2 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[1] + 2 + delta_h][wall[3] + 1 + delta_w] == 127 or self.inside_map[wall[1] + 2 + delta_h][wall[3] + 1 + delta_w] == 0:
                            flag = False
                return flag
        if wall[4] == 1:
            if dir == 0:
                flag = True
                for delta_h in range(-1, 2):
                    for delta_w in range(-1, 2):
                        if self.interior_wall_map[wall[0] + 1 + delta_h][wall[2] - 2 + delta_w] == 127 or self.inside_map[wall[0] + 1 + delta_h][wall[2] - 2 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[0] - 2 + delta_h][wall[2] + 1 + delta_w] == 127 or self.inside_map[wall[0] - 2 + delta_h][wall[2] + 1 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[0] + 1 + delta_h][wall[3] + 2 + delta_w] == 127 or self.inside_map[wall[0] + 1 + delta_h][wall[3] + 2 + delta_w] == 0:
                            flag = False
                return flag
            if dir == 1:
                flag = True
                for delta_h in range(-1, 2):
                    for delta_w in range(-1, 2):
                        if self.interior_wall_map[wall[1] - 1 + delta_h][wall[2] - 2 + delta_w] == 127 or self.inside_map[wall[1] - 1 + delta_h][wall[2] - 2 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[1] + 2 + delta_h][wall[2] + 1 + delta_w] == 127 or self.inside_map[wall[1] + 2 + delta_h][wall[2] + 1 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[1] - 1 + delta_h][wall[3] + 2 + delta_w] == 127 or self.inside_map[wall[1] - 1 + delta_h][wall[3] + 2 + delta_w] == 0:
                            flag = False
                return flag

    def interior_wall_final(self):
        interior_wall_map = np.zeros((data_size, data_size), dtype=np.uint8)
        for wall in self.interior_walls:
            for h in range(wall[0], wall[1]+1):
                for w in range(wall[2], wall[3]+1):
                    interior_wall_map[h, w] = 127
        self.interior_wall_map = interior_wall_map

        # wall-boundary extension
        i = 0
        while i < len(self.interior_walls):
            j = 0
            while j < len(self.exterior_boundary):
                wall = self.interior_walls[i]
                boundary = self.exterior_boundary[j]
                boundary_pre = self.exterior_boundary[j-1]
                min_h = min(boundary[0], boundary_pre[0])
                max_h = max(boundary[0], boundary_pre[0])
                min_w = min(boundary[1], boundary_pre[1])
                max_w = max(boundary[1], boundary_pre[1])
                if wall[4] == 0 and min_w == max_w:
                    if wall[0] >= min_h and wall[1] <= max_h:
                        if self.is_near_boundary(wall, 2, min_w):
                            self.interior_walls[i][2] = min_w
                        if self.is_near_boundary(wall, 3, min_w):    
                            self.interior_walls[i][3] = min_w
                if wall[4] == 1 and min_h == max_h:
                    if wall[2] >= min_w and wall[3] <= max_w:
                        if self.is_near_boundary(wall, 0, min_h):
                            self.interior_walls[i][0] = min_h
                        if self.is_near_boundary(wall, 1, min_h):
                            self.interior_walls[i][1] = min_h
                j = j + 1
            i = i + 1

        # wall shrink
        i = 0
        while i < len(self.interior_walls):
            j = 0
            while j < len(self.interior_walls): 
                wall1 = self.interior_walls[i]
                wall2 = self.interior_walls[j]   

                if wall2[4] == 0 and wall1[4] == 1:
                    if wall1[2] > wall2[2] and wall1[3] < wall2[3]:
                        base_line = (wall2[0] + wall2[1]) // 2
                        distance_up = abs(wall1[0] - base_line)
                        distance_down = abs(wall1[1] - base_line)
                        if min(distance_up, distance_down) < 9 + 3:
                            if distance_up > distance_down:
                                self.interior_walls[i][1] = wall2[0]-1
                            else:
                                self.interior_walls[i][0] = wall2[1]+1
                    
                if wall1[4] == 0 and wall2[4] == 1:
                    if wall1[0] > wall2[0] and wall1[1] < wall2[1]:
                        base_line = (wall2[2] + wall2[3]) // 2
                        distance_left = abs(wall1[2] - base_line)
                        distance_right = abs(wall1[3] - base_line)
                        if min(distance_left, distance_right) < 9 + 3:
                            if distance_left > distance_right:
                                self.interior_walls[i][3] = wall2[2]-1
                            else:
                                self.interior_walls[i][2] = wall2[3]+1
                j = j + 1
            i = i + 1

        # delete short wall
        i = 0
        while i < len(self.interior_walls):
            wall = self.interior_walls[i]
            flag = False
            if wall[4] == 0:
                if self.is_break_wall(wall, 2) and self.is_break_wall(wall, 3):
                    self.interior_walls.pop(i)
                    flag = True
                else:
                     if self.is_break_wall(wall, 2) or self.is_break_wall(wall, 3):
                        if wall[3] - wall[2] + 1 < 20:
                            self.interior_walls.pop(i)
                            flag = True
            else:
                if self.is_break_wall(wall, 0) and self.is_break_wall(wall, 1):
                    self.interior_walls.pop(i)
                    flag = True
                else:
                    if self.is_break_wall(wall, 0) or self.is_break_wall(wall, 1):
                        if wall[1] - wall[0] + 1 < 20:
                            self.interior_walls.pop(i)
                            flag = True
            if flag is not True:
                i = i + 1

        # wall shrink
        i = 0
        while i < len(self.interior_walls):
            j = 0
            while j < len(self.interior_walls): 
                wall1 = self.interior_walls[i]
                wall2 = self.interior_walls[j]   

                if wall2[4] == 0 and wall1[4] == 1:
                    if wall1[2] > wall2[2] and wall1[3] < wall2[3]:
                        base_line = (wall2[0] + wall2[1]) // 2
                        distance_up = abs(wall1[0] - base_line)
                        distance_down = abs(wall1[1] - base_line)
                        if min(distance_up, distance_down) < 9 + 3:
                            if distance_up > distance_down:
                                self.interior_walls[i][1] = wall2[0]-1
                            else:
                                self.interior_walls[i][0] = wall2[1]+1
                    
                if wall1[4] == 0 and wall2[4] == 1:
                    if wall1[0] > wall2[0] and wall1[1] < wall2[1]:
                        base_line = (wall2[2] + wall2[3]) // 2
                        distance_left = abs(wall1[2] - base_line)
                        distance_right = abs(wall1[3] - base_line)
                        if min(distance_left, distance_right) < 9 + 3:
                            if distance_left > distance_right:
                                self.interior_walls[i][3] = wall2[2]-1
                            else:
                                self.interior_walls[i][2] = wall2[3]+1
                j = j + 1
            i = i + 1

    def interior_wall_abstract(self):
        self.interior_wall_padding()
        self.interior_wall_decomposition()
        self.interior_walls = self.wall_abstract(self.interior_wall_map)
        self.interior_wall_merge(tolerance=5)
        self.interior_wall_adjustment()
        self.interior_wall_merge(tolerance=2)
        self.interior_wall_final()

    def exterior_boundary_abstract(self):
        # search direction:0(right)/1(down)/2(left)/3(up)
        flag = False
        for h in range(self.house_min_h, self.house_max_h):
            for w in range(self.house_min_w, self.house_max_w):
                if self.inside_map[h, w] == 255:
                    self.exterior_boundary.append((h, w, 0))
                    flag = True
                    break
            if flag:
                break

        while(flag):
            if self.exterior_boundary[-1][2] == 0:
                for w in range(self.exterior_boundary[-1][1]+1, self.house_max_w):
                    corner_sum = 0
                    if self.inside_map[self.exterior_boundary[-1][0], w] == 255:
                        corner_sum += 1
                    if self.inside_map[self.exterior_boundary[-1][0]-1, w] == 255:
                        corner_sum += 1
                    if self.inside_map[self.exterior_boundary[-1][0], w-1] == 255:
                        corner_sum += 1
                    if self.inside_map[self.exterior_boundary[-1][0]-1, w-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (self.exterior_boundary[-1][0], w, 1)
                        break
                    if corner_sum == 3:
                        new_point = (self.exterior_boundary[-1][0], w, 3)
                        break

            if self.exterior_boundary[-1][2] == 1:      
                for h in range(self.exterior_boundary[-1][0]+1, self.house_max_h): 
                    corner_sum = 0                
                    if self.inside_map[h, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside_map[h-1, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside_map[h, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if self.inside_map[h-1, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (h, self.exterior_boundary[-1][1], 2)
                        break
                    if corner_sum == 3:
                        new_point = (h, self.exterior_boundary[-1][1], 0)
                        break

            if self.exterior_boundary[-1][2] == 2:   
                for w in range(self.exterior_boundary[-1][1]-1, self.house_min_w, -1):
                    corner_sum = 0                     
                    if self.inside_map[self.exterior_boundary[-1][0], w] == 255:
                        corner_sum += 1
                    if self.inside_map[self.exterior_boundary[-1][0]-1, w] == 255:
                        corner_sum += 1
                    if self.inside_map[self.exterior_boundary[-1][0], w-1] == 255:
                        corner_sum += 1
                    if self.inside_map[self.exterior_boundary[-1][0]-1, w-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (self.exterior_boundary[-1][0], w, 3)
                        break
                    if corner_sum == 3:
                        new_point = (self.exterior_boundary[-1][0], w, 1)
                        break

            if self.exterior_boundary[-1][2] == 3:       
                for h in range(self.exterior_boundary[-1][0]-1, self.house_min_h, -1):
                    corner_sum = 0                
                    if self.inside_map[h, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside_map[h-1, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside_map[h, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if self.inside_map[h-1, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (h, self.exterior_boundary[-1][1], 0)
                        break
                    if corner_sum == 3:
                        new_point = (h, self.exterior_boundary[-1][1], 2)
                        break

            if new_point != self.exterior_boundary[0]:
                self.exterior_boundary.append(new_point)
            else:
                flag = False

def getWall(image, a1, a2):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, a1, a2)
    canvas_size = dst.shape[0] 
    mask = np.zeros([canvas_size, canvas_size], np.uint8)
    mask[dst == 0] = 127
    return mask

def delOutwall(output_map):
    threshold = 8
    boun = output_map[:,:,0]
    wall = output_map[:,:,1]

    inside_mask = output_map[:,:,3]
    rows, cols = np.where(inside_mask == 255)

    minh = np.clip(np.min(rows) - 10, 0, 255)
    maxh = np.clip(np.max(rows) + 10, 0, 255)
    minw = np.clip(np.min(cols) - 10, 0, 255)
    maxw = np.clip(np.max(cols) + 10, 0, 255)

    for h in range(minh, maxh):
        for w in range(minw, maxw):
            if boun[h][w] == 127:
                for i in range (threshold * 2):
                    wall[h-threshold+i][w] = 0
                    wall[h][w-threshold+i] = 0
            elif boun[h][w] == 255:
                for i in range (threshold * 2):
                    wall[h-threshold+i][w] = 0
                    wall[h][w-threshold+i] = 0

    return wall

def getPixelCategory(c1, c2, c3):
    #   0     1    2    3    4    5    6    7    8     9    10   11   12   13   14  15  16 17
    r = [255, 156, 255, 255, 0,   65,  0,   65,  65,   128, 0,   0,   0,   255, 0,  127, 0,  0]
    g = [0,   102, 255, 97,  255, 105, 255, 105, 105,  42,  255, 255, 255, 255, 0,  127, 0,  0]
    b = [0,   31,  0,   0,   0,   225, 255, 225, 225,  42,  0,   0,   0,   255, 0,  127, 0,  0]
    rc =[0,   1,   2,   3,   4,   7,   6,   7,   7,    9,   4,   4,   4,   13,  14, 15,  14, 14]

    min_i = 13
    minimum  = 255 * 3

    for i in range(18):
        e1 = abs(c1 - r[i])
        e2 = abs(c2 - g[i])
        e3 = abs(c3 - b[i])
        esum = e1 + e2 + e3
        if esum < minimum:
            min_i = i
            minimum  = esum

    if minimum < 50:
        c1 = r[min_i]
        c2 = g[min_i] 
        c3 = b[min_i]

    else:
        c1 = 255
        c2 = 255
        c3 = 255
        min_i = 13

    return int(c1), int(c2),int(c3), rc[min_i]

def compute_centroid(mask, i):
    polygon = np.zeros(mask.shape)
    polygon[mask == i] = 1
    coordinates = np.argwhere(polygon == 1)
    centroid_h = int(np.mean(coordinates[:, 0]))
    centroid_w = int(np.mean(coordinates[:, 1]))
    return centroid_h, centroid_w

def getLabel(inside_mask, gen_fp):

    datasize = 256
    output_map = np.ones((datasize, datasize), dtype=np.uint8) * 13

    inside_mask = copy.deepcopy(inside_mask)

    rows, cols = np.where(inside_mask == 255)
    minh = np.clip(np.min(rows) - 10, 0, 255)
    maxh = np.clip(np.max(rows) + 10, 0, 255)
    minw = np.clip(np.min(cols) - 10, 0, 255)
    maxw = np.clip(np.max(cols) + 10, 0, 255)

    for h in range(minh, maxh):
        for w in range(minw, maxw):
            cr, cg, cb, output_map[h][w] = getPixelCategory(gen_fp[h,w,2], gen_fp[h,w,1], gen_fp[h,w,0])
    
    rcategory_list = [0, 1, 2, 3, 4, 6, 7, 9]

    room_index = 0
    index_mask = np.zeros(output_map.shape, dtype=np.uint8)
    label_mask = np.zeros(output_map.shape, dtype=np.uint8)

    for rc in rcategory_list:
        temp_mask = np.zeros(output_map.shape, dtype=np.uint8)
        temp_mask[output_map == rc] = 1
            
        if np.sum(temp_mask) < 200:
            continue

        labeled_array, num_rooms = label(temp_mask)

        for ri in range(num_rooms):
            temp_map = np.zeros(output_map.shape, dtype=np.uint8)
            temp_map[labeled_array == (ri+1)] = 1
            if np.sum(temp_map) < 200:
                continue

            room_index+=1
            index_mask[temp_map > 0] = room_index

            h, w = compute_centroid(index_mask, room_index)

            label_mask[h,w] = rc + 100

    return label_mask

def main(dataset_dir, genfps_dir, output_dir):

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)

    genfps_path = os.listdir(genfps_dir)

    test_number = 0
    for floorplan in genfps_path:
        
        test_number = test_number + 1
        if test_number % 1000 == 0:
            print(test_number)
        if os.path.exists(f'{output_dir}/{floorplan}'):
            continue
        
        floorplan_ini = Image.open(f'{dataset_dir}/{floorplan}')
        input_map = np.asarray(floorplan_ini, dtype=np.uint8)

        input_map_temp = np.zeros(input_map.shape, dtype=np.uint8)
        input_map_temp[:,:,0] = input_map[:,:,0]
        input_map_temp[:,:,3] = input_map[:,:,3]

        genfp_image = cv.imread(f'{genfps_dir}/{floorplan}')
        input_map_temp[:,:,1] = getWall(genfp_image, 25, 10)
        input_map_temp[:,:,1] = delOutwall(input_map_temp)

        output_map = np.zeros(input_map.shape, dtype=np.uint8)
        output_map[:,:,0] = input_map[:,:,0]
        output_map[:,:,3] = input_map[:,:,3]
        
        abstracter = WallAbstract(input_map_temp) 
        abstracter.exterior_boundary_abstract()
        abstracter.interior_wall_abstract() 

        for wall in abstracter.interior_walls:
            for h in range(wall[0], wall[1]+1):
                for w in range(wall[2], wall[3]+1):
                    output_map[h, w, 1] = 127
        
        label_mask = getLabel(output_map[:,:,3], genfp_image)
        output_map[:,:,2] = label_mask

        output = Image.fromarray(np.uint8(output_map))
        output.save(f'{output_dir}/{floorplan}')

    print(f'Total test number: {test_number}')

if __name__ == '__main__':

    dataset_rootDir = '../Dataset'
    dataset_dir = f'{dataset_rootDir}/dataset_4c'
    genfps_dir = 'genRasterLayout'
    output_dir = 'normalization'

    main(dataset_dir, genfps_dir, output_dir)
    
    