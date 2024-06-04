import os
import random
import numpy as np
from PIL import Image
import shutil
import math
import copy

from birrtUtils import birrt
# import matplotlib.pyplot as plt

def get_roomInfo(index_mask, category_mask, i):

    sum_h = 0
    sum_w = 0
    count = 0

    sum_door_h = 0
    sum_door_w = 0
    count_door = 0

    start_h = 0
    start_w = 0
    end_h = 0
    end_w = 0
    state = 0

    front_door_h = 0
    front_door_w = 0
    count_front_door = 0

    shape_array = index_mask.shape
    for h in range(shape_array[0]):
        for w in range(shape_array[1]):
            if index_mask[h, w] == i:
                if state == 0:
                    state = 1
                    start_h = h
                    start_w = w

                sum_h += h
                sum_w += w
                count += 1

                end_h = h
                end_w = w
            
            if category_mask[h, w] == 15:
                front_door_h += h
                front_door_w += w
                count_front_door += 1

            if category_mask[h, w] == 17:
                if index_mask[h, w-1] == i and category_mask[h, w+8] == 0:
                    sum_door_h += h
                    sum_door_w += w
                    count_door+=1
                if index_mask[h, w+1] == i and category_mask[h, w-8] == 0:
                    sum_door_h += h
                    sum_door_w += w
                    count_door+=1
                if index_mask[h-1, w] == i and category_mask[h+8, w] == 0:
                    sum_door_h += h
                    sum_door_w += w
                    count_door+=1
                if index_mask[h+1, w] == i and category_mask[h-8, w] == 0:
                    sum_door_h += h
                    sum_door_w += w
                    count_door+=1
    
    if count!=0:
        cen_h = sum_h // count
        cen_w = sum_w // count
    else:
        cen_h = 0
        cen_w = 0

    if count_door != 0:
        cen_door_h = sum_door_h // count_door
        cen_door_w = sum_door_w // count_door
    else:
        cen_door_h = 0
        cen_door_w = 0

    if category_mask[cen_h, cen_w] == 0:
        cen_door_h = front_door_h // count_front_door
        cen_door_w = front_door_w // count_front_door

    return (count, cen_h, cen_w, cen_door_h, cen_door_w, start_h, start_w, end_h, end_w)

def bedArrange(category, roomsize, cen_h, cen_w, cen_door_h, cen_door_w, start_h, start_w, end_h, end_w):

    out_h = 0
    out_w = 0
    length = 0
    width = 0
    direction = 0

    if category == 1:
        ratio_wl = 1.8 / 2
    if category == 7:
        ratio_wl = 1.5 / 2

    bedSize = roomsize / 3
    width = int(math.sqrt((ratio_wl) * bedSize) / 2)
    length = int((1/ratio_wl) * width)

    if cen_door_h <= cen_h and cen_door_w >= cen_w:
        k = random.randint(0, 1)
        if k == 0:  
            out_h = cen_h
            out_w = start_w
            direction = 4
        else:
            out_h = end_h
            out_w = cen_w
            direction = 1
    if cen_door_h <= cen_h and cen_door_w <= cen_w:
        k = random.randint(0, 1)
        if k == 0:   
            out_h = cen_h
            out_w = end_w
            direction = 2
        else:
            out_h = end_h
            out_w = cen_w
            direction = 1
    if cen_door_h >= cen_h and cen_door_w <= cen_w:
        k = random.randint(0, 1)
        if k == 0:  
            out_h = cen_h
            out_w = end_w
            direction = 2
        else:
            out_h = start_h
            out_w = cen_w
            direction = 3
    if cen_door_h >= cen_h and cen_door_w >= cen_w:
        k = random.randint(0, 1)
        if k == 0:  
            out_h = cen_h
            out_w = start_w
            direction = 4
        else:
            out_h = start_h
            out_w = cen_w
            direction = 3

    return out_h, out_w, length, width, direction

def deskArrange(roomsize, cen_h, cen_w, cen_door_h, cen_door_w, start_h, start_w, end_h, end_w):

    out_h = 0
    out_w = 0
    length = 0
    width = 0
    direction = 0

    ratio_wl = 2 / 1
    bedSize = roomsize / 8
    width = int(math.sqrt((ratio_wl) * bedSize) / 2)
    length = int((1/ratio_wl) * width)

    if cen_door_h <= cen_h and cen_door_w >= cen_w:
        k = random.randint(0, 1)
        if k == 0:  
            out_h = cen_h
            out_w = start_w
            direction = 4
        else:
            out_h = end_h
            out_w = cen_w
            direction = 1
    if cen_door_h <= cen_h and cen_door_w <= cen_w:
        k = random.randint(0, 1)
        if k == 0:   
            out_h = cen_h
            out_w = end_w
            direction = 2
        else:
            out_h = end_h
            out_w = cen_w
            direction = 1
    if cen_door_h >= cen_h and cen_door_w <= cen_w:
        k = random.randint(0, 1)
        if k == 0:  
            out_h = cen_h
            out_w = end_w
            direction = 2
        else:
            out_h = start_h
            out_w = cen_w
            direction = 3
    if cen_door_h >= cen_h and cen_door_w >= cen_w:
        k = random.randint(0, 1)
        if k == 0:  
            out_h = cen_h
            out_w = start_w
            direction = 4
        else:
            out_h = start_h
            out_w = cen_w
            direction = 3

    return out_h, out_w, length, width, direction

def toiletArrange(cen_h, cen_w, cen_door_h, cen_door_w, start_h, start_w, end_h, end_w):

    out_h = 0
    out_w = 0
    length = 5
    width = 5
    direction = 0

    if cen_door_h <= cen_h and cen_door_w >= cen_w:
        k = random.randint(0, 1)
        if k == 0:  
            out_h = cen_h
            out_w = start_w
            direction = 4
        else:
            out_h = end_h
            out_w = cen_w
            direction = 1

    if cen_door_h <= cen_h and cen_door_w <= cen_w:
        k = random.randint(0, 1)
        if k == 0:   
            out_h = cen_h
            out_w = end_w
            direction = 2
        else:
            out_h = end_h
            out_w = cen_w
            direction = 1
    if cen_door_h >= cen_h and cen_door_w <= cen_w:
        k = random.randint(0, 1)
        if k == 0:  
            out_h = cen_h
            out_w = end_w
            direction = 2
        else:
            out_h = start_h
            out_w = cen_w
            direction = 3
    if cen_door_h >= cen_h and cen_door_w >= cen_w:
        k = random.randint(0, 1)
        if k == 0:  
            out_h = cen_h
            out_w = start_w
            direction = 4
        else:
            out_h = start_h
            out_w = cen_w
            direction = 3

    return out_h, out_w, length, width, direction

def stoveArrange(cen_h, cen_w, cen_door_h, cen_door_w, start_h, start_w, end_h, end_w):

    out_h = 0
    out_w = 0
    width = 5
    direction = 0

    if cen_door_h <= cen_h and cen_door_w >= cen_w:
        k = random.randint(0, 1)
        if k == 0:  
            out_h = cen_h
            out_w = start_w
            direction = 4
        else:
            out_h = end_h
            out_w = cen_w
            direction = 1
    if cen_door_h <= cen_h and cen_door_w <= cen_w:
        k = random.randint(0, 1)
        if k == 0:   
            out_h = cen_h
            out_w = end_w
            direction = 2
        else:
            out_h = end_h
            out_w = cen_w
            direction = 1
    if cen_door_h >= cen_h and cen_door_w <= cen_w:
        k = random.randint(0, 1)
        if k == 0:  
            out_h = cen_h
            out_w = end_w
            direction = 2
        else:
            out_h = start_h
            out_w = cen_w
            direction = 3
    if cen_door_h >= cen_h and cen_door_w >= cen_w:
        k = random.randint(0, 1)
        if k == 0:  
            out_h = cen_h
            out_w = start_w
            direction = 4
        else:
            out_h = start_h
            out_w = cen_w
            direction = 3

    return out_h, out_w, width, direction

def wmArrange(cen_h, cen_w, start_h, start_w, end_h, end_w):
    
    out_h = 0
    out_w = 0
    width = 5
    direction = 0

    room_h = end_h - start_h
    room_w = end_w - start_w

    if room_h < room_w:
        k = random.randint(0, 1)
        if k == 0:  
            out_h = cen_h
            out_w = start_w
            direction = 4
        else:
            out_h = cen_h
            out_w = end_w
            direction = 2
    else:
        k = random.randint(0, 1)
        if k == 0:  
            out_h = start_h
            out_w = cen_w
            direction = 3
        else:
            out_h = end_h
            out_w = cen_w
            direction = 1

    return out_h, out_w, width, direction

def furnitureArrange(category_mask, index_mask):

    category_mask_wFurniture = category_mask.copy()
    category_mask = category_mask.copy()
    category_mask[category_mask == 5] = 7
    category_mask[category_mask == 8] = 7

    data_size = 256

    num_rooms = index_mask[np.unravel_index(index_mask.argmax(), index_mask.shape)]

    rooms_info = []
    
    for room_i in range(1, num_rooms+1):
        
        roomsize, cen_h, cen_w, cent_door_h, cent_door_w ,start_h, start_w, end_h, end_w = get_roomInfo(index_mask, category_mask, room_i)
        
        if roomsize == 0:
            continue
        
        if cent_door_h == 0 and cent_door_w == 0:
            continue

        category = category_mask[cen_h, cen_w]

        if category == 0:
            mask_size = 10
            min_h = max(cen_h - mask_size, 0)
            max_h = min(cen_h + mask_size, data_size - 1)
            min_w = max(cen_w - mask_size, 0)
            max_w = min(cen_w + mask_size, data_size - 1)
            #category_mask_wFurniture[min_h:max_h+1, min_w:max_w+1] = category + 100
            furniture_cen_h = cen_h
            furniture_cen_w = cen_w

        elif category == 1 or category == 7:
            out_h, out_w, length, width, direction = bedArrange(category, roomsize, cen_h, cen_w, cent_door_h, cent_door_w ,start_h, start_w, end_h, end_w)
            if direction == 1:
                min_h = max(out_h - length * 2, 0) 
                max_h = min(out_h, data_size - 1)
                min_w = max(out_w - width, 0)
                max_w = min(out_w + width, data_size - 1)
                furniture_cen_h = out_h - length
                furniture_cen_w = out_w
            elif direction == 2:
                min_h = max(out_h - width, 0) 
                max_h = min(out_h + width, data_size - 1)
                min_w = max(out_w - length * 2, 0)
                max_w = min(out_w, data_size - 1)
                furniture_cen_h = out_h
                furniture_cen_w = out_w - length
            elif direction == 3:
                min_h = max(out_h , 0) 
                max_h = min(out_h + length * 2, data_size - 1)
                min_w = max(out_w - width, 0)
                max_w = min(out_w + width, data_size - 1)
                furniture_cen_h = out_h + length
                furniture_cen_w = out_w
            elif direction == 4:
                min_h = max(out_h - width, 0) 
                max_h = min(out_h + width, data_size - 1)
                min_w = max(out_w, 0)
                max_w = min(out_w + length * 2, data_size - 1)
                furniture_cen_h = out_h
                furniture_cen_w = out_w + length
            category_mask_wFurniture[min_h:max_h+1, min_w:max_w+1] = category + 100

        elif category == 6:
            
            out_h, out_w, length, width, direction = deskArrange(roomsize, cen_h, cen_w, cent_door_h, cent_door_w ,start_h, start_w, end_h, end_w)
            
            if direction == 1:
                min_h = max(out_h - length * 2, 0) 
                max_h = min(out_h, data_size - 1)
                min_w = max(out_w - width, 0)
                max_w = min(out_w + width, data_size - 1)
                furniture_cen_h = out_h - length
                furniture_cen_w = out_w
            elif direction == 2:
                min_h = max(out_h - width, 0) 
                max_h = min(out_h + width, data_size - 1)
                min_w = max(out_w - length * 2, 0)
                max_w = min(out_w, data_size - 1)
                furniture_cen_h = out_h
                furniture_cen_w = out_w - length
            elif direction == 3:
                min_h = max(out_h , 0) 
                max_h = min(out_h + length * 2, data_size - 1)
                min_w = max(out_w - width, 0)
                max_w = min(out_w + width, data_size - 1)
                furniture_cen_h = out_h + length
                furniture_cen_w = out_w
            elif direction == 4:
                min_h = max(out_h - width, 0) 
                max_h = min(out_h + width, data_size - 1)
                min_w = max(out_w, 0)
                max_w = min(out_w + length * 2, data_size - 1)
                furniture_cen_h = out_h
                furniture_cen_w = out_w + length
            category_mask_wFurniture[min_h:max_h+1, min_w:max_w+1] = category + 100
        
        elif category == 3:
            out_h, out_w, length, width, direction = toiletArrange(cen_h, cen_w, cent_door_h, cent_door_w ,start_h, start_w, end_h, end_w)
            if direction == 1:
                min_h = max(out_h - length * 2, 0) 
                max_h = min(out_h, data_size - 1)
                min_w = max(out_w - width, 0)
                max_w = min(out_w + width, data_size - 1)
                furniture_cen_h = out_h - length
                furniture_cen_w = out_w
            elif direction == 2:
                min_h = max(out_h - width, 0) 
                max_h = min(out_h + width, data_size - 1)
                min_w = max(out_w - length * 2, 0)
                max_w = min(out_w, data_size - 1)
                furniture_cen_h = out_h
                furniture_cen_w = out_w - length
            elif direction == 3:
                min_h = max(out_h , 0) 
                max_h = min(out_h + length * 2, data_size - 1)
                min_w = max(out_w - width, 0)
                max_w = min(out_w + width, data_size - 1)
                furniture_cen_h = out_h + length
                furniture_cen_w = out_w
            elif direction == 4:
                min_h = max(out_h - width, 0) 
                max_h = min(out_h + width, data_size - 1)
                min_w = max(out_w, 0)
                max_w = min(out_w + length * 2, data_size - 1)
                furniture_cen_h = out_h
                furniture_cen_w = out_w + length
            category_mask_wFurniture[min_h:max_h+1, min_w:max_w+1] = category + 100
           
        elif category == 2:
            out_h, out_w, width, direction = stoveArrange(cen_h, cen_w, cent_door_h, cent_door_w ,start_h, start_w, end_h, end_w)

            if direction == 1:
                length = width
                width = int(abs(end_w - start_w) / 2)
                min_h = max(out_h - length * 2, 0) 
                max_h = min(out_h, data_size - 1)
                min_w = max(out_w - width, 0)
                max_w = min(out_w + width, data_size - 1)
                furniture_cen_h = out_h - length
                furniture_cen_w = out_w
            elif direction == 2:
                length = width
                width = int(abs(end_h - start_h) / 2)
                min_h = max(out_h - width, 0) 
                max_h = min(out_h + width, data_size - 1)
                min_w = max(out_w - length * 2, 0)
                max_w = min(out_w, data_size - 1)
                furniture_cen_h = out_h
                furniture_cen_w = out_w - length
            elif direction == 3:
                length = width
                width = int(abs(end_w - start_w) / 2)
                min_h = max(out_h , 0) 
                max_h = min(out_h + length * 2, data_size - 1)
                min_w = max(out_w - width, 0)
                max_w = min(out_w + width, data_size - 1)
                furniture_cen_h = out_h + length
                furniture_cen_w = out_w
            elif direction == 4:
                length = width
                width = int(abs(end_h - start_h) / 2)
                min_h = max(out_h - width, 0) 
                max_h = min(out_h + width, data_size - 1)
                min_w = max(out_w, 0)
                max_w = min(out_w + length * 2, data_size - 1)
                furniture_cen_h = out_h
                furniture_cen_w = out_w + length
            category_mask_wFurniture[min_h:max_h+1, min_w:max_w+1] = category + 100
            
        elif category == 9:
            out_h, out_w, width ,direction = wmArrange(cen_h, cen_w ,start_h, start_w, end_h, end_w)

            if direction == 1:
                length = width
                width = int(abs(end_w - start_w) / 2)
                min_h = max(out_h - length * 2, 0) 
                max_h = min(out_h, data_size - 1)
                min_w = max(out_w - width, 0)
                max_w = min(out_w + width, data_size - 1)
                furniture_cen_h = out_h - length
                furniture_cen_w = out_w
            elif direction == 2:
                length = width
                width = int(abs(end_h - start_h) / 2)
                min_h = max(out_h - width, 0) 
                max_h = min(out_h + width, data_size - 1)
                min_w = max(out_w - length * 2, 0)
                max_w = min(out_w, data_size - 1)
                furniture_cen_h = out_h
                furniture_cen_w = out_w - length
            elif direction == 3:
                length = width
                width = int(abs(end_w - start_w) / 2)
                min_h = max(out_h , 0) 
                max_h = min(out_h + length * 2, data_size - 1)
                min_w = max(out_w - width, 0)
                max_w = min(out_w + width, data_size - 1)
                furniture_cen_h = out_h + length
                furniture_cen_w = out_w
            elif direction == 4:
                length = width
                width = int(abs(end_h - start_h) / 2)
                min_h = max(out_h - width, 0) 
                max_h = min(out_h + width, data_size - 1)
                min_w = max(out_w, 0)
                max_w = min(out_w + length * 2, data_size - 1)
                furniture_cen_h = out_h
                furniture_cen_w = out_w + length
            category_mask_wFurniture[min_h:max_h+1, min_w:max_w+1] = category + 100
        
        else:
            mask_size = 5
            min_h = max(cen_h - mask_size, 0) 
            max_h = min(cen_h + mask_size, data_size - 1)
            min_w = max(cen_w - mask_size, 0)
            max_w = min(cen_w + mask_size, data_size - 1)
            category_mask_wFurniture[min_h:max_h+1, min_w:max_w+1] = category + 100
            furniture_cen_h = cen_h
            furniture_cen_w = cen_w

        if index_mask[furniture_cen_h, furniture_cen_w] != room_i:
            furniture_cen_h = cen_h
            furniture_cen_w = cen_w

        room_info = {}
        room_info['index'] = room_i
        room_info['categary'] = category
        room_info['door_w'] = cent_door_w
        room_info['door_h'] = cent_door_h
        room_info['furniture_w'] = furniture_cen_w
        room_info['furniture_h'] = furniture_cen_h
        room_info['min_w'] = min_w
        room_info['min_h'] = min_h
        room_info['max_w'] = max_w
        room_info['max_h'] = max_h
        
        rooms_info.append(room_info)
    
    # plt.imshow(category_mask_wFurniture)
    # plt.show()
    return rooms_info

def get_movements_livingRoom(node_rooms, category_mask):

    num_rooms = len(node_rooms)
    if num_rooms < 2:
        return None

    birrt_mask = np.zeros(category_mask.shape, dtype=np.uint8)
    birrt_mask[category_mask == 0] = 1
    birrt_mask[category_mask == 17] = 1

    movements_livingRoom = []

    for i in range(num_rooms):
        for j in range(i+1, num_rooms):
            node_start = node_rooms[i]
            node_goal = node_rooms[j]
            for k in range(5):
                movements = birrt.get_finalpath(node_start, node_goal, birrt_mask)
                if movements is not None:
                    movements_livingRoom.append(movements)

    return movements_livingRoom

def get_activityMap(rooms_info, category_mask, index_mask):

    data_size = 256
    node_rooms = []
    movements_otherRooms = []

    for room_info in rooms_info:

        room_i = room_info['index']
        category = room_info['categary']
        cent_door_w = room_info['door_w']
        cent_door_h = room_info['door_h']
        furniture_cen_w = room_info['furniture_w']
        furniture_cen_h = room_info['furniture_h']

        node_rooms.append(birrt.Node(cent_door_w, cent_door_h))

        if category != 0:
            node_start = birrt.Node(cent_door_w, cent_door_h)
            node_goal = birrt.Node(furniture_cen_w, furniture_cen_h)

            birrt_mask = np.zeros(category_mask.shape, dtype=np.uint8)
            birrt_mask[index_mask == room_i] = 1
            birrt_mask[category_mask == 17] = 1

            movements = birrt.get_finalpath(node_start, node_goal, birrt_mask)
            if movements is not None:
                movements_otherRooms.append(movements)

    if movements_otherRooms is None:
        return None
    activityMap_otherRooms = birrt.get_activityMap(movements_otherRooms)
    if activityMap_otherRooms is None:
        return None

    movements_livingRoom = get_movements_livingRoom(node_rooms, category_mask)
    if movements_livingRoom is None:
        return None
    activityMap_livingRoom = birrt.get_activityMap(movements_livingRoom) 
    if activityMap_livingRoom is None:
        return None
    
    output_activityMap = np.zeros((data_size, data_size))
    output_activityMap = (0.4 * activityMap_otherRooms + 0.6 * activityMap_livingRoom) * 255
    output_activityMap = np.uint8(output_activityMap)

    # plt.imshow(output_activityMap)
    # plt.show()

    return output_activityMap

def main(input_dir, output_dir):

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    for name_fp in os.listdir(input_dir):

        print(name_fp)
        floorplan_path = os.path.join(input_dir, name_fp)
        with Image.open(floorplan_path) as temp:
            image_array_ini = np.asarray(temp, dtype=np.uint8)

        image_array = copy.deepcopy(image_array_ini)
        #boundary_mask = image_array[:,:,0]
        category_mask = image_array[:,:,1]
        index_mask = image_array[:,:,2]
        #inside_mask = image_array[:,:,3]

        rooms_info = furnitureArrange(category_mask, index_mask)
        if len(rooms_info) == 0:
            continue

        activityMap = get_activityMap(rooms_info, category_mask, index_mask)
        if activityMap is None:
            continue

        output_image_array = copy.deepcopy(image_array_ini)
        output_image_array[:,:,2] = activityMap

        output = Image.fromarray(np.uint8(output_image_array))
        id_fp = name_fp.split('.')[0]
        output.save(f'{output_dir}/{id_fp}.png')

if __name__=='__main__':

    input_dir = f"dataset_rplan"
    output_dir = f"dataset_4c"
    main(input_dir, output_dir)
