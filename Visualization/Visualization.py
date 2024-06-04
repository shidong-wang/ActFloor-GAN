import os
import cv2
import numpy as np
from pythonds.basic.stack import Stack
from turtle import*

allData = {}
canvas_size = 256
canvas_size_k = canvas_size // 256

# input data
inputDir = 'inputFPs'
inputFPs = os.listdir(inputDir)
# ouput data
outputDir = 'outputFPs'
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

color_map = np.array([
    [255, 0,   0  ],     # Living room 0
    [156, 102, 31 ],     # Bedroom  1
    [255, 255, 0  ],     # kitchen 2
    [255, 97,  0  ],     # Bathroom 3
    [128, 42,  42 ],     # Balcony 4
    [0,   255, 0  ],     # OtherRoom 5
    [65,  105, 225],     # SecondRoom 6
    [0,   255, 255],     # StudyRoom 7
    [0,   0,   0  ],     # Exterior wall 8
    [127, 127, 127],     # FrontDoor 9
    [0,   0,   0  ],     # Interior wall 10
    [127, 127, 127]      # External 11
], dtype=np.int64)

color_map[:, [0, 2]] = color_map[:, [2, 0]]

colors ={
    'ExteriorWall': color_map[8].tolist(),
    'InteriorWall': color_map[10].tolist(),
    'FrontDoor':    color_map[9].tolist(),
    'Door':         color_map[9].tolist(), 
    #'Door':         color_map[10].tolist(),
    'Window':       color_map[8].tolist(),
    'Balcony':      color_map[4].tolist(),
    'Bathroom':     color_map[3].tolist(),
    'LivingRoom':   color_map[0].tolist(),
    'Kitchen':      color_map[2].tolist(),
    'MasterRoom':   color_map[1].tolist(),
    'SecondRoom':   color_map[6].tolist(),
    'StudyRoom':    color_map[7].tolist(),
    'OpenWall':     color_map[9].tolist(),
    #'OpenWall':     color_map[10].tolist(),
    'OtherRoom':    color_map[5].tolist()
}

def door(data, index, name):
    lineData = data[index].strip('\n')
    frontDoor = lineData.split() 
    allData[name] = {
        'x': int(frontDoor[0]),
        'y': int(frontDoor[1]),
        'width': int(frontDoor[2]),
        'height': int(frontDoor[3]),
        'direction': frontDoor[4],
    }
        
def wall(data, index, name):
    allData[name] =[]
    index += 2
    while ( not('}' in data[index]) ):
        lineData = data[index].strip('\n')
        lineMidData = lineData.split() 
        lineMid = [int(lineMidData[0]), int(lineMidData[1])]
        allData[name].append(lineMid)
        index += 1
    return index

def room(data, index, name, nameNum = 0):
    if name not in allData.keys():
        allData[name] = {}
    allData[name][nameNum] ={}
    index += 2
    line1 = data[index].strip('\n')
    if line1 == 'Boundary':
        iw = Stack()
        iw.push('{')
        index += 2
        lineMid = []
        while not iw.isEmpty():
            if('}' in data[index]):
                iw.pop()
                index += 1
                continue
            lineData = data[index].strip('\n')
            lineMidChildData = lineData.split()

            lineMid.append([int(lineMidChildData[0]), int(lineMidChildData[1])])
            index += 1
        allData[name][nameNum]['Boundary'] = lineMid
        
        line1 = data[index].strip('\n')
    if line1 == 'Door' or line1 =='OpenWall':
        index += 2
        lineData = data[index].strip('\n')
        door = lineData.split() 
        allData[name][nameNum][line1] = {
        'x': int(door[0]),
        'y': int(door[1]),
        'width': int(door[2]),
        'height': int(door[3]),
        'direction': door[4],
        }
        index += 2
        line1 = data[index].strip('\n')

    if line1 =='Window':
        iw = Stack()
        iw.push('{')
        index += 2
        lineMid = 0
        allData[name][nameNum]['Window'] = {}
        while not iw.isEmpty():
            if('}' in data[index]):
                iw.pop()
                index += 1
                continue
            lineData = data[index].strip('\n')
            window = lineData.split() 
            allData[name][nameNum]['Window'][lineMid] = {
                'x': int(window[0]),
                'y': int(window[1]),
                'width': int(window[2]),
                'height': int(window[3]),
                'direction': window[4],
            }
            lineMid += 1
            index += 1
    return index

def drawRect(name, data, bgr_img):
    pt1 = (data['x'] * canvas_size_k, data['y'] * canvas_size_k) 
    pt2 = ((data['x'] + data['width']) * canvas_size_k, (data['y'] + data['height']) * canvas_size_k) 
    
    cv2.rectangle(bgr_img, pt1, pt2, colors[name], thickness=-1)
    if name == 'Door' or name == 'OpenWall':
        cv2.rectangle(bgr_img, pt1, pt2, colors['InteriorWall'], thickness=1)
    if name == 'Window':
        cv2.rectangle(bgr_img, pt1, pt2, colors['ExteriorWall'], thickness=1)
        if data['width'] > data['height']:
            mid_y = (pt1[1] + pt2[1]) // 2
            cv2.line(bgr_img, (pt1[0], mid_y), (pt2[0], mid_y), colors['ExteriorWall'], 1)
        else:
            mid_x = (pt1[0] + pt2[0]) // 2
            cv2.line(bgr_img, (mid_x, pt1[1]), (mid_x, pt2[1]), colors['ExteriorWall'], 1)

def drawPoly(name, data, bgr_img):
    triangle = np.array(data) * canvas_size_k
    cv2.fillPoly(bgr_img, [triangle], colors[name])

def drawRoom(name, bgr_img):
    if name not in allData:
        return None
    Room = allData[name]
    for i in range(len(Room)):
        if len(Room[i]['Boundary']) < 4:
            continue
        drawPoly(name, Room[i]['Boundary'], bgr_img)

def drawRoomDW(name, bgr_img, doorName, windowName = 'Window'):
    if name not in allData:
        return None
    Room = allData[name]
    for i in range(len(Room)):
        if len(Room[i]['Boundary']) < 4:
            continue
        if doorName in Room[i].keys():
            drawRect(doorName, Room[i][doorName], bgr_img)
        if windowName in Room[i].keys():
            for j in range(len(Room[i][windowName])):
                drawRect(windowName, Room[i][windowName][j], bgr_img)

test_number = 0
for inputFP in inputFPs:
    test_number += 1
    if test_number % 1000 == 0:
        print(test_number)

    allData = {}
    inputFPPath = inputDir + '/' + inputFP

    with open(inputFPPath, "r") as f:
        data = f.readlines()
        i = 0
        index_MasterRoom = 0
        index_SecondRoom = 0
        index_Balcony = 0
        index_BathRoom = 0
        index_LivingRoom = 0

        while(i < len(data)):
            line = data[i].strip('\n')
            if line == "FrontDoor":
                i += 2
                door(data, i, "FrontDoor")

            if line =='ExteriorWall':
                i = wall(data, i, 'ExteriorWall')

            
            if line =='InteriorWall':
                iw = Stack()
                iwNum = -1
                iwNumMid = 0
                allData['InteriorWall'] ={}
                iw.push('{')
                i += 2
                while not iw.isEmpty():
                    
                    if('\n' == data[i]):
                        i +=1
                        continue
                    if '{' in data[i]:
                        i += 1
                        iwNum += 1
                        lineMid = []
                        iw.push('{')
                        continue
                    if '}' in data[i]:
                        i += 1
                        allData['InteriorWall'][iwNum] = lineMid
                        iw.pop()
                        continue
                    lineData = data[i].strip('\n')
                    lineMidChildData = lineData.split()
                    lineMidChildData = [int(lineMidChildData[0]),int(lineMidChildData[1])]
                    lineMid.append(lineMidChildData)
                    i += 1
                    
            if line =='MasterRoom':
                i = room(data, i, 'MasterRoom', index_MasterRoom)
                index_MasterRoom += 1

            if line =='Balcony':
                i = room(data, i, 'Balcony', index_Balcony)
                index_Balcony += 1

            if line =='Bathroom':
                i = room(data, i, 'Bathroom', index_BathRoom)
                index_BathRoom +=1

            if line =='Kitchen':
                i = room(data, i, 'Kitchen')
            
            if line =='StudyRoom':
                i = room(data, i, 'StudyRoom')

            if line =='SecondRoom':
                i = room(data, i, 'SecondRoom', index_SecondRoom)
                index_SecondRoom += 1

            if line =='LivingRoom':
                i = room(data, i, 'LivingRoom', index_LivingRoom)
                index_LivingRoom += 1
            i +=1
    
    img = np.ones((canvas_size, canvas_size), dtype=np.uint8)
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    output_img[:,:,0] = 255
    output_img[:,:,1] = 255
    output_img[:,:,2] = 255

    # 1. draw rooms
    drawRoom('Balcony', output_img)
    drawRoom('Bathroom', output_img)
    drawRoom('Kitchen', output_img)
    drawRoom('MasterRoom', output_img)
    drawRoom('SecondRoom', output_img)
    drawRoom('LivingRoom', output_img)
    drawRoom('StudyRoom', output_img)

    # 2. draw wall and front door
    InteriorWall = allData['InteriorWall']
    for i in InteriorWall:
        if len(InteriorWall[i]) < 4:
            continue
        drawPoly('InteriorWall', InteriorWall[i], output_img)
    drawRect('FrontDoor', allData['FrontDoor'], output_img)
    drawPoly('ExteriorWall', allData['ExteriorWall'], output_img)

    # 3. draw doors and windows
    drawRoomDW('Balcony', output_img, 'OpenWall', 'Window')
    drawRoomDW('Bathroom', output_img, 'Door', 'Window')
    drawRoomDW('Kitchen', output_img, 'OpenWall', 'Window')
    drawRoomDW('MasterRoom', output_img, 'Door', 'Window')
    drawRoomDW('SecondRoom', output_img, 'Door', 'Window')
    drawRoomDW('LivingRoom', output_img, 'Door', 'Window')
    drawRoomDW('StudyRoom', output_img, 'Door', 'Window')

    # 4. save floorplan
    outputFPPath = outputDir + '/' + inputFP.split('.')[0] + '.png'
    cv2.imwrite(outputFPPath, output_img)

print(test_number)