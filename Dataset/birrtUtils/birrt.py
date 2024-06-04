import copy
import random
from math import atan2, cos, sin, sqrt
from PIL import Image
import time
from .heatmap import HeatMap

XDIM = 256
YDIM = 256
EPSILON = 3.0
NUMNODES = 500
RADIUS = 5

def dist(p1, p2):
    return sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))

def step_from_to(p1, p2):
    if dist(p1, p2) < EPSILON:
        return p2
    else:
        theta = atan2(p2[1] - p1[1], p2[0] - p1[0])
    return p1[0] + EPSILON * cos(theta), p1[1] + EPSILON * sin(theta)

def chooseParent(nn, newnode, nodes, birrt_mask):
    for p in nodes:
        if (birrt_mask[newnode.y, newnode.x] == 1) and dist([p.x, p.y], [newnode.x, newnode.y]) <RADIUS and p.cost + dist([p.x, p.y], [newnode.x, newnode.y]) < nn.cost + dist([nn.x, nn.y], [newnode.x, newnode.y]):
            nn = p
    newnode.cost = nn.cost + dist([nn.x, nn.y], [newnode.x, newnode.y])
    newnode.parent = nn
    return newnode, nn

def reWire(nodes, newnode, birrt_mask):
    for i in range(len(nodes)):
        p = nodes[i]
        if (birrt_mask[newnode.y, newnode.x] == 1) and p!=newnode.parent and dist([p.x,p.y],[newnode.x,newnode.y]) <RADIUS and newnode.cost+dist([p.x,p.y],[newnode.x,newnode.y]) < p.cost:
            p.parent = newnode
            p.cost = newnode.cost + dist([p.x,p.y], [newnode.x,newnode.y])
            nodes[i] = p   
    return nodes

def drawSolutionPath(start, goal,nodes):
    nn = nodes[0]
    for p in nodes:
        if dist([p.x,p.y], [goal.x, goal.y]) < dist([nn.x, nn.y], [goal.x, goal.y]):
            nn = p
    while (nn.x, nn.y) != (start.x, start.y):
        nn = nn.parent

class Node:
    x = 0
    y = 0
    cost = 0  
    parent = None
    def __init__(self, xcoord, ycoord):
        self.x = int(xcoord)
        self.y = int(ycoord)

def extend(nodes, birrt_mask):
    # This function is to sample a new configuration and extend the tree toward that direction
    rand = Node(random.random() * XDIM, random.random() * YDIM)
    nn = nodes[0]
    for p in nodes:
        if dist([p.x, p.y], [rand.x, rand.y]) < dist([nn.x, nn.y], [rand.x, rand.y]):
            nn = p
    interpolatedNode = step_from_to([nn.x, nn.y], [rand.x, rand.y])
    newnode = Node(interpolatedNode[0], interpolatedNode[1])
  
    if birrt_mask[newnode.y, newnode.x] == 1:
        [newnode, nn]=chooseParent(nn, newnode, nodes, birrt_mask)
        nodes.append(newnode)
        nodes = reWire(nodes, newnode, birrt_mask)
    return nodes


def connect(nodes, birrt_mask):
    # this function is to sample a new configuration and connect the tree
    # to the sampled configuration unless there is an obstacle in between
    # the nearest node in the tree and configuration node
    rand = Node(random.random() * XDIM, random.random() * YDIM)
    nn = nodes[0]
    for p in nodes:
        if dist([p.x, p.y], [rand.x, rand.y]) < dist([nn.x, nn.y], [rand.x, rand.y]):
            nn = p

    # extend the tree till the sampled configuration
    while((nn.x, nn.y) != (rand.x, rand.y)):
        # or till an obstacle is found in the middle
        if birrt_mask[rand.y, rand.x] == 1:
            interpolatedNode = step_from_to([nn.x, nn.y], [rand.x, rand.y])
            newnode = Node(interpolatedNode[0], interpolatedNode[1])
            if birrt_mask[newnode.y, newnode.x] == 1: 
                [newnode, nn]=chooseParent(nn, newnode, nodes, birrt_mask)
                nodes.append(newnode)
                nodes = reWire(nodes, newnode, birrt_mask)
            nn = newnode
        else:
            break
    return nodes

def find_q_nearest(nodes, target):
    # a criteria to connect the trees by finding the 
    # nearest node the target node in the second tree
    q_near = nodes[0]
    ccost = 9999
    nodes_near = []
    for node in nodes:
        if dist([target.x, target.y], [node.x, node.y])<RADIUS:
            nodes_near.append(node)
    for node in nodes_near:
        if node.cost < ccost:
            q_near = copy.deepcopy(node)
            ccost = node.cost
    return q_near

def get_path(start, goal, nodes):
    # returns a path from start to goal from a list of nodes
    ret_nodes = []
    nn = nodes[0]
    for p in nodes:
        if dist([p.x, p.y], [goal.x, goal.y]) < dist([nn.x, nn.y], [goal.x, goal.y]):
            nn = p
    while nn != start:
        ret_nodes.append(nn)
        nn = nn.parent
    return ret_nodes

def reverse_path(parent, nodes):
    # reverst the parent and child in the second tree to combine the both trees
    ret_nodes = []
    cur_parent = parent
    if len(nodes) != 0:
        cur_node = nodes[0]
        while cur_node != None:
            newnode = Node(cur_node.x, cur_node.y)
            newnode.parent = cur_parent
            cur_node = cur_node.parent
            cur_parent = newnode
            ret_nodes.append(newnode)
    return ret_nodes

def path_length(nodes):
    length = 0
    for i in range(len(nodes)-1):
        length += dist([nodes[i].x, nodes[i].y], [nodes[i+1].x, nodes[i+1].y])
    return length

def get_finalpath(node_start, node_goal, birrt_mask):
  
    # t = time.time()
    nodes_from_root = []
    nodes_from_goal = []

    nodes_from_root.append(node_start) # Start in the corner (upper left)
    nodes_from_goal.append(node_goal) # Start in the corner (lower right)

    # two different starting points for two trees
    start_root = nodes_from_root[0]
    start_goal = nodes_from_goal[0]

    # corresponding goals for the two trees
    q_nearest = None
    q_target = nodes_from_goal[0]

    for i in range(NUMNODES):
        if(i % 2):
            nodes_from_root = extend(nodes_from_root, birrt_mask)
        else:
            nodes_from_goal = extend(nodes_from_goal, birrt_mask)
            q_target = nodes_from_goal[len(nodes_from_goal)-1]
        q_nearest = find_q_nearest(nodes_from_root, q_target)

        # check if the target node and nearest nodes are close enough for the trees to be connected
        if(dist([q_target.x, q_target.y], [q_nearest.x, q_nearest.y]) < RADIUS):
            if birrt_mask[q_target.y, q_target.x] == 1:
      
                newnode = Node(q_target.x, q_target.y)
                newnode.parent = q_nearest
                nodes_from_root.append(newnode)
          
            break
    # nodes_expanded = len(nodes_from_root) + len(nodes_from_goal)

    # get the path from root to the nearest node to the target in the second tree
    pppath = get_path(start_root, q_nearest, nodes_from_root)

    # get the path from the goal to the target
    ppath = get_path(start_goal, q_target, nodes_from_goal)

    # reverse the relationship between the nodes and reverse the elements
    ppath = reverse_path(q_nearest, ppath)
    ppath.reverse()

    # add the second path to the first path
    ppath.extend(pppath)

    # elapsed = time.time() - t
    # path_len = path_length(ppath)
    # print(f'{nodes_expanded}, {elapsed}, {path_len}')

    if i < NUMNODES:
        return ppath
    else:
        return None

def get_activityMap(movements):
    data = []
    for i in range(len(movements)):
        for j in range(1, len(movements[i])):
            tmp = [int(movements[i][j].x), int(movements[i][j].y), 1]
            data.append(tmp)
    
    background = Image.new("RGB", (XDIM, YDIM), color=0)
    hm = HeatMap(data)
    hm_img, activityMap = hm.heatmap(base=background, r = 5)
    
    return activityMap