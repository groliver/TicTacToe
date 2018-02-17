# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:03:48 2017

@author: qoliver
"""
import numpy as np
import random

def makeMatrix():
    matrix=np.zeros((3,3))
    return matrix



def placeX(matrix, x, y):
    matrix[3-y][x-1]=2.0
    
def placeO(matrix, x, y):
    matrix[3-y][x-1]=1.0
    
def horizCheck(matrix):
    win = False
    for i in range(1,4):  
        if matrix[i-1][0]==matrix[i-1][1]==matrix[i-1][2] and  matrix[i-1][0] != 0.0:
            win=True
            if matrix[i-1][1] == 2.0:
                print("the winner is x!")
            else:
                print("the winner is o!")
    return win

def vertCheck(matrix):
    win = False
    for i in range(0, 3):
        if matrix[0][i]==matrix[1][i]==matrix[2][i] and  matrix[1][i] != 0.0:
            win=True
            if matrix[i-1][1] == 2.0:
                print("the winner is x!")
            else:
                print("the winner is o!")  
    return win

def diagCheck(matrix):
    win = False
    if matrix[0][0]==matrix[1][1]==matrix[2][2] and matrix[0][0] != 0.0:
        win=True
        if matrix[1][1] == 2.0:
            print("the winner is x!")
        else:
            print("the winner is o!")  
    if matrix[0][2]==matrix[1][1]==matrix[2][0] and matrix[0][2] != 0.0:
        win=True
        if matrix[1][1] == 2.0:
            print("the winner is x!")
        else:
            print("the winner is o!")
    return win
    


def randomMatch():
    m = makeMatrix()
    win=False
    moves=[[1,1],[1,2],[1,3],[2,3],[2,1],[2,2],[3,1],[3,2],[3,3]]
    while win==False:
        win = True
        turnX=random.choice(moves)
        moves.remove(turnX)
        placeX(m, turnX[0], turnX[1])
        win = diagCheck(m)
        if win != True:
            win = vertCheck(m)  
        if win != True:
            win = horizCheck(m)
        turnO=random.choice(moves)
        moves.remove(turnO)
        placeO(m, turnO[0], turnO[1])
        win = diagCheck(m)
        if win != True:
            win = vertCheck(m)  
        if win != True:
            win = horizCheck(m)   
matrix = makeMatrix()
coord=[]        
for index,x in np.ndenumerate(matrix):
   coord.append( [index[0],index[1]])
      
