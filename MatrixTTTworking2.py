# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:03:48 2017

@author: qoliver
"""
import numpy as np
import random
import tensorflow as tf

def makeMatrix():
    matrix=np.zeros((3,3))
    return matrix



def placeX(matrix, x, y):
    matrix[y-1][x-1]=2.0
    
def placeO(matrix, x, y):
    matrix[y-1][x-1]=1.0
#Place pieces on grid
def horizCheck(matrix):
    win = False
    for i in range(0,3):  
        if matrix[i][0]==matrix[i][1]==matrix[i][2] and  matrix[i][0] != 0.0:
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
    moves=[[1,1],[1,2],[1,3],[2,2],[2,1],[2,3],[3,1],[3,2],[3,3]]
    
    fileList=[]
    winner = "next"     
    count = 0
    while win==False:
        try:
            win = True
            turnX=random.choice(moves)
            print('X:')
            
            moves.remove(turnX)
            fileList.append(turnX)
            placeX(m, turnX[0], turnX[1])
            print(m)
            win = diagCheck(m)
            if win == False:
                win = vertCheck(m)  
            if win == False:
                win = horizCheck(m)
            if horizCheck(m) == True or vertCheck(m) == True or diagCheck(m) == True:
                if winner == 'next':
                    winner ="X"
                    count += 1
            turnO=random.choice(moves)
            print('O:')
            
            moves.remove(turnO)
            placeO(m, turnO[0], turnO[1])
            print(m)
            win = diagCheck(m)
            if win == False:
                win = vertCheck(m)  
            if win == False:
                win = horizCheck(m)   
            fileList.append(turnO)
            if horizCheck(m) == True or vertCheck(m) == True or diagCheck(m) == True:
                if winner == 'next':
                    winner = "O"
                    count -= 1
        except:
            print("out of terms")
            break
    fileList.remove(fileList[-1])
    print(m)
    print(count)
    return count
#Plays a random match and sees who wins
#export fileList
#Sample Code


matrix = makeMatrix()
def bias(turns):
    win = 0
    loss = 0
    draw = 0
    count = 0
    for i in range(1, turns +1):
        bias = randomMatch()
        if bias == 1:
            win += 1
        elif bias == 0:
            draw += 1
        else:
            loss += 1
        count += bias
    print(win, loss, draw)
    return count


def trainingSetMakerTTT():
    m = makeMatrix()
    win=False
    moves=[[1,1],[1,2],[1,3],[2,2],[2,1],[2,3],[3,1],[3,2],[3,3]]
    fileList=[]
    winner = "next"     
    count = 0
    tie=0
    while win==False:
        try:
            
            turnX=random.choice(moves)
            print('X:')
            
            moves.remove(turnX)
            fileList.append(turnX)
            placeX(m, turnX[0], turnX[1])
            print(m)
            win = diagCheck(m)
            if win == False:
                win = vertCheck(m)  
            if win == False:
                win = horizCheck(m)
            if horizCheck(m) == True or vertCheck(m) == True or diagCheck(m) == True:
                if winner == 'next':
                    winner = "X"
                    count += 1
            turnO=random.choice(moves)
            print('O:')
            
            moves.remove(turnO)
            placeO(m, turnO[0], turnO[1])
            print(m)
            win = diagCheck(m)
            if win == False:
                win = vertCheck(m)  
            if win == False:
                win = horizCheck(m)   
            fileList.append(turnO)
            if horizCheck(m) == True or vertCheck(m) == True or diagCheck(m) == True:
                if winner == 'next':
                    winner = "O"
                    count -= 1
        except:
            print("out of terms")
            tie+=1
            break
    
    print(m)
    print(count)
    #May need to remove array command
    flat=np.array([[0,0,0,0,0,0,0,0,0]])
    flat[0][0]=m[0][0]
    flat[0][1]=m[0][1]
    flat[0][2]=m[0][2]
    flat[0][3]=m[1][0]
    flat[0][4]=m[1][1]
    flat[0][5]=m[1][2]
    flat[0][6]=m[2][0]
    flat[0][7]=m[2][1]
    flat[0][8]=m[2][2]
    fileList[0][0]=flat
    fileList[0][1]=tie
    #
    return fileList
#Runs random match and then removes certain moves

def controlledMatch(moveset):
    m = makeMatrix()
    win=False
    moves=[[1,1],[1,2],[1,3],[2,2],[2,1],[2,3],[3,1],[3,2],[3,3]]
    fileList=[]
    winner = "next"     
    count = 0    
    while True==True:
        try:
            placeX(m,moveset[0][0],moveset[0][1])
            moves.remove(moveset[0])
            moveset.remove(moveset[0])
            
            try:
                placeO(m,moveset[0][0],moveset[0][1])
                moves.remove(moveset[0])
                moveset.remove(moveset[0])
            except:
                break
        except:
            break
    #Enters controlled movelist
    while win==False:
        try:
            
            turnX=random.choice(moves)
            print('X:')
            
            moves.remove(turnX)
            fileList.append(turnX)
            placeX(m, turnX[0], turnX[1])
            print(m)
            win = diagCheck(m)
            if win == False:
                win = vertCheck(m)  
            if win == False:
                win = horizCheck(m)
            if horizCheck(m) == True or vertCheck(m) == True or diagCheck(m) == True:
                if winner == 'next':
                    winner = "X"
                    count += 1
            turnO=random.choice(moves)
            print('O:')
            
            moves.remove(turnO)
            placeO(m, turnO[0], turnO[1])
            print(m)
            win = diagCheck(m)
            if win == False:
                win = vertCheck(m)  
            if win == False:
                win = horizCheck(m)   
            fileList.append(turnO)
            if horizCheck(m) == True or vertCheck(m) == True or diagCheck(m) == True:
                if winner == 'next':
                    winner = "O"
                    count -= 1
        except:
            print("out of terms")
            break
    #Runs Random match w/ controlled moves already inputted
    fileList.remove(fileList[-1])
    print(m)
    print(count)
    fileList.append(winner)
    return count
#Returns Bias(This can be changed)

def smartMove(movesRemaining):
    i=-100000
    
    movelist=[[1,1],[1,2],[1,3],[2,2],[2,1],[2,3],[3,1],[3,2],[3,3]]
    for move in movesRemaining:
        movelist.remove(move)
    for move in movesRemaining:
        movetemp=movelist.append(move)
        b=0
    
        for i in range(1, 10 +1):
            bias = controlledMatch(movetemp)
            b+=bias
        if b >= i:
            movefinal = move
    return movefinal
            
wtg=np.array([[0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0]])
for i in range(1,1000):
    addee = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
    moves=[0,1,2,3,4,5,6,7,8]
    m1=random.choice(moves)
    addee[m1]=1
    moves.remove(m1)
    m2=random.choice(moves)
    addee[m2]=1
    moves.remove(m2)
    m3=random.choice(moves)
    addee[m3]=1
    moves.remove(m3)
    m4=random.choice(moves)
    addee[m4]=1
    moves.remove(m4)
    m5=random.choice(moves)
    addee[m5]=1
    moves.remove(m5)
    squaree = np.array(addee)[:,None]
    moveset=squaree*squaree.T
    wtg = wtg + moveset
    #Makes a semi-constant(signs are consant, values are not)
    
'''Key:wtg=weighting matrix
       addee:set to be converted to array
       squaree:array to be squared
       moveset:matrix of the one loop
Notes for lecture:
        -Find good algorithm(not random)
        -Apply O only, whole board filters
        -Comment Yourself:

'''



training_data=[]
training_labels=[]
for i in range(100):
    t=trainingSetMakerTTT()
    training_data.append(t[0][0])
    training_labels.append(t[0][1])
train_data=np.array(training_data)
train_labels=np.array([training_labels])

testing_data=[]
testing_labels=[]
for i in range(100):
    t=trainingSetMakerTTT()
    testing_data.append(t[0][0])
    testing_labels.append(t[0][1])
test_data=np.array(training_data)
test_labels=np.array([training_labels])
