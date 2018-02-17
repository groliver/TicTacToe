# -*- coding: utf-8 -*-
def makeGrid():
    grid={}
    return grid

def placeX(grid,place):
    grid[place]="X"
    return grid

def placeO(grid,place):
    grid[place]="O"
    return grid
def printData(grid):
    print('_l_l_\n_l_l_\n l l')
    print("Coordinate System")
    print('A1 A2 A3\nB1 B2 B3\nC1 C2 C3')
    print(grid)
    
def vertCheck(grid,win):
    for x in range(1, 4):
        x=str(x)
        try:
            if grid["A"+x]==grid["B"+x]==grid["C"+x]:
                print(grid["A"+x]+" is the winner!")
                win = True
        except:
            continue
    return win

def horizCheck(grid,win):
    x=1
    try:
        if grid["A"+str(x)]==grid["A"+str(x+1)]==grid["A"+str(x+2)]:
            print(grid["A"+x]+" is the winner!")
            win = True
        elif grid["B"+str(x)]==grid["B"+str(x+1)]==grid["B"+str(x+2)]:
            print(grid["B"+x]+" is the winner!")
            win = True
        elif grid["C"+str(x)]==grid["C"+str(x+1)]==grid["C"+str(x+2)]:
            print(grid[""+x]+" is the winner!")
            win = True
        
    except:
        try:
            if grid["B"+str(x)]==grid["B"+str(x+1)]==grid["B"+str(x+2)]:
                print(grid["B"+x]+" is the winner!")
                win = True
            elif grid["C"+str(x)]==grid["C"+str(x+1)]==grid["C"+str(x+2)]:
                print(grid["C"+x]+" is the winner!")
                win = True
            elif grid["A"+str(x)]==grid["A"+str(x+1)]==grid["A"+str(x+2)]:
                print(grid["A"+x]+" is the winner!")
                win = True
        except:
            try:
                if grid["C"+str(x)]==grid["C"+str(x+1)]==grid["C"+str(x+2)]:
                    print(grid["C"+x]+" is the winner!")
                    win = True
                elif grid["A"+str(x)]==grid["A"+str(x+1)]==grid["A"+str(x+2)]:
                    print(grid["A"+x]+" is the winner!")
                    win = True
                elif grid["B"+str(x)]==grid["B"+str(x+1)]==grid["B"+str(x+2)]:
                    print(grid["B"+x]+" is the winner!")
                    win = True
            except:
                print('\n')
    return win

def diagCheck(grid, win):
    try:
        if grid["B2"]==grid["A1"]==grid["C3"]:
            print(grid["B2"]+" is the winner!")
        elif grid["B2"]==grid["A3"]==grid["C1"]:
            print(grid["B2"]+" is the winner!")
            win=True
    except:
        try:
            if grid["B2"]==grid["A3"]==grid["C1"]:
                print(grid["B2"]+" is the winner!")
            elif grid["B2"]==grid["A1"]==grid["C3"]:
                print(grid["B2"]+" is the winner!")
                win = True
        except:
            print('\n')
    return win            
def check(grid,win):
    try:
        while win == False:
            win = horizCheck(grid, win)
            win = vertCheck(grid, win)
            win = diagCheck(grid, win)
            if win == False:
                break
    except:
        try:
            while win == False:
                win = vertCheck(grid, win)
                win = diagCheck(grid, win)
                win = horizCheck(grid, win)
                if win == False:
                    break
        except:
            try:
                while win == False:
                    win = diagCheck(grid, win)
                    win = horizCheck(grid, win)
                    win = vertCheck(grid, win)
                    if win == False:
                        break
            except:
                print('\n')
    return win

def initTicTacToe():
    win=False
    grid=makeGrid()
    while win == False:
        turn = 1
        if turn <= 9:
            printData(grid)
            placeX(grid, input('Place x where?'))
            turn+=1
            check(grid, win)
            win = check(grid, win)
            placeO(grid, input('Place O where?'))
            turn+=1
            check(grid, win)
            win = check(grid, win)
        
        if turn > 9:
            print('Tie!')
            break
        
    
#May lead to multiple winners

