N = int(input())
routes = input().split()

x,y = 1,1
for m in routes:
    if m=="R":
        if y+1<=N: y+=1
    elif m=="L":
        if y-1>=1: y-=1
    elif m=="U":
        if x-1>=1: x-=1
    elif m=="D":
        if x+1<=N: x+=1

print(x, y)
