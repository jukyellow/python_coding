pos = input()

y = ord(pos[0])-96
x = int(pos[1])
# print('x:',x, ',y:',y)

def chk_cond(dir, cond1,cond2,cond3, MIN=1, MAX=8):
    cnt=0
    if dir=="L":
        if cond1>=MIN:
            if cond2>=MIN:
                cnt+=1
            if cond3<=MAX:
                cnt+=1
    else:
        if cond1<MAX:
            if cond2>=MIN:
                cnt+=1
            if cond3<=MAX:
                cnt+=1
    return cnt

m_cnt = 0
m_cnt+= chk_cond("L", x-2, y-1, y+1)
m_cnt+= chk_cond("L", y-2, x-1, x+1)
m_cnt+= chk_cond("R", x+2, y-1, y+1)
m_cnt+= chk_cond("R", y+2, x-1, y+1)

print(m_cnt)