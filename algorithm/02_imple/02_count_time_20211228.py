end_hour = int(input())

three_cnt = 0
for hour in range(0,24): # 0~23
    if end_hour < hour:
        break
    for min in range(0,60): # 0~59
        for sec in range(0,60):
            if '3' in str(hour):
                three_cnt+=1
                continue
            if '3' in str(min):
                three_cnt += 1
                continue
            if '3' in str(sec):
                three_cnt += 1
                continue
print(three_cnt)