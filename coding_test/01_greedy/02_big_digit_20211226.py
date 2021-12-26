def sum_by_big_order(conds, big_nums):
    n, m, k = conds
    print(f'n:{n},m:{m},k:{k}')

    big_nums_sorted = sorted(big_nums, reverse=True)
    add_num_list = []
    sum_cnt = 0
    while True:
        for _ in range(k):
            add_num_list.append(big_nums_sorted[0])
            sum_cnt+=1
            if sum_cnt>=m: break
        if sum_cnt >= m: break

        add_num_list.append(big_nums_sorted[1])
        sum_cnt += 1
        if sum_cnt >= m: break

    print('add_num_list:', add_num_list)
    return sum(add_num_list)

while True:
    print('input(exit==-1):')
    input_strs = input()
    if input_strs=="-1":
        print('--end--')
        break

    conds = map(int, input_strs.split())
    print('input number list:')
    big_nums = list(map(int, input().split()))

    result = sum_by_big_order(conds, big_nums)
    print('result:', result)
    print('---')