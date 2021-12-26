def get_min_coint_cnt(N, IS_LOG=False):
    cnt_500=0
    cnt_100=0
    cnt_50=0
    cnt_10=0

    (cnt_500, remain) = divmod(N, 500)
    (cnt_100, remain) = divmod(remain, 100)
    (cnt_50, remain) = divmod(remain, 50)
    (cnt_10, remain) = divmod(remain, 10)
    if IS_LOG==True:
        print(f'[Log] cnt_500:{cnt_500}, cnt_100:{cnt_100}, cnt_50:{cnt_50}, cnt_10:{cnt_10}')

    min_coin_cnt = cnt_500 + cnt_100 + cnt_50 + cnt_10
    return min_coin_cnt

while True:
    print("input return money(exit==-1):")
    input_str = input()
    if input_str=="-1":
        print('--end--')
        break

    if input_str.isdigit()==False:
        print("input only digit, retry!")
        continue

    N = int(input_str)
    min_coin_cnt = get_min_coint_cnt(N, IS_LOG=True)
    print('min_coin_cnt:', min_coin_cnt)
    print('---')
