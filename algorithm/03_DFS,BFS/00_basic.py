# 1) stack
stack = []
stack.append(1)
stack.append(2)
stack.append(3)
stack.append(4)
stack.pop()
stack.append(5)
stack.append(6)
stack.pop()
print('stack:', stack)
print('stack:', stack[::-1])

# 2) queue
from collections import deque
queue = deque()

queue.append(1)
queue.append(2)
queue.append(3)
queue.append(4)
queue.popleft()
queue.append(5)
queue.append(6)
queue.popleft()

print('queue:', queue)
queue.reverse()
print('queue:', queue)

# 3) recursive
def factorial_recursive(n):
    if n<=1: return 1
    result = n * factorial_recursive(n - 1)
    print(f'n:{n},result:{result}')
    return result
print(factorial_recursive(5))