# 1) DFS(Depth First Search)

# 그래프를 각 노드(index)와 연결된 edge를 리스트로 표현
graph = [
    [],
    [2,3,8],
    [1,7],
    [1,4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
]
# 방문기록
visited_list = [False]*9
def dfs(graph, v_idx, visited):
    #현재 노드 방문
    visited[v_idx] = True
    print(v_idx, end=' ')
    # 현재 노드와 연결된 다른 노드를 재귀적으로 방문
    for i in graph[v_idx]:
        if not visited[i]:
            dfs(graph, i, visited)
# 깊이우선 탐색
dfs(graph, 1, visited_list)
print('')

# 2) BFS(Breadth First Search)
from _collections import deque
def bfs(graph, start, visied):
    queue = deque([start])
    visied[start] = True
    while queue:
        v = queue.popleft()
        print(v, end=' ')
        # 해당 원소와 연결된, 아직 방문하지 않은 원소들을 큐에 삽입
        for i in graph[v]:
            if not visied[i]:
                queue.append(i)
                visied[i] = True
visited_list = [False]*9
bfs(graph, 1, visited_list)
