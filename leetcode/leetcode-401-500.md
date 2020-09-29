# LeetCode 401 - 500

## 406. [Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height/)

```python3
def reconstruct_queue(people):
    people.sort(key=lambda (h, k): (-h, k))
    queue = []
    for p in people:
        queue.insert(p[1], p)
    return queue
```


## 412. [Fizz Buzz](https://leetcode.com/problems/fizz-buzz/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(1) | String |

```python3
def fizzBuzz(n):
    ans = []
    fizz_buzz_dict = {3 : "Fizz", 5 : "Buzz"}

    for num in range(1, n+1):
        num_ans_str = ""
        for key in fizz_buzz_dict.keys():
            if num % key == 0:
                num_ans_str += fizz_buzz_dict[key]

        if not num_ans_str:
            num_ans_str = str(num)

        ans.append(num_ans_str)  

    return ans
```


## 415. [Add Strings](https://leetcode.com/problems/add-strings/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(m + n) | O(m + n) | String |

```python3
def addStrings(num1, num2):
    x1 = x2 = 0
    for i in num1:
        x1 = x1 * 10 + int(i)
    for j in num2:
        x2 = x2 * 10 + int(j)
    x = x1 + x2
    return str(x)
```

## 417. [Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N * M) | O(N * M) | Graph, DFS |

```python3
def pacific_atlantic(matrix):
    if not matrix: return []
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    m = len(matrix)
    n = len(matrix[0])
    p_visited = [[False for _ in range(n)] for _ in range(m)]
    
    a_visited = [[False for _ in range(n)] for _ in range(m)]
    result = []

    def dfs(matrix, i, j, visited, m, n):
        visited[i][j] = True
        for dir in directions:
            x, y = i + dir[0], j + dir[1]
            if x < 0 or x >= m or y < 0 or y >= n or visited[x][y] or matrix[x][y] < matrix[i][j]:
                continue
            dfs(matrix, x, y, visited, m, n)
    
    for i in range(m):
        dfs(matrix, i, 0, p_visited, m, n)
        dfs(matrix, i, n-1, a_visited, m, n)
    for j in range(n):
        dfs(matrix, 0, j, p_visited, m, n)
        dfs(matrix, m-1, j, a_visited, m, n)
        
    for i in range(m):
        for j in range(n):
            if p_visited[i][j] and a_visited[i][j]:
                result.append([i,j])
    return result
```


## 424. [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n) | O(n) | String, Sliding Window |

```python3
def character_replacement(s, k):
    count = {}
    max_count = start = result = 0
    for end in range(len(s)):
        count[s[end]] = count.get(s[end], 0) + 1
        max_count = max(max_count, count[s[end]])
        if end - start + 1 - max_count > k:
            count[s[start]] -= 1
            start += 1
        result = max(result, end - start + 1)
    return result
```

## 435. [Non-Overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(nlogn) | O(1) | Greedy |

```python3
def erase_overlapping_intervals(intervals):
    if not intervals: 
        return 0
    intervals.sort(key=lambda x: x.start)
    running_end, erased = intervals[0].end, 0
    for interval in intervals[1:]:
        if interval.start < running_end:
            erased += 1
            running_end = min(running_end, interval.end)
        else:
            running_end = interval.end   # update end time
    return erased
```

## 438. Find All Anagrams in a String

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n) | O(n) | String |

```python3
def find_anagrams(s, p):
    res = []
    pCounter = collections.Counter(p)
    sCounter = collections.Counter(s[:len(p) - 1])
    for i in range(len(p) - 1, len(s)):
        sCounter[s[i]] += 1
        if sCounter == pCounter:
            res.append(i - len(p) + 1)
        sCounter[s[i - len(p) + 1]] -= 1
        if sCounter[s[i - len(p) + 1]] == 0:
            del sCounter[s[i - len(p) + 1]]
    return res
```


## 448. [Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)

```
def find_disappeared_numbers(nums):
    nums = [0] + nums
    for i in range(len(nums)):
        index = abs(nums[i])
        nums[index] = -abs(nums[index])

    return [i for i in range(len(nums)) if nums[i] > 0]
```
