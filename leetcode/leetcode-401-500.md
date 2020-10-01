# LeetCode 401 - 500

## [404. Sum of Left Leaves](https://leetcode.com/problems/sum-of-left-leaves/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Tree |

```python3
def sum_of_left_leaves(root, is_left=False):
    if not root: return 0
    if not (root.left or root.right): return root.val * is_left
    return sum_of_left_leaves(root.left, True) + sum_of_left_leaves(root.right)
```

## [405. Convert a Number to Hexadecimal](https://leetcode.com/problems/convert-a-number-to-hexadecimal/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | String |

```python3
def to_hex(num):
    if num == 0: return '0'
    HEXc = '0123456789abcdef'
    HEXs = ''
    for i in range(8):
        n = num & 15
        c = HEXc[n]
        HEXs = c + HEXs
        num = num >> 4
    return HEXs.lstrip('0')
```


## 406. [Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height/)

```python3
def reconstruct_queue(people):
    people.sort(key=lambda (h, k): (-h, k))
    queue = []
    for p in people:
        queue.insert(p[1], p)
    return queue
```


## [408. Valid Word Abbreviation](https://leetcode.com/problems/valid-word-abbreviation/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | String |

```python3
def valid_word_abbreviation(word, abbr):
    i = j = 0 
    while j < len(abbr) and i < len(word): 
        if abbr[j].isalpha(): 
            if abbr[j] != word[i]: 
                return False 
            i += 1 
            j += 1 
        else: 
            if abbr[j] == '0':
                return False 
            temp = 0 
            while j < len(abbr) and abbr[j].isdigit(): 
                temp = temp * 10 + int(abbr[j]) 
                j += 1 
            i += temp  
    
    return j == len(abbr) and i == len(word)
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


## [434. Number of Segments in a String](https://leetcode.com/problems/number-of-segments-in-a-string/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | String |

```python3
def count_segments(s):
    return len(s.split())
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


## [443. String Compression](https://leetcode.com/problems/string-compression/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | String |

```python3
def compress(chars):
    anchor = write = 0
    for read, c in enumerate(chars):
        if read + 1 == len(chars) or chars[read + 1] != c:
            chars[write] = chars[anchor]
            write += 1
            if read > anchor:
                for digit in str(read - anchor + 1):
                    chars[write] = digit
                    write += 1
            anchor = read + 1
    return write
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

## [459. Repeated Substring Pattern](https://leetcode.com/problems/repeated-substring-pattern/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N<sup>2</sup>) | O(N) | String |

```python3
def repeated_substring_pattern(s):
    return s in (s + s)[1: -1]
```


## [468. Validate IP Address](https://leetcode.com/problems/validate-ip-address/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(1) | O(1) | String |

```python3
import re

chunk_IPv4 = r'([0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])'
patten_IPv4 = re.compile(r'^(' + chunk_IPv4 + r'\.){3}' + chunk_IPv4 + r'$')

chunk_IPv6 = r'([0-9a-fA-F]{1,4})'
patten_IPv6 = re.compile(r'^(' + chunk_IPv6 + r'\:){7}' + chunk_IPv6 + r'$')

def validIPAddress(IP):        
    if patten_IPv4.match(IP):
        return "IPv4"
    return "IPv6" if patten_IPv6.match(IP) else "Neither" 
```