# LeetCode 201 - 300

## 200. [Number Of Islands](https://leetcode.com/problems/number-of-islands/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(n) | Graph |

```python3
def num_islands(grid):
    def sink(i, j):
        if 0 <= i < len(grid) and 0 <= j < len(grid[i]) and grid[i][j] == '1':
            grid[i][j] = '0'
            map(sink, (i+1, i-1, i, i), (j, j, j+1, j-1))
            return 1
        return 0
    return sum(sink(i, j) for i in range(len(grid)) for j in range(len(grid[i])))
```

## 202. [Happy Number](https://leetcode.com/problems/happy-number/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(logn) | O(1) | Hash Table, Floyd's Cycle Detection |

```python3
def isHappy(n):

    def get_next(n):
        total_sum = 0
        while n > 0:
            n, digit = divmod(n, 10)
            total_sum += digit ** 2
        return total_sum

    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = get_next(n)

    return n == 1
```

## 206. [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n) | O(1) | Linked List |

```python3
def reverse_list(head):
    cur, prev = head, None
    while cur:
        cur.next, prev, cur = prev, cur, cur.next
    return prev
```

## 207. [Course Schedule](https://leetcode.com/problems/course-schedule/solution/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(∣E∣+∣V∣) | O(∣E∣+∣V∣) | Graph, Topological Sort |

```python3
def canFinish(numCourses, prerequisites):
    from collections import defaultdict, deque

    graph = defaultdict(GNode)

    totalDeps = 0
    for relation in prerequisites:
        nextCourse, prevCourse = relation[0], relation[1]
        graph[prevCourse].outNodes.append(nextCourse)
        graph[nextCourse].inDegrees += 1
        totalDeps += 1

    nodepCourses = deque()
    for index, node in graph.items():
        if node.inDegrees == 0:
            nodepCourses.append(index)

    removedEdges = 0
    while nodepCourses:
        course = nodepCourses.pop()

        for nextCourse in graph[course].outNodes:
            graph[nextCourse].inDegrees -= 1
            removedEdges += 1
            if graph[nextCourse].inDegrees == 0:
                nodepCourses.append(nextCourse)

    if removedEdges == totalDeps:
        return True
    else:
        return False
```


## 208. [Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(~) | O(N) | Trie |

```python3
class Trie:
    def __init__(self):
        self.trie = {}

    def insert(self, word):
        t = self.trie
        for w in word:
            if w not in t:
                t[w] = {}
            t = t[w]
        t['#'] = '#'

    def search(self, word):
        t = self.trie
        for w in word:
            if w not in t:
                return False
            t = t[w]
        if '#' in t:
            return True
        return False

    def starts_with(self, prefix):
        t = self.trie
        for w in prefix:
            if w not in t:
                return False
            t = t[w]
        return True
```

## [211. Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N * M) | O(M) | Trie |

```python3
class WordDictionary:

    def __init__(self):
        self.trie = {}
        

    def addWord(self, word):
        node = self.trie
        
        for ch in word:
            if not ch in node:
                node[ch] = {}
            node = node[ch]
        node['$'] = True

    def search(self, word):
        def search_in_node(word, node):
            for i, ch in enumerate(word):
                if not ch in node:
                    if ch == '.':
                        for x in node:
                            if x != '$' and search_in_node(word[i + 1:], node[x]):
                                return True
                    return False
                else:
                    node = node[ch]
            return '$' in node
            
        return search_in_node(word, self.trie)
```

## [212. Word Search II](https://leetcode.com/problems/word-search-ii/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(M(4 * 3^(L-1)) | O(N) | Trie |

```python3
def findWords(board, words):
    WORD_KEY = '$'
    trie = {}
    for word in words:
        node = trie
        for letter in word:
            node = node.setdefault(letter, {})
        node[WORD_KEY] = word

    row_num = len(board)
    col_num = len(board[0])

    matched_words = []

    def dfs(row, col, parent):    
        letter = board[row][col]
        current_node = parent[letter]

        word_match = current_node.pop(WORD_KEY, False)
        if word_match:
            matched_words.append(word_match)

        board[row][col] = '#'

        for (row_offset, col_offset) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            new_row, new_col = row + row_offset, col + col_offset     
            if new_row < 0 or new_row >= row_num or new_col < 0 or new_col >= col_num:
                continue
            if not board[new_row][new_col] in current_node:
                continue
            dfs(new_row, new_col, current_node)

        board[row][col] = letter

        if not curent_node:
            parent.pop(letter)

    for row in range(row_num):
        for col in range(col_num):
            if board[row][col] in trie:
                dfs(row, col, trie)

    return matched_words
```

## 213. [House Rober II](https://leetcode.com/problems/house-robber-ii/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | Dynamic Programming |

```python3
def rob(nums):
    def rob_simple(nums):
        t1 = 0
        t2 = 0
        for current in nums:
            temp = t1
            t1 = max(current + t2, t1)
            t2 = temp
        return t1

    if len(nums) == 0 or nums is None:
        return 0

    if len(nums) == 1:
        return nums[0]

    return max(rob_simple(nums[:-1]), rob_simple(nums[1:]))
```

## 217. [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(n) | Array, Set, Hash Table |

```python3
def contains_duplicate(nums):
    return len(nums) != len(set(nums))
```

## [226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(1) | Tree |

```
def invertTree(root):
    if root is None:
        return None
    root.left, root.right = \
        self.invertTree(root.right), self.invertTree(root.left)
    return root
```


## 230. [Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(logN + k) | O(logN) | Tree |

```python3
def kth_smallest(root, k):
    stack = []
    
    while True:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        k -= 1
        if not k:
            return root.val
        root = root.right
```



## 234. [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)

```python3
def isPalindrome(head):
    rev = None
    slow = fast = head
    while fast and fast.next:
        fast = fast.next.next
        rev, rev.next, slow = slow, rev, slow.next
    if fast:
        slow = slow.next
    while rev and rev.val == slow.val:
        slow = slow.next
        rev = rev.next
    return not rev
```


## 235. [Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | Tree |

```python3
def lowest_common_ancestor(root, p, q):
    p_val = p.val
    q_val = q.val
    node = root

    while node:
        parent_val = node.val

        if p_val > parent_val and q_val > parent_val:    
            node = node.right
        elif p_val < parent_val and q_val < parent_val:
            node = node.left
        else:
            return node
```

## 238. [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)

Make the left-product array and the right-product array and multiply them together. Or make the left-product array, then while looping through the right-product keep a running product of the right products and multiply them by the running right-left products.

```python3
def product_except_self(nums):
    left_product = 1
    n = len(nums)
    output = []
    for i in range(0, n):
        output.append(left_product)
        left_product = left_product * nums[i]
    right_product = 1
    for i in range(n-1, -1, -1):
        output[i] = output[i] * right_product
        right_product = right_product * nums[i]
    return output
```

## 242. [Valid Anagram](https://leetcode.com/problems/valid-anagram/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(nlogn) | O(n) | String |

```python3
def valid_anagram(s, t):
    return len(s) == len(t) and sorted(s) == sorted(t)
```


## 252. [Meeting Rooms](https://leetcode.com/problems/meeting-rooms/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(nlogn) | O(1) | Interval |

```python3
def can_attend_meetings(intervals):
    intervals.sort(key=lambda x: x.start)   

    for i in range(1, len(intervals)):
        if intervals[i].start < intervals[i - 1].end:
            return False
        
    return True
```

## 253. [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(nlogn) | O(n) | Interval, Sweep The Line |

```python3
def min_meeting_rooms(intervals):
    meetings = []
    for interval in intervals:
        meetings.append((interval[0], 's'))
        meetings.append((interval[1], 'e'))
    meetings.sort(key=lambda x: x[0])
    
    min_meeting_rooms = 0
    i = 0
    temp = 0
    while i < len(meetings):
        time = meetings[i][0]
        while i < len(meetings) and meetings[i][0] == time:
            if meetings[i][1] == 's':
                temp += 1
            else:
                temp -= 1
            i += 1
        min_meeting_rooms = max(min_meeting_rooms, temp)
    return min_meeting_rooms
```

## [257. Binary Tree Paths](https://leetcode.com/problems/binary-tree-paths/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Tree |

```python3
def binary_tree_paths(root):
    if not root:
        return []
    paths, queue = [], collections.deque([(root, "")])
    while queue:
        node, ls = queue.popleft()
        if not node.left and not node.right:
            paths.append(ls + str(node.val))
        if node.left:
            queue.append((node.left, ls + str(node.val) + "->"))
        if node.right:
            queue.append((node.right, ls + str(node.val) + "->"))
    return paths
```

## 261. [Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N+E) | O(N+E) | Graph, BFS, Union Find |

```python3
def valid_tree(n, edges):
    
    if len(edges) != n - 1: return False
    
    adj_list = [[] for _ in range(n)]
    for A, B in edges:
        adj_list[A].append(B)
        adj_list[B].append(A)
    
    parent = {0: -1}
    queue = collections.deque([0])
    
    while queue:
        node = queue.popleft()
        for neighbour in adj_list[node]:
            if neighbour == parent[node]:
                continue
            if neighbour in parent:
                return False
            parent[neighbour] = node
            queue.append(neighbour)
    
    return len(parent) == n
```

## 268. [Missing Number](https://leetcode.com/problems/missing-number/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | Bitwise, Sort + Binary Search |

```python3
def missing_number(nums):
    missing_number = len(nums)
    for i in range(0, len(nums) + 1):
        missing_number ^= i
        missing_number ^= nums[i]
    return missing_number
```

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | Math |

```python3
def missingNumber(nums):
    return (len(nums)*(len(nums) + 1)/2) - sum(nums)
```


## 269. [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(~) | O(~) | Graph, Topological Sort |

```python3
from collections import defaultdict, Counter, deque

def alien_order(words):
    adj_list = defaultdict(set)
    in_degree = Counter({c : 0 for word in words for c in word})
            
    for first_word, second_word in zip(words, words[1:]):
        for c, d in zip(first_word, second_word):
            if c != d:
                if d not in adj_list[c]:
                    adj_list[c].add(d)
                    in_degree[d] += 1
                break
        else:
            if len(second_word) < len(first_word): return ""
    
    output = []
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    while queue:
        c = queue.popleft()
        output.append(c)
        for d in adj_list[c]:
            in_degree[d] -= 1
            if in_degree[d] == 0:
                queue.append(d)
                
    if len(output) < len(in_degree):
        return ""
    return "".join(output)
```

## [270. Closest Binary Search Tree Value](https://leetcode.com/problems/closest-binary-search-tree-value/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(H) | O(1) | Tree |

```python3
def closest_value(root, target):
    closest = root.val
    while root:
        closest = min(root.val, closest, key = lambda x: abs(target - x))
        root = root.left if target < root.val else root.right
    return closest
```


## 283. [Move Zeroes](https://leetcode.com/problems/move-zeroes/)

```python3
def move_zeroes(nums):
    zeroes = nums.count(0)
    nums[:] = [i for i in nums if i != 0]
    nums += [0] * zeroes
```

## 284. Peeking Iterator

```python3
class PeekingIterator(object):
    def __init__(self, iterator):
        self.iter = iterator
        self.temp = self.iter.next() if self.iter.hasNext() else None

    def peek(self):
        return self.temp

    def next(self):
        ret = self.temp
        self.temp = self.iter.next() if self.iter.hasNext() else None
        return ret

    def hasNext(self):
        return self.temp is not None
```

## 287. [Find The Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n) | O(1) | Array, Flyod's Cycle Detection, Binary Search |

```python3
def find_duplicate(nums):
    for i in range(len(nums)):
        if nums[abs(nums[i])] < 0:
            return abs(nums[i])
        else:
            nums[abs(nums[i])] = nums[abs(nums[i])]*-1

def find_duplicate(nums):
    slow = len(nums) - 1
    fast = len(nums) - 1
    while True:
        slow = nums[slow]
        fast = nums[array[fast]]
        if slow == fast:
            break
    
    finder = len(nums) - 1
    while True:
        slow   = nums[slow]
        finder = nums[finder]
        if slow == finder:
            return slow
```

## 295. Find the Median From Data Stream

https://leetcode.com/problems/find-median-from-data-stream/

Simple sorting will be an O(nlogn) solution. Sort every time. Maintaining two heaps - one max and one min heap where each is maintained with equal sizes within 2 is the key to this problem.

## 300. [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n<sup>2</sup>) | O(n) | Dynamic Programming, Binary Search |

```python3
def longest_increasing_subsequence(n):
    dp = [1]*len(nums)
    for i in range (1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j]+1)
    return max(dp)
```
