
## [709. To Lower Case](https://leetcode.com/problems/to-lower-case/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | String |

```python3
def to_lower_case(str):
    return "".join(chr(ord(c) + 32) if 65 <= ord(c) <= 90 else c for c in str)
```


## 739. [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)

```python3
def daily_temperatures(T):
    temps = [0] * len(T)
    stack = []
    for i, t in enumerate(T):
      while stack and T[stack[-1]] < t:
        cur = stack.pop()
        temps[cur] = i - cur
      stack.append(i)

    return temps
```

## [760. Find Anagram Mappings](https://leetcode.com/problems/find-anagram-mappings/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | String |

```python3
def anagram_mappings(A, B):
    D = {x: i for i, x in enumerate(B)}
    return [D[x] for x in A]
```

## [783. Minimum Distance Between BST Nodes](https://leetcode.com/problems/minimum-distance-between-bst-nodes/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(H) | Tree |

```python3
def min_diff_in_BST(root):
    def dfs(node):
        if node:
            dfs(node.left)
            self.ans = min(self.ans, node.val - self.prev)
            self.prev = node.val
            dfs(node.right)

    self.prev = float('-inf')
    self.ans = float('inf')
    dfs(root)
    return self.ans
```

## 794. Valid Tic-Tac-Toe State

Given a board state, determine if it is valid.

```python3
def is_win(b):
    # 0 1 2
    # 3 4 5
    # 6 7 8
    board = [0] * 9
    for i, j in [range(3), range(3)]:
        board[i+j*3] = b[i][j]
    tests = [ [0,3,6], [1,4,7], [2,5,8], [0,1,2], [3,4,5], [6,7,8], [0,4,8], [2,4,6] ]
    for a, b, c in tests:
        if board[a] == board[b] == board[c] and board[a] != '':
            return True
    return False

def valid_tic_tac_toe(board):
    count_X = count_O = 0
    for i in range(3):
        for j in range(3):
            count_X += 1 if board[i][j] == 'X' else 0
            count_O += 1 if board[i][j] == 'O' else 0
    if count_O > count_X or count_X > count_O + 1:
        return False
    if count_O == count_X and self.isWin(board, 'X') or count_X == count_O + 1 and self.isWin(board, 'O'):
        return False
    return True

```
Check the board winner with hard coded values. And verify the player counts beforehand as well.


## 819. Most Common Word

Given a paragraph of words return the most frequent. Exclude banned list of words.

```python3
def most_common_word(paragraph, banned):
    banned_words = set(banned)
    words = re.findall(r'\w+', p.lower())
    return collections.Counter(w for w in words if w not in banned_words).most_common(1)[0][0]
```


## [897. Increasing Order Search Tree](https://leetcode.com/problems/increasing-order-search-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(H) | Tree |

```python3
def increasing_BST(root):
    ans = cur = TreeNode(None)

    def inorder(node):
        if node:
            inorder(node.left)
            node.left = None
            cur.right = node
            cur = node
            inorder(node.right)

    ans = cur = TreeNode(None)
    inorder(root)
    return ans.right
```


## 937. [Reorder Data In Log Files](https://leetcode.com/problems/reorder-data-in-log-files/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(m * nlogn) | O(m * n) | String |

```python3
def reorderLogFiles(logs):

    def get_key(log):
        _id, rest = log.split(" ", maxsplit=1)
        return (0, rest, _id) if rest[0].isalpha() else (1, )

    return sorted(logs, key=get_key)
```


## [938. Range Sum of BST](https://leetcode.com/problems/range-sum-of-bst/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(H) | Tree |

```python3
def range_sum_bst(root, L, R):
    range_sum = 0

    def dfs(node):
        if node:
            if L <= node.val <= R:
                range_sum += node.val
            if L < node.val:
                dfs(node.left)
            if node.val < R:
                dfs(node.right)

    dfs(root)
    return range_sum
```


## 953. [Verifying an Alien Dictionary](https://leetcode.com/problems/verifying-an-alien-dictionary/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(m + n) | O(1) | String |

```python3
def isAlienSorted(words, order):
    order_index = {c: i for i, c in enumerate(order)}

    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i+1]

        for k in range(min(len(word1), len(word2))):
            if word1[k] != word2[k]:
                if order_index[word1[k]] > order_index[word2[k]]:
                    return False
                break
        else:
            if len(word1) > len(word2):
                return False

    return True
```


## [965. Univalued Binary Tree](https://leetcode.com/problems/univalued-binary-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(H) | Tree |

```python3
def is_unival_tree(root):
    left_correct = (not root.left or root.val == root.left.val
            and is_unival_tree(root.left))
    right_correct = (not root.right or root.val == root.right.val
            and is_unival_tree(root.right))
    return left_correct and right_correct
```


## [993. Cousins in Binary Tree](https://leetcode.com/problems/cousins-in-binary-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Tree |

```python3
def is_cousins(root, x, y):
    queue = collections.deque([root])

    while queue:
        siblings = False
        cousins = False
        nodes_at_depth = len(queue)
        for _ in range(nodes_at_depth):
            node = queue.popleft()

            if node is None:
                siblings = False
            else:
                if node.val == x or node.val == y:
                    if not cousins:
                        siblings, cousins = True, True
                    else:
                        return not siblings
                queue.append(node.left) if node.left else None
                queue.append(node.right) if node.right else None
                queue.append(None)
        if cousins:
            return False
    return False
```


## [1022. Sum of Root To Leaf Binary Numbers](https://leetcode.com/problems/sum-of-root-to-leaf-binary-numbers/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(H) | Tree |

```python3
def sum_root_to_leaf(root):
    root_to_leaf = 0
    stack = [(root, 0) ]
    
    while stack:
        root, curr_number = stack.pop()
        if root is not None:
            curr_number = (curr_number << 1) | root.val
            if root.left is None and root.right is None:
                root_to_leaf += curr_number
            else:
                stack.append((root.right, curr_number))
                stack.append((root.left, curr_number))
                    
    return root_to_leaf
```


## [1038. Binary Search Tree to Greater Sum Tree](https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Tree |

```python3
def convert_BST(root):
    total = 0
    def convert(root):
        if root is not None:
            convert_BST(root.right)
            total += root.val
            root.val = total
            self.convertBST(root.left)
        return root
    return convert(root)
```

## [1065. Index Pairs of a String](https://leetcode.com/problems/index-pairs-of-a-string/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | String |

```python3
import functools
def index_pairs(text, words):
    Trie = lambda: collections.defaultdict(Trie)
    trie = Trie()        
    for word in words:
        functools.reduce(dict.__getitem__, word, trie)["END"] = word
        
    ans = []
    for i in range(len(text)):
        T = trie
        for j in range(i, len(text)):
            if text[j] in T: T = T[text[j]]
            else: break
            if "END" in T: ans.append([i, j])
    return ans
```


## [1104. Path In Zigzag Labelled Binary Tree](https://leetcode.com/problems/path-in-zigzag-labelled-binary-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(logN) | O(logN) | Tree |

```python3
def path_in_zig_zag_tree(label):
    res = []
    node_count = 1
    level = 1
    while label >= node_count * 2:
        node_count *= 2
        level += 1
    while label != 0:
        res.append(label)
        level_max = 2**(level) - 1
        level_min = 2**(level-1)
        label = int((level_max + level_min - label)/2)
        level -= 1
    return res[::-1]
```

## [1120. Maximum Average Subtree](https://leetcode.com/problems/maximum-average-subtree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(H) | Tree |

```python3
def maximum_average_subtree(root):
    res = 0
    def dfs(root):
        if not root: return [0, 0.0]
        n1, s1 = dfs(root.left)
        n2, s2 = dfs(root.right)
        n = n1 + n2 + 1
        s = s1 + s2 + root.val
        res = max(res, s / n)
        return [n, s]
    dfs(root)
    return res
```


## [1221. Split a String in Balanced Strings](https://leetcode.com/problems/split-a-string-in-balanced-strings/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Stack |

```python3
def balanced_string_split(s):
    res = count = 0         
    for c in s:
        count += 1 if c == 'L' else -1            
        if count == 0:
            res += 1
    return res
```


## 1275. Find Winner on Tac Tac Toe Game

Given an order of moves, determine the winner of a tic tac toe game.

```python3
def tic_tac_toe(moves: List[List[int]]):
    # 0 1 2
    # 3 4 5
    # 6 7 8
    tests = [ [0,3,6], [1,4,7], [2,5,8], [0,1,2], [3,4,5], [6,7,8], [0,4,8], [2,4,6] ]
    matrix = [0] * 9
    player = 'A'
    
    for x,y in moves:
        matrix[x+y*3] = player
        for a,b,c in tests:
            if matrix[a] == matrix[b] == matrix[c] and matrix[a] != 0:
                return matrix[a]
        player = 'B' if player == 'A' else 'A'
    
    if len(moves) == 9:
        return "Draw"
    return "Pending"
```



## [1266. Minimum Time Visiting All Points](https://leetcode.com/problems/minimum-time-visiting-all-points/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | Array, Math |

```python3
def min_time_to_visit_all_points(points):
    time = 0
    for i in range(1, len(points)):
        prev, cur = points[i - 1 : i + 1]
        time += max(map(abs, (prev[0] - cur[0], prev[1] - cur[1])))
    return time
```

## [1407. Top Travellers](https://leetcode.com/problems/top-travellers/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(~) | O(~) | SQL |

```sql
SELECT U.Name, IFNULL(SUM(R.Distance), 0) AS Travelled_Distance
FROM Users U
LEFT JOIN Rides R
ON U.Id = R.User_Id
GROUP BY R.User_Id
ORDER BY Travelled_Distance DESC, U.Name ASC
```


## [1431. Kids With the Greatest Number of Candies](https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Array, Math |

```python3
def kids_with_candies(candies, extraCandies):
	high_enough = max(candies) - extraCandies
	return [i >= high_enough for i in candies]
```

## [1469. Find All The Lonely Nodes](https://leetcode.com/problems/find-all-the-lonely-nodes/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Tree |

```python3
def get_lonely_nodes(root):
    lonely = []

    def dfs(node):
        if node.left and not node.right:
            lonely.append(node.left.val)
        if node.right and not node.left:
            lonely.append(node.right.val)
        if node.left:
            dfs(node.left)
        if node.right:
            dfs(node.right)

    dfs(root)
    return lonely
```


## 1470. [Shuffle the Array](https://leetcode.com/problems/shuffle-the-array/)

```python3
def shuffle(nums, n):
    shuffled_array = []
    for i in range(n):
        shuffled_array += [nums[i]]
        shuffled_array += [nums[i+n]]
    return shuffled_array
```

Or use zip.

```
def shuffle(nums, n):
    return list(sum(zip(nums[:n],nums[n:]), ()))
```


## 1480. [Running Sum of 1D Array](https://leetcode.com/problems/running-sum-of-1d-array/)

The simple solution uses itertools. The complex uses DP.

```python3
from itertools import accumulate

def running_sum(nums):
    return accumulate(nums)
    
```

```python3
def running_sum(nums):
    i = 1
    while i < len(nums):
        nums[i] += nums[i-1]
        i += 1
    return nums
```




## [1486. XOR Operation in an Array](https://leetcode.com/problems/xor-operation-in-an-array/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | Array |

```python3
def XOR_operation(n, start):
    op = 0
    nums = [start + n * 2 for n in range(n)]
    for n in nums:
        op = op ^ n
    return op 
```


## [1512. Number of Good Pairs](https://leetcode.com/problems/number-of-good-pairs/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Array, Math |

```python3
def num_identical_pairs(A):
    return sum(k * (k - 1) / 2 for k in collections.Counter(A).values())
```

## [1534. Count Good Triplets](https://leetcode.com/problems/count-good-triplets/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N<sup>3</sup>) | O(1) | Array |

```python3
def count_good_triplets(A, a, b, c):
    count = 0
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            for k in range(j+1, len(A)):
                if abs(A[i] - A[j]) <= a and abs(A[j] - A[k]) <= b and abs(A[i] - A[k]) <= c: count += 1
    return count
```