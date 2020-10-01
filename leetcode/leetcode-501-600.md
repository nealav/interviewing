# LeetCode 501 - 600

## 509. Fibonacci Number

```
def fib(n):
    if not n: return 0
    memo = [0, 1]
    for _ in range(2, n+1):
        memo = [memo[1], memo[0] + memo[1]]
    return memo[1]
```

The iterative solution is described above. It can use variables as well with swapping assignment on the right hand side. Mathematically Fibonacci numbers can be determined in O(1).

```
def fib(n):
    golden_ratio = (1 + 5 ** 0.5) / 2
    return int((golden_ratio ** (n+1)) / 5 ** 0.5)
```

This can be derived with Linear Algebra.


## [530. Minimum Absolute Difference in BST](https://leetcode.com/problems/minimum-absolute-difference-in-bst)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Tree |

```python3
def get_minimum_difference(root):
    vals = []

    def dfs(node):
        if node.left: dfs(node.left)
        vals.append(node.val)
        if node.right: dfs(node.right)
    
    dfs(root)
    return min(b - a for a, b in zip(L, L[1:]))
```

## [538. Convert BST to Greater Tree](https://leetcode.com/problems/convert-bst-to-greater-tree/)

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


## 543. [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)

```python3
def diameter_binary_tree(root):
    diameter = 0

    def depth(p):
        if not p: 
            return 0
        left, right = depth(p.left), depth(p.right)
        diameter = max(diameter, left+right)
        return 1 + max(left, right)

    depth(root)
    return diameter
```

## [563. Binary Tree Tilt](https://leetcode.com/problems/binary-tree-tilt/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Tree |

```python3
def find_tilt(root):
    tilt = 0

    def node_tilt(node):
        if not node: return 0
        left, right = node_tilt(node.left), node_tilt(node.right)
        tilt += abs(left - right)
        return node.val + left + right

    node_tilt(root)
    return tilt
```

## [572. Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(|S| * |T|) | O(|T|) | Tree |

```python3
def is_sub_tree(s, t):
    def is_match(s, t):
        if not(s and t):
            return s is t
        return (s.val == t.val and
                self.is_match(s.left, t.left) and
                self.is_match(s.right, t.right))

    if is_match(s, t): return True
    if not s: return False
    return is_sub_tree(s.left, t) or is_sub_tree(s.right, t)
```

## 581. [Shortest Unsorted Continuous Subarray](https://leetcode.com/problems/shortest-unsorted-continuous-subarray/)

```python3
def find_unsorted_subarray(nums):
    if len(nums) < 2: 
        return 0

    l, r = 0, len(nums) - 1
    while l < len(nums) - 1 and nums[l] <= nums[l + 1]:
        l += 1
    while r > 0 and nums[r] >= nums[r -1]:
        r -= 1
    if l > r:
        return 0

    temp = nums[l:r+1]
    tempMin = min(temp)
    tempMax = max(temp)

    while l > 0 and tempMin < nums[l-1]:
        l -= 1
    while r < len(nums) - 1 and tempMax > nums[r+1]:
        r += 1

    return r - l + 1
```
