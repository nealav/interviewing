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
