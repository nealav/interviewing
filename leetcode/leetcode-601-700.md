# LeetCode 601 - 700

## 617. [Merge Two Binary Trees](https://leetcode.com/problems/merge-two-binary-trees/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n+m) | O(n+m) | Binary Search Tree, Recursion |

```python3
def merge_trees(t1, t2):
    if t1 and t2:
        root = TreeNode(t1.val + t2.val)
        root.left = merge_trees(t1.left, t2.left)
        root.right = merge_trees(t1.right, t2.right)
        return root
    else:
        return t1 or t2
```

## 647. [Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n^2) | O(1) | String |

```python3
def count_substrings(s):
    length = len(s)
    result = 0
    for i in range(2*length - 1):
        left = i//2
        right = (i+1)//2
        while left >= 0 and right < length and s[left] == s[right]:
            result += 1
            left -= 1
            right += 1
    return result
```

## 680. [Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(n) | String |

```python3
def validPalindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            one, two = s[left:right], s[left + 1:right + 1]
            return one == one[::-1] or two == two[::-1]
        left, right = left + 1, right - 1
    return True
```

## 692. Top K Frequent Words

Given list of words, return k most frequent.

```
def top_K_frequent(words, k):
    freqs = collections.Counter(words)
    return freqs.most_common(k)
```

This is O(n) space, and O(nlogn) time. It can be improved to O(nlogk) time by using a heap.

```
def top_K_frequent(words, k):
    freqs = collections.Counter(words)
    return heapq.nsmallest(k, freqs, key=lambda word:(~Freqs[word], word))
```