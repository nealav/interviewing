# LeetCode 601 - 700

## [606. Construct String from Binary Tree](https://leetcode.com/problems/construct-string-from-binary-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(n) | Tree |

```python3
def tree2str(t):
    if not t: return ''
    left = '({})'.format(tree2str(t.left)) if (t.left or t.right) else ''
    right = '({})'.format(tree2str(t.right)) if t.right else ''
    return '{}{}{}'.format(t.val, left, right)
```

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

## [637. Average of Levels in Binary Tree](https://leetcode.com/problems/average-of-levels-in-binary-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Tree |

```python3
def average_of_levels(root):
    averages = []
    level = [root]
    while level:
        averages.append(sum(node.val for node in level) / len(level))
        level = [kid for node in level for kid in (node.left, node.right) if kid]
    return averages
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


## [653. Two Sum IV](https://leetcode.com/problems/two-sum-iv-input-is-a-bst/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Tree |

```python3
def find_target(root, k):
    if not root: return False
    queue, s = [root], set()
    for i in queue:
        if k - i.val in s: return True
        s.add(i.val)
        if i.left: queue.append(i.left)
        if i.right: queue.append(i.right)
    return False
```


## [680. Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/)

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


## [687. Longest Univalue Path](https://leetcode.com/problems/longest-univalue-path/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Tree |

```python3
def longest_univalue_path(root):
    longest = [0]

    def traverse(node):
        if not node:
            return 0
        left_len, right_len = traverse(node.left), traverse(node.right)
        left = (left_len + 1) if node.left and node.left.val == node.val else 0
        right = (right_len + 1) if node.right and node.right.val == node.val else 0
        longest[0] = max(longest[0], left + right)
        return max(left, right)

    traverse(root)
    return longest[0]
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