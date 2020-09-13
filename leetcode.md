# Leetcode

## 1. [Two Sum](https://leetcode.com/problems/two-sum)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(n) | Hash Table, Two Pointer |

```python3
def two_sum(nums, target):
    dict = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement not in dict:
            dict[num] = i
        else:
            return [dict[complement], i]
```

## 5. [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n<sup>2</sup>) | O(1) | String, [Manacher's Algorithm](https://en.wikipedia.org/wiki/Longest_palindromic_substring#Manacher's_algorithm) |

```python3
def longest_palindrome(s):
    res = ""
    for i in range(len(s)):
        res = max(self.helper(s, i, i), self.helper(s, i, i+1), res, key=len)
    return res

def helper(s, l, r):
    while 0 <= l and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
    return s[l+1:r]
```

## 11. [Container With Most Water](https://leetcode.com/problems/container-with-most-water/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(1) | Two Pointer |

```python3
def max_area(height):
    i, j = 0, len(height) - 1
    water = 0
    while i < j:
        water = max(water, (j - i) * min(height[i], height[j]))
        if height[i] < height[j]:
            i += 1
        else:
            j -= 1
    return water
```

## 15. [3 Sum](https://leetcode.com/problems/3sum/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n<sup>2</sup>) | O(1) | Sort, Two Pointer |

```python3
def three_sum(nums):
    res = []
    nums.sort()
    for i in xrange(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        l, r = i+1, len(nums)-1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l +=1 
            elif s > 0:
                r -= 1
            else:
                res.append((nums[i], nums[l], nums[r]))
                while l < r and nums[l] == nums[l+1]:
                    l += 1
                while l < r and nums[r] == nums[r-1]:
                    r -= 1
                l += 1; r -= 1
    return res
```

## 17. [Letter Combinations of Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(4<sup>n</sup>) | O(4<sup>n</sup>) | Graph, BFS |

```python3
def letter_combinations(digits):
    interpret_digit = {
        '1': '',
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz',
        '0': ' '}
    all_combinations = [''] if digits else []
    for digit in digits:
        current_combinations = list()
        for letter in interpret_digit[digit]:
            for combination in all_combinations:
                current_combinations.append(combination + letter)
        all_combinations = current_combinations
    return all_combinations
```

## 18. [4 Sum](https://leetcode.com/problems/4sum/)

* N Sum

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n<sup>k</sup>) | O(n) | Recursion, Sort |

```python3
def four_sum(nums, target):
    def n_sum(nums, target, n, result, results):
        if len(nums) < n or n < 2 or target < nums[0] * n or target > nums[-1] * n:
            return
        if n == 2:
            l, r =  0, len(nums) - 1
            while l < r:
                s = nums[l] + nums[r]
                if s == target:
                    results.append(result + [nums[l], nums[r]])
                    l += 1
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                elif s < target:
                    l += 1
                else:
                    r -= 1
        else:
            for i in range(len(nums) - n + 1):
                if i == 0 or (i > 0 and nums[i - 1] != nums[i]):
                    n_sum(nums[i+1:], target - nums[i], n - 1, result + [nums[i]], results)

    results = []
    n_sum(sorted(nums), target, 4, [], results)
    return results
```

## 20. [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(n) | Stack |

```python3
def valid_parentheses(s):
    stack = ['N']
    m = {')' : '(', ']' : '[', '}' : '{'}
    for i in s:
        if i in m.keys():
            if stack.pop() != m[i]:
                return False
        else:
            stack.append(i)
    return len(stack) == 1
```

## 21. [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n+m) | O(1) | Linked List |

```python3
def merge_two_sorted_lists(l1, l2):
    dummy = cur = ListNode(0)
    while l1 and l2:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next
```

## 22. [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(2<sup>n</sup>) | O(n) | Recursion |

```python3
def generate_parentheses(n):
    def generate(p, left, right, parens=[]):
        if left:
            generate(p + '(', left - 1, right)
        if right > left:
            generate(p + ')', left, right - 1)
        if not right:
            parens += p,
        return parens
    return generate('', n, n)
```

## 23. Merge K Sorted Lists

Using a heap you can add an element from every sorted list into the heap along with it's list and pop them off in order. This can organize the sorted lists in O(m*n*log(n)) where m is the number of lists and n is the total number of ListNodes.

```python3
import heapq

def mergeKLists(self, lists: List[ListNode]) -> ListNode:
    curr = head = ListNode(0)
    queue = []
    count = 0
    for l in lists:
        if l is not None:
            count += 1
            heapq.heappush(queue, (l.val, count, l))
    while len(queue) > 0:
        _, _, curr.next = heapq.heappop(queue)
        curr = curr.next
        if curr.next is not None:
            count += 1
            heapq.heappush(queue, (curr.next.val, count, curr.next))
    return head.next   
```

## 26. [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(1) | Array, Two Pointer |

```python3
def remove_duplicates(nums):
    if not nums:
        return 0
    new_tail = 0
    for i in range(1, len(A)):
        if A[i] != A[new_tail]:
            new_tail += 1
            A[new_tail] = A[i]
    return new_tail + 1
```

Maintain a tail to add numbers and skip over duplicates until you reach the next new number every time.


## 33. [Search in a Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)

Modified binary search conditions based on low/mid/high target ranges.

```python3
def search(nums, target):
    if not nums:
        return -1

    low, high = 0, len(nums) - 1

    while low <= high:
        mid = (low + high) / 2
        if target == nums[mid]:
            return mid

        if nums[low] <= nums[mid]:
            if nums[low] <= target <= nums[mid]:
                high = mid - 1
            else:
                low = mid + 1
        else:
            if nums[mid] <= target <= nums[high]:
                low = mid + 1
            else:
                high = mid - 1

    return -1
```

## 45. [Jump Game II](https://leetcode.com/problems/jump-game-ii/description/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n) | O(1) | Dynamic Programming |

```python3
def jump(nums):
    jumps, edge, max_edge = 0, 0, 0
    for i in range(len(nums)):
        if i > edge:
            edge = max_edge
            jumps += 1
        maxEdge = max(max+edge, i + nums[i])
    return jumps
```

## 46. [Permutations](https://leetcode.com/problems/permutations/)

```python3
def permute(self, nums):
    perms = [[]]   
    for n in nums:
        new_perms = []
        for perm in perms:
            for i in range(len(perm)+1):   
                new_perms.append(perm[:i] + [n] + perm[i:])
        perms = new_perms
    return perms
```

## 49. [Group Anagrams](https://leetcode.com/problems/group-anagrams/)

```python3
def group_anagrams(words):
    groups = {}
    for word in words:
        s_word = sorted(word)
        groups[s_word] = d.get(s_word, []) + [word]
    return groups.values()
```

## 53. [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)

The brute-force algorithm loops through all possible subarrays in O(n^2). The ideal algorithm keeps a sliding window and resets the pointers if the sum becomes 0 or negative. The optimized algorithm uses DP and only keeps track of the previous sum as we do a single pass through the array.

```python3
def max_subarray(self, nums):
    for i in range(1, len(nums)):
        nums[i] = max(nums[i], nums[i-1] + nums[i])
    return max(nums)
```

## 55. [Jump Game](https://leetcode.com/problems/jump-game/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n) | O(1) | Dynamic Programming |

```python3
def can_jump(nums):
    m = 0
    for i, n in enumerate(nums):
        if i > m:
            return False
        m = max(m, i+n)
    return True
```

## 56. [Merge Intervals](https://leetcode.com/problems/merge-intervals/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(nlogn) | O(n) | Sorting |

```python3
def merge(intervals):
    if len(intervals) == 0: 
        return []
    
    intervals = sorted(intervals, key = lambda x: x.start)
    res = [intervals[0]]
    for interval in intervals[1:]:
        if interval.start <= res[-1].end:
            res[-1].end = max(interval.end, res[-1].end)
        else:
            res.append(interval)
    return res
```

## 70. Climbing Stairs

There are two ways to reach the ith step, (1) by taking 2 steps from step ith-2 (2) by taking 1 step from step ith-1. Therefore by adding them we get the possibilities for reaching the ith step.

```python3
def climingStairs(n):
    stairs = [0 for _ in range(n)]
    stairs[0] = 1
    stairs[1] = 2
    for i in range(2, n):
        stairs[i] = stairs[i-1] + stairs[i-2]
    return stairs[n-1]
```

## 72. [Edit Distance](https://leetcode.com/problems/edit-distance/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(mn) | O(mn) | Dynamic Programming, [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance) |

```python3
def min_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    table = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        table[i][0] = i
    for j in range(n + 1):
        table[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                table[i][j] = table[i - 1][j - 1]
            else:
                table[i][j] = 1 + min(table[i - 1][j], table[i][j - 1], table[i - 1][j - 1])
    return table[-1][-1]
```

## 76. [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

Maintain a sliding window with desireable substrings and two-pointers.

```python3
def min_window(s, t):
    need = collections.Counter(t)
    missing = len(t)
    start, end = 0, 0
    i = 0
    for j, char in enumerate(s, 1):
        if need[char] > 0:
            missing -= 1
        need[char] -= 1
        if missing == 0:
            while i < j and need[s[i]] < 0:
                need[s[i]] += 1
                i += 1
            need[s[i]] += 1
            missing += 1
            if end == 0 or j-i < end-start:
                start, end = i, j
            i += 1
    return s[start:end]
```

## 79. Word Search

```python3
def exist(board, word):

    def dfs(board, i, j, word):
        if len(word) == 0:
            return True
        if i<0 or i>=len(board) or j<0 or j>=len(board[0]) or word[0]!=board[i][j]:
            return False
        tmp = board[i][j]
        board[i][j] = "#"
        res = self.dfs(board, i+1, j, word[1:]) \
                or self.dfs(board, i-1, j, word[1:]) \
                or self.dfs(board, i, j+1, word[1:]) \
                or self.dfs(board, i, j-1, word[1:]) \
        board[i][j] = tmp
        return res


    if not board:
        return False
    for i in range(len(board)):
        for j in range(len(board[0])):
            if self.dfs(board, i, j, word):
                return True
    return False
```

## 82. Remove Duplicates from Sorted Linked List II

Delete duplicates leaving only distinct numbers.

```python3
def deleteDuplicates(self, head):
    dummy = pre = ListNode(0)
    dummy.next = head
    while head and head.next:
        if head.val == head.next.val:
            while head and head.next and head.val == head.next.val:
                head = head.next
            head = head.next
            pre.next = head
        else:
            pre = pre.next
            head = head.next
    return dummy.next
```


## 83. Remove Duplicates from Sorted Linked List

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n) | O(1) | Linked List, Two Pointer |

```python3
def delete_duplicates(head):
    cur = head
    while cur:
        while cur.next and cur.next.val == cur.val:
            cur.next = cur.next.next
        cur = cur.next
    return head
```

## 91. [Decode Ways]

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n) | O(n) | Dynamic Programming |

```python3
def num_decodings(s):
    if not s or s[0] == '0':
        return 0
        
    dp = [0 for x in range(len(s) + 1)] 
    dp[0:2] = [1, 1]
    for i in range(2, len(s) + 1): 
        if 0 < int(s[i-1:i]):
            dp[i] = dp[i - 1]
        if 10 <= int(s[i-2:i]) <= 26:
            dp[i] += dp[i - 2]
    return dp[-1]
```

## 100. [Same Tree](https://leetcode.com/problems/same-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n) | O(n) | Binary Search Tree |

```python3
def is_same_tree(root):
    if not p and not q:
        return True
    if not q or not p:
        return False
    if p.val != q.val:
        return False
    return is_same_tree(p.right, q.right) and \
           is_same_tree(p.left, q.left)    
```

## 101. [Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)

```python3
def is_symmetric(root):
    def mirror(left, right):
        if left and right and left.val == right.val: 
            return mirror(left.left, right.right) and mirror(left.right, right.left)
        return left == right
    return not root or mirror(root.left, root.right)
```

## 104. [Maximum Depth of a Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n) | O(n) | Binary Search Tree, Recursion |

```python3
def max_depth(root):
    return 1 + max(max_depth(root.left), max_depth(root.right)) if root else 0
```

## 105. [Construct Binary Tree From Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n) | O(n) | Binary Search Tree, Recursion |

```python3
def build_tree(preorder, inorder):
    if inorder:
        ind = inorder.index(preorder.pop(0))
        root = TreeNode(inorder[ind])
        root.left = build_tree(preorder, inorder[0:ind])
        root.right = build_tree(preorder, inorder[ind+1:])
        return root
```

## 108. Convert Sorted Array to BST

Convert sorted array to height-balanced binary tree.

```python3
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def array_to_BST(nums):
    if not nums: return None

    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = array_to_BST(nums[:mid])
    root.right = array_to_BST(nums[mid:])

    return root
```

Recursively build the tree top-down. Time complexity O(n), space complexity O(n).

## 121. [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

Keep track of the minimum price and maximum profit found: if a new minimum price is found then calculate the running profits using that. The current minimum price will always yield the maximum profit for every stock price change after it, and if a new minimum is found the same rule applies.

```python3
def max_profit(prices):
    max_profit, min_price = 0, float('inf')
    for price in prices:
        min_price = min(min_price, price)
        profit = price - min_price
        max_profit = max(max_profit, profit)
    return max_profit
```

## 124. Binary Tree Maximum Path Sum

```python3
def maxPathSum(root: TreeNode) -> int:
    max_path = float("-inf")

    def get_max_gain(node):
        nonlocal max_path
        if node is None:
            return 0
            
        gain_on_left = max(get_max_gain(node.left), 0)
        gain_on_right = max(get_max_gain(node.right), 0)
            
        current_max_path = node.val + gain_on_left + gain_on_right
        max_path = max(max_path, current_max_path)
            
        return node.val + max(gain_on_left, gain_on_right)
        
        
    get_max_gain(root)
    return max_path	
```

## 125. [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)

Reverse and equality check, or use two-pointers and contract inward from the sides.

```python3
def is_palindrome(self, s):
    s = ''.join(c for c in s if c.isalnum()).lower()
    return s == s[::-1]
```

## 128. Longest Consecutive Sequence

Given an unsorted array of ints, find the longest consecutive sequence of elements. Must be O(n).

```
def longestConsecutive(self, nums):
    nums = set(nums)
    best = 0
    for x in nums:
        if x - 1 not in nums:
            y = x + 1
            while y in nums:
                y += 1
            best = max(best, y - x)
    return best
```

Convert the array to a set and walk each streak individually and keep track of it's length as it runs. Reset if a number is not within it's streak.

## 136. [Single Number](https://leetcode.com/problems/single-number/)

```python3
def single_number(nums):
    n = 0
    for num in nums:
        n ^= num
    return n
```

## 141. [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

```python3
def floyd_cycle_detection(head): 
    slow = head 
    fast = head.next
    while slow and fast and fast.next: 
        slow = slow.next
        fast = fast.next.next
        if slow == fast: 
            return True
    return False
```

## 146. LRU Cache

```python3
class LRUCache:
    def __init__(self, MSize):
        self.size = MSize
        self.cache = {}
        self.next, self.before = {}, {}
        self.head, self.tail = '#', '$'
        self.connect(self.head, self.tail)

    def connect(self, a, b):
        self.next[a], self.before[b] = b, a

    def delete(self, key):
        self.connect(self.before[key], self.next[key])
        del self.before[key], self.next[key], self.cache[key]

    def append(self, k, v):
        self.cache[k] = v
        self.connect(self.before[self.tail], k)
        self.connect(k, self.tail)
        if len(self.cache) > self.size:
            self.delete(self.next[self.head])

    def get(self, key):
        if key not in self.cache: return -1
        val = self.cache[key]
        self.delete(key)
        self.append(key, val)
        return val

    def put(self, key, value):
        if key in self.cache: self.delete(key)
        self.append(key, value)
```

A shorter version using OrderedDict

```
from collections import OrderedDict
class LRUCache:
    def __init__(self, Capacity):
        self.size = Capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache: return -1
        val = self.cache[key]
        self.cache.move_to_end(key)
        return val

    def put(self, key, val):
        if key in self.cache: del self.cache[key]
        self.cache[key] = val
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)
```

## 152. [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)

```python3
    def max_product(self, A):
        B = A[::-1]
        for i in range(1, len(A)):
            A[i] *= A[i - 1] or 1
            B[i] *= B[i - 1] or 1
        return max(A + B)
```

First, if there's no zero in the array, then the subarray with maximum product must start with the first element or end with the last element. And therefore, the maximum product must be some prefix product or suffix product. So in this solution, we compute the prefix product A and suffix product B, and simply return the maximum of A and B.

Say, we have a subarray A[i : j](i != 0, j != n) and the product of elements inside is P. Take P > 0 for example: if A[i] > 0 or A[j] > 0, then obviously, we should extend this subarray to include A[i] or A[j]; if both A[i] and A[j] are negative, then extending this subarray to include both A[i] and A[j] to get a larger product. Repeating this procedure and eventually we will reach the beginning or the end of A.

What if there are zeroes in the array? Well, we can split the array into several smaller ones. That's to say, when the prefix product is 0, we start over and compute prefix product from the current element instead. And this is exactly what A[i] *= (A[i - 1]) or 1 does.


## 153. [Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

Modified binary search (Binary Select?) in the left and right trees in O(logn).

```python3
def find_min(nums):
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid

    return nums[left]
```

The main idea is to converge the left and right bounds to the minimum value regardless of pivot. If there is a larger number at the mid we know that the array is rotated to have the minimum to the right of it, and if it is smaller than the rightest element we know we are cycling through the rotated part but we don't know that the number is the minimum. We need to wait for the loop to converge to right which will be the minimum.

## 155. [Min Stack](https://leetcode.com/problems/min-stack/)

```python3
class MinStack(object):

    def __init__(self):
        self.stack = []
        
    def push(self, x):
        self.stack.append((x, min(self.get_min(), x))) 
           
    def pop(self):
        self.stack.pop()

    def top(self):
        if self.stack:
            return self.stack[-1][0]
        
    def get_min(self):
        if self.stack:
            return self.stack[-1][1]
        return sys.maxint          
```

## 160. [Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/)

```python3
def get_intersection_node(headA, headB):
    if headA and headB:
        A, B = headA, headB
        while A != B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A
```

## 169. [Majority Element](https://leetcode.com/problems/majority-element/)

```python3
def majority_element(nums):
    count, cand = 0, 0
    for num in nums:
        if num == cand:
            count += 1
        elif count == 0:
            cand, count = num, 1
        else:
            count -= 1
    return cand
```

## 189. Rotate Array

Rotate array by k steps.

```
def rotate(nums, k):
    n = len(nums)
    k = k % n
    nums[:] = nums[n-k:] + nums[:n-k]   # Can also use negative indices.
```

## 190. [Reverse Bits](https://leetcode.com/problems/reverse-bits/)

Pop off least siginficant bit and right shift it.

```python3
def reverse_bits(n):
    res = 0
    for i in range(32):
        res = (res << 1) + (n & 1)
        n >>= 1
```

## 191. [Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/)

Pop the least significant bit and count until 0.

```python3
def hamming_weight(n):
    c = 0
    while n:
        n &= n - 1
        c += 1
    return c
```

## 198. [House Robber](https://leetcode.com/problems/house-robber/)

```python3
def rob(nums):
    # Base Case: nums[0] = nums[0]
    # nums[1] = max(nums[0], nums[1])
    # nums[k] = max(k + nums[k-2], nums[k-1])
    
    prev = curr = 0
    for num in nums:
      temp = prev
      prev = curr
      curr = max(num + temp, prev)
    return curr
```

## 200. [Number Of Islands](https://leetcode.com/problems/number-of-islands/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n) | O(n) | Graph, BFS, DFS |

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

## 212. [Word Search II](https://leetcode.com/problems/word-search-ii/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(M(4\*3^(L-1)) | O(N) | Trie, Depth First Search |

```
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


## 217. [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n) | O(n) | Array, Set, Hash Table |

```python3
def contains_duplicate(nums):
    return len(nums) != len(set(nums))
```

## 226. Invert Binary Tree

https://leetcode.com/problems/invert-binary-tree/solution/

```
def invertTree(self, root):
    if root is None:
        return None
    root.left, root.right = \
        self.invertTree(root.right), self.invertTree(root.left)
    return root
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

```python3
def valid_anagram(s, t):
    return len(s) == len(t) and sorted(s) == sorted(t)
```

## 268. [Missing Number](https://leetcode.com/problems/missing-number/)

XOR can be used to eliminate pairs of the index XOR number since there is a constant distribution. Binary search can be used in a sorted array as well.

```python3
def missing_number(nums):
    missing_number = len(nums)
    for i in range(0, len(nums) + 1):
        missing_number ^= i
        missing_number ^= nums[i]
    return missing_number
```

Discrete math sum equation.

```python3
def missingNumber(nums):
    return (len(nums)*(len(nums) + 1)/2) - sum(nums)
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

## 322. [Coin Change](https://leetcode.com/problems/coin-change/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n\*k) | O(k) | Dynamic Programming |

```python3
def coin_change(coins, amount):
        dp = [0] + [float('inf') for i in range(amount)]
        for i in range(1, amount + 1):
            for coin in coins:
                if i - coin >= 0:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        if dp[-1] == float('inf'):
            return -1
        return dp[-1]
```

## 331. [Verify Preorder Serialization Of A Binary Tree](https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n) | O(1) | Binary Search Tree |

```python3
    def is_valid_serialization(preorder):
        p = preorder.split(',')
        slot = 1
        for node in p:
            if slot == 0:
                return False
            if node == '#':
                slot -= 1
            else:
                slot += 1
        return slot == 0
```


## 338. [Counting Bits](https://leetcode.com/problems/counting-bits/)

Bit manipulation through single level DP. A number that is even can be turned into an odd number through a right shift by 1 and a number that is odd has the name number of bits as the previous number + 1.

```python3
def count_bits(n):
    res = []
    res.append(0)
    for i in range(1, num + 1):
        if (i & 1) == 0: #odd
            res.append(res[i >> 1])
        else:
            res.append(res[i - 1] + 1)
    return res
```

## 342. Is Power Of Four

Powers of four in binary have a few characteristics: (1) greater than 0 (2) only have one 1 bit in their binary notation (3)) the 1 bit should be at an odd location.

```
def isPowerOfFour(n):
    return (
        n > 0 && #1
        !(n & (n - 1)) && #clear the 1 bit
        !(n & 0x55555555) #check the odd positions
    )

```

## 347. Top K Frequent Elements

Given a non-empty array of integers, return the k most frequent elements.

https://leetcode.com/problems/top-k-frequent-elements/solution/

Using bucket sort we can maintain an array where the frequency is the index. Make a hashmap with the frequencies and add them to an array. This will take O(N). Using the bucket array we can return the top K values where K will always be less than N because the frequency can be no larger than the number of integers.


## 371. [Sum of Two Integers Without '+' or '-'](https://leetcode.com/problems/sum-of-two-integers/)

Add the two numbers with the XOR operator. However in Binary this will take 1+1 and instead of 0 with a carry it will just be 0. We need to account for the carry with the & operator. The carry gets added to the next number so we keep adding until it becomes 0.

```python3
def get_sum(a, b):
    c = 0
    while b != 0:
        c = a & b
        a = a ^ b
        b = c << 1 #reassign carry
    return a
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

## 509. Fibonacci Number

Calculate the Nth Fibonacci number.

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

## 621. [Task Scheduler](https://leetcode.com/problems/task-scheduler/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O() | O() | S |




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

