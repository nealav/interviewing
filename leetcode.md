# Leetcode

## 1. Two Sum

Given an array of ints, return indices of two numbers such that they add to the target number.

Brute Force - 2 nested loops.
Ideal - One-pass Hash Table, iterate the array while inserting elements into the table. The key being the number and it’s value the index.

```
def twoSum(self, nums, target):
    dict = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement not in dict:
            dict[num] = i
        else:
            return [dict[complement], i]
```


## 11. Container With Most Water

Brute Force - O(n^2). Find all containers and their volume.
Ideal - Start with the widest container, the container with the first and last lines. We cannot increase the width of this container to increase the volume therefore we need to increase the height. Removing the larger height candidate will not increase the height so we remove and move the smaller height candidate (either the left or right most).


```
def maxArea(self, height):
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

## 15. 3-Sum

Brute force - O(n^3). Loop through each triplet.
Ideal - Sort the array, then, for each element we neeed to find the matching 2 other targets that complete the triplet. Have a left and right pointer to the ends of the rest of the array. We know that if the sum of the two-target is higher than necessary, we move the right pointer down and vice versa the left pointer up. O(n^2)

```
def threeSum(self, nums):
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

## 17. Letter Combinations of Phone Number

Given a string of digits, return all possible letter combinations.

```
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

## 21. [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)

Loop through the lists iteratively, verifying their values and incrementing them forwards respectively. Weave them together.

```
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

## 23. Merge K Sorted Lists

Using a heap you can add an element from every sorted list into the heap along with it's list and pop them off in order. This can organize the sorted lists in O(m*n*log(n)) where m is the number of lists and n is the total number of ListNodes.

```
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

## 26. Remove Duplicates from Sorted Array

In-place.

```
def removeDuplicates(self, A):
    if not A:
        return 0

    newTail = 0
    for i in range(1, len(A)):
        if A[i] != A[newTail]:
            newTail += 1
            A[newTail] = A[i]
    return newTail + 1
```

Maintain a tail to add numbers and skip over duplicates until you reach the next new number every time.


## 33. Search in a Rotated Sorted Array

Given a sorted array of integers rotated around a pivot and a target num, find the index of the target num.

Ideal - Modify binary search conditions based on low/mid/high target ranges.

```
def search(self, nums, target):
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

## 49. Group Anagrams

Given an array of strings, group anagrams.

```
def group_anagrams(words):
    groups = {}
    for word in words:
        s_word = sorted(word)
        groups[s_word] = d.get(s_word, []) + [word]
    return groups.values()
```

## 53. Maximum Subarray

Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Brute force - loop through all the subarrays starting at an index and calc the sum, keep track of max. O(n^2).
Ideal - Keep a sliding window. If the subarray sum is positive keep sliding, if it ever goes to 0 or a negative number, restart after the subarray window. Keep track of max sum. O(n).
Optimized - Dynamic Programming, we only need to keep track of the previous sum. (1) If the subarray sum is positive, it can make the next number bigger so we store the sum in the index (2) if it is 0 or negative then we lose it and restart the sum count. This is a one pass.

```
def maxSubArray(self, nums):
    for i in range(1, len(nums)):
        nums[i] = max(nums[i], nums[i-1] + nums[i])
    return max(nums)
```

## 56. Merge Intervals

Given a collection of intervals, merge overlapping entries.

```
def merge(intervals):
    if len(intervals) == 0: return []
    intervals = sorted(intervals, key = lambda x: x.start)
    res = [intervals[0]]
    for n in intervals[1:]:
        if n.start <= res[-1].end: res[-1].end = max(n.end, res[-1].end)
        else: res.append(n)
    return res
```

Sort the list on starting points. Check if the new interval overlaps with the previous in the result list. If yes, update it to the end, otherwise append a new interval.

## 70. Climbing Stairs

There are two ways to reach the ith step, (1) by taking 2 steps from step ith-2 (2) by taking 1 step from step ith-1. Therefore by adding them we get the possibilities for reaching the ith step.

```
def climingStairs(n):
    stairs = [0 for _ in range(n)]
    stairs[0] = 1
    stairs[1] = 2
    for i in range(2, n):
        stairs[i] = stairs[i-1] + stairs[i-2]
    return stairs[n-1]
```

## 79. Word Search

Given a 2D board, find if the word exists in the grid.

```
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

```
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

```
def delete_duplicates(head):
    cur = head
    while cur:
        while cur.next and cur.next.val == cur.val:
            cur.next = cur.next.next
        cur = cur.next
    return head
```


## 100. Same Tree

https://leetcode.com/problems/same-tree/solution/

Recursively go down both trees and compare the nodes.

```
def isSameTree(self, root):
    if not p and not q:
        return True
    if not q or not p:
        return False
    if p.val != q.val:
        return False
    return self.isSameTree(p.right, q.right) and \
           self.isSameTree(p.left, q.left)    
```

## 104. Maximum Depth of a Binary Tree

https://leetcode.com/problems/maximum-depth-of-binary-tree/

Simply go down the tree recursively in both directions on every node. Every node will add the next node to the stack. O(n) as it traverses all nodes.

```
def maxDepth(self, root):
    return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right)) if root else 0
```

## 108. Convert Sorted Array to BST

Convert sorted array to height-balanced binary tree.

```
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


def array_to_BST(nums):
    if not nums: return None

    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = array_to_BST(nums[:mid])
    root.right = array_to_BST(nums[mid:])

    return root
```

Recursively build the tree top-down. Time complexity O(n), space complexity O(n).

## 121. Best Time to Buy and Sell Stock

Given an array of stock prices on given days, with only one transaction on a single day, determine best price to first buy and sell a stock and return the max profit.

Brute force - loop through n^2 and find the max difference between every pair.
Ideal - One-pass loop, keep track of the min_price and the max_profit found, if a new min_price is found calculate the max profit using that. It will work since the current min_price will always be the lowest point that yields the max_profit for every stock price after it, then if a new one is found the same rule applies.


```
def maxProfit(self, prices):
    max_profit, min_price = 0, float('inf')
    for price in prices:
        min_price = min(min_price, price)
        profit = price - min_price
        max_profit = max(max_profit, profit)
    return max_profit
```


## 124. Binary Tree Maximum Path Sum

Find the max path sum of a binary tree.

```
def maxPathSum(self, root: TreeNode) -> int:
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

128. Longest Consecutive Sequence.

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


## 146. LRU Cache

Design and implement a Least Recently Used cache. Support __get(key)__ and __put(key, val)__ in O(1).

```
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

The basic implementation of a cache relies on a data structure which uses an underlying hashtable.

## 152. Maximum Product Subarray

Given an integer array nums, find the contiguous subarray within an array (containing at least one number) which has the largest product.

```
    def maxProduct(self, A):
        B = A[::-1]
        for i in range(1, len(A)):
            A[i] *= A[i - 1] or 1
            B[i] *= B[i - 1] or 1
        return max(A + B)
```

First, if there's no zero in the array, then the subarray with maximum product must start with the first element or end with the last element. And therefore, the maximum product must be some prefix product or suffix product. So in this solution, we compute the prefix product A and suffix product B, and simply return the maximum of A and B.

Why? Here's the proof:

Say, we have a subarray A[i : j](i != 0, j != n) and the product of elements inside is P. Take P > 0 for example: if A[i] > 0 or A[j] > 0, then obviously, we should extend this subarray to include A[i] or A[j]; if both A[i] and A[j] are negative, then extending this subarray to include both A[i] and A[j] to get a larger product. Repeating this procedure and eventually we will reach the beginning or the end of A.

What if there are zeroes in the array? Well, we can split the array into several smaller ones. That's to say, when the prefix product is 0, we start over and compute prefix product from the current element instead. And this is exactly what A[i] *= (A[i - 1]) or 1 does.


## 153. Find Minimum in Rotated Sorted Array

Suppose an array of integers is sorted and rotated around a pivot. Find the minimum.

Brute Force - One-pass check for the minimum in O(n)
Ideal - Modified binary search (binary select?) in the left and right trees in O(logn)

```
def findMin(self, nums):
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


## 189. Rotate Array

Rotate array by k steps.

```
def rotate(nums, k):
    n = len(nums)
    k = k % n
    nums[:] = nums[n-k:] + nums[:n-k]   # Can also use negative indices.
```

## 190. Reverse Bits

Pop off least siginficant bit and right shift it.

```
def reverseBits(n):
    res = 0
    for i in range(32):
        res = (res << 1) + (n & 1)
        n >>= 1
```

## 191. Number of 1 Bits

Pop the least significant bit and count until 0.

```
def hammingWeight(n):
    c = 0
    while n:
        n &= n - 1
        c += 1
    return c
```


## 217. Contains Duplicate

Given array of integers, find if array has duplicates.

Brute force - check one int across the array (n^2) or sort the array and check if any repeats one-after (nlogn)
Ideal - throw ints into set until repeat or check set length inequality.

```
def containsDuplicate(self, nums):
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


## 238. Product of Array Except Self

Given an array of ints, return the product of the array except the element for each element as an array.

Brute Force - make an array and loop through for each element multiplying and skipping the element in question (n^2)
Ideal - Loop through once, making an array of the ‘right product’ to the element. Loop through again making an array of the ‘left product’ to the element. Multiple the left and right product arrays.
Ideal Optimized - Make the left product array, and then while looping through the rights keep a running product of the right products and multiply them by the running right product. [O(n) time]

```
def productExceptSelf(self, nums):
    left_product = 1
    n = len(nums)
    output = []
    for i in range(0, n):
        output.append(left_product)
        left_product = left_product * nums[i]
    right_product = 1
    for i in range(n-1,-1,-1): #backwards
        output[i] = output[i] * right_product
        right_product = right_product * nums[i]
    return output
```

## 242. Valid Anagram

```
def valid_anagram(s, t):
    return len(s) == len(t) and sorted(s) == sorted(t)
```

## 268. Missing Number

XOR can be used to eliminate pairs of the index XOR number since there is a constant distribution. Binary search can be used in a sorted array. Or the sum in O(n).

```
def missingNumber(nums):
    missing_number = len(nums)
    for i in range(0, len(nums)+1):
        missing_number ^= i
        missing_number ^= nums[i]
    return missing_number
```

```
def missingNumber(nums):
    return (len(nums)*(len(nums) + 1)/2) - sum(nums)
```

## 283. [Move Zeroes](https://leetcode.com/problems/move-zeroes/)

```
def move_zeroes(nums):
    zeroes = nums.count(0)
    nums[:] = [i for i in nums if i != 0]
    nums += [0] * zeroes
```

## 284. Peeking Iterator

```
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

## 295. Find the Median From Data Stream

https://leetcode.com/problems/find-median-from-data-stream/

Simple sorting will be an O(nlogn) solution. Sort every time. Maintaining two heaps - one max and one min heap where each is maintained with equal sizes within 2 is the key to this problem.

## 322. Coin Change

This problem can be broken down into subtracting a single coin from the current amount value and checking if the subtracted amount has a number of times associated with it for all the coins in the bank.


## 338. Counting Bits

Bit manipulation through single level DP. A number that is even can be turned into an odd number through a right shift by 1 and a number that is odd has the name number of bits as the previous number + 1.

```
def countBits(n):
    res = []
    res.append(0)
    for i in range(1, num+1):
        if (i&1)==0: #odd
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


## 371. Sum of Two Integers Without '+'

Add the two numbers with the XOR operator. However in Binary this will take 1+1 and instead of 0 with a carry it will just be 0. We need to account for the carry with the & operator. The carry gets added to the next number so we keep adding until it becomes 0.

```
def get_sum(a, b):
    c = 0
    while b != 0:
        c = a & b
        a = a ^ b
        b = c<<1 #reassign carry
    return a
```

## 438. Find All Anagrams in a String

```
def find_anagrams(s, p):
    res = []
    pCounter = collections.Counter(p)
    sCounter = collections.Counter(s[:len(p)-1])
    for i in range(len(p)-1,len(s)):
        sCounter[s[i]] += 1
        if sCounter == pCounter:
            res.append(i-len(p)+1)
        sCounter[s[i-len(p)+1]] -= 1
        if sCounter[s[i-len(p)+1]] == 0:
            del sCounter[s[i-len(p)+1]]
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

## 617. [Merge Two Binary Trees](https://leetcode.com/problems/merge-two-binary-trees/)

```python3
def mergeTrees(self, t1, t2):
    if t1 and t2:
        root = TreeNode(t1.val + t2.val)
        root.left = self.mergeTrees(t1.left, t2.left)
        root.right = self.mergeTrees(t1.right, t2.right)
        return root
    else:
        return t1 or t2
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

## 794. Valid Tic-Tac-Toe State

Given a board state, determine if it is valid.

```
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

```
def most_common_word(paragraph, banned):
    banned_words = set(banned)
    words = re.findall(r'\w+', p.lower())
    return collections.Counter(w for w in words if w not in banned_words).most_common(1)[0][0]
```

## 1275. Find Winner on Tac Tac Toe Game

Given an order of moves, determine the winner of a tic tac toe game.

```
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

```
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

```
from itertools import accumulate

def running_sum(nums):
    return accumulate(nums)
    
```

```
    
def running_sum(nums):
    i = 1
    while i < len(nums):
        nums[i] += nums[i-1]
        i += 1
    return nums
```

