# LeetCode 1 - 100

## 1. [Two Sum](https://leetcode.com/problems/two-sum)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(n) | Hash Table |

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


## [2. Add Two Numbers](https://leetcode.com/problems/add-two-numbers/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(max(M, N)) | O(max(M, N)) | Linked List |

```python3
def addTwoNumbers(l1, l2):
    carry = 0;
    res = n = ListNode(0);
    while l1 or l2 or carry:
        if l1:
            carry += l1.val
            l1 = l1.next;
        if l2:
            carry += l2.val;
            l2 = l2.next;
        carry, val = divmod(carry, 10)
        n.next = n = ListNode(val);
    return res.next;
```


## [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Sliding Window |

```python3
def lengthOfLongestSubstring(s):
    used = {}
    max_length = start = 0

    for i, c in enumerate(s):
        if c in used and start <= used[c]:
            start = used[c] + 1
        else:
            max_length = max(max_length, i - start + 1)
        used[c] = i

    return max_length
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


## [6. ZigZag Conversion](https://leetcode.com/problems/zigzag-conversion/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | String |

```python3
def convert(s, numRows):
    if numRows == 1 or numRows >= len(s):
        return s

    L = [''] * numRows
    index, step = 0, 1

    for x in s:
        L[index] += x
        if index == 0:
            step = 1
        elif index == numRows - 1:
            step = -1
        index += step

    return ''.join(L)
```

## 7. [Reverse Integer](https://leetcode.com/problems/reverse-integer/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(logn) | O(1) | Math |

```python3
def reverse(self, x):
    sign = [1, -1][x < 0]
    rev, x = 0, abs(x)
    while x:
        x, mod = divmod(x, 10)
        rev = rev * 10 + mod
    return sign * rev if -pow(2, 31) <= sign * rev <= pow(2, 31) - 1 else 0
```


## [8. String to Integer (atoi)](https://leetcode.com/problems/string-to-integer-atoi/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | String |

```python3
def atoi(s):
    s = s.strip()
    s = re.findall('(^[\+\-0]*\d+)\D*', s)

    try:
        result = int(''.join(s))
        MAX_INT = 2147483647
        MIN_INT = -2147483648
        if result > MAX_INT > 0:
            return MAX_INT
        elif result < MIN_INT < 0:
            return MIN_INT
        else:
            return result
    except:
        return 0
```


## 9. [Palindrome Number](https://leetcode.com/problems/palindrome-number/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(logn) | O(1) | String |

```python3
def isPalindrome(x):
    if x < 0 or (x % 10 == 0 and x != 0):
        return False

    reverted = 0
    while x > reverted:
        reverted = reverted * 10 + x % 10
        x = x // 10
        print(x, reverted)

    return x == reverted or x == reverted // 10
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


## [12. Integer to Roman](https://leetcode.com/problems/integer-to-roman/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(1) | O(1) | String |

```python3
def intToRoman(num):
    thousands = ["", "M", "MM", "MMM"]
    hundreds = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
    tens = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
    ones = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
    return thousands[num // 1000] + hundreds[num % 1000 // 100] + tens[num % 100 // 10] + ones[num % 10]
```

## 13. [Roman To Integer](https://leetcode.com/problems/roman-to-integer/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(1) | O(1) | String |

```python3
values = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}

def romanToInt(s):
    total = values.get(s[-1])
    for i in reversed(range(len(s) - 1)):
        if values[s[i]] < values[s[i + 1]]:
            total -= values[s[i]]
        else:
            total += values[s[i]]
    return total
```

## 14. [Happy Number](https://leetcode.com/problems/longest-common-prefix/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(1) | String, Trie |

```python3
 def longestCommonPrefix(strs):
    if not strs:
        return ""
    shortest = min(strs,key=len)
    for i, ch in enumerate(shortest):
        for other in strs:
            if other[i] != ch:
                return shortest[:i]
    return shortest 
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


## [16. 3Sum Closest](https://leetcode.com/problems/3sum-closest/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n<sup>2</sup>) | O(1) | Two Pointer |

```python3
def threeSumClosest(nums, target):
    diff = float('inf')
    nums.sort()
    for i in range(len(nums)):
        lo, hi = i + 1, len(nums) - 1
        while (lo < hi):
            sum = nums[i] + nums[lo] + nums[hi]
            if abs(target - sum) < abs(diff):
                diff = target - sum
            if sum < target:
                lo += 1
            else:
                hi -= 1
        if diff == 0:
            break
    return target - diff
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

## 19. [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(1) | Linked List |

```python3
def remove_nth_from_end(head, n):
    fast = slow = head
    for _ in range(n):
        fast = fast.next
    if not fast:
        return head.next
    while fast.next:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return head
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



## [27. Remove Element](https://leetcode.com/problems/remove-element/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(1) | Two Pointer |

```python3
def removeElement(nums, val):
    start, end = 0, len(nums) - 1
    while start <= end:
        if nums[start] == val:
            nums[start], nums[end], end = nums[end], nums[start], end - 1
        else:
            start +=1
    return start
```


## [28. Implement strStr()](https://leetcode.com/problems/implement-strstr/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n * h) | O(1) | Two Pointer |

```python3
def strStr(haystack, needle):
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1
```


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


## [35. Search Insert Position](https://leetcode.com/problems/search-insert-position/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(logN) | O(1) | Binary Search |

```python3
def searchInsert(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        pivot = (left + right) // 2
        if nums[pivot] == target:
            return pivot
        if target < nums[pivot]:
            right = pivot - 1
        else:
            left = pivot + 1
    return left
```


## [38. Count and Say](https://leetcode.com/problems/count-and-say/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(2<sup>N</sup>) | O(2<sup>N</sup>) | Sliding Window |

```python3
def countAndSay(n):
    s = '1'
    for _ in range(n - 1):
        s = ''.join(str(len(list(group))) + digit \
                    for digit, group in itertools.groupby(s))
    return s
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
| O(n) | O(1) | Dynamic Programming |

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
| O(nlogn) | O(n) | Sorting |

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

## 57. [Insert Interval](https://leetcode.com/problems/insert-interval/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(n) | Interval |

```python3
def insert(intervals, new_interval):
    s, e = new_interval.start, new_interval.end
    left = [i for i in intervals if i.end < s]
    right = [i for i in intervals if i.start > e]
    if left + right != intervals:
        s = min(s, intervals[len(left)].start)
        e = max(e, intervals[~len(right)].end)
    return left + [Interval(s, e)] + right
```

## 62. [Unique Paths](https://leetcode.com/problems/unique-paths/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N * M) | O(N * M) | Dynamic Programming, Math |

```python3
def uniquePaths(m, n):
    d = [[1] * n for _ in range(m)]

    for col in range(1, m):
        for row in range(1, n):
            d[col][row] = d[col - 1][row] + d[col][row - 1]

    return d[m - 1][n - 1]
```

## 66. [Plus One](https://leetcode.com/problems/plus-one/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | Math |

```python3
def plusOne(digits):
    for i in reversed(range(len(digits))):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        digits[i] = 0
    return [1] + [0] * len(digits)
```

## 67. [Add Binary](https://leetcode.com/problems/add-binary/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(1) | Bit Manipulation |

```python3
def addBinary(a, b):
    x, y = int(a, 2), int(b, 2)
    while y:
        x, y = x ^ y, (x & y) << 1
    return bin(x)[2:]
```

## 69. [Sqrt(x)](https://leetcode.com/problems/sqrtx/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(logn) | O(1) | Math |

```python3
def mySqrt(x):
    if x < 2:
        return x
    
    x0 = x
    x1 = (x0 + x / x0) / 2
    while abs(x0 - x1) >= 1:
        x0 = x1
        x1 = (x0 + x / x0) / 2        
        
    return int(x1)
```

## 70. [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(nlogn) | O(n) | Dynamic Programming |

```python3
def climing_stairs(n):
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
| O(mn) | O(mn) | Dynamic Programming, [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance) |

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

## 73. [Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n + m) | O(1) | Matrix |

```python3
def set_zeroes(matrix):
    is_col = False
    R = len(matrix)
    C = len(matrix[0])
    for i in range(R):
        if matrix[i][0] == 0:
            is_col = True
        for j in range(1, C):
            if matrix[i][j]  == 0:
                matrix[0][j] = 0
                matrix[i][0] = 0

    for i in range(1, R):
        for j in range(1, C):
            if not matrix[i][0] or not matrix[0][j]:
                matrix[i][j] = 0

    if matrix[0][0] == 0:
        for j in range(C):
            matrix[0][j] = 0

    if is_col:
        for i in range(R):
            matrix[i][0] = 0
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

## 88. [Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O((n+m)log(n+m)) | O(1) | Sort, Two Pointer |

```python3
def merge(nums1, m, nums2, n):
    nums1[m:] = nums2[:n]
    nums1.sort()
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



## 98. [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Binary Search Tree |

```python3
def is_valid_BST(root):
    stack, inorder = [], float('-inf')
    
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        if root.val <= inorder:
            return False
        inorder = root.val
        root = root.right

    return True
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
