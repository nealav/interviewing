# LeetCode 101 - 200

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
| O(n) | O(n) | Tree |

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

## 110. [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(nlogn) | O(n) | Tree |

```python3
def height(root):
    if not root:
        return -1
    return 1 + max(height(root.left), height(root.right))

def isBalanced(root):
    if not root:
        return True

    return abs(height(root.left) - height(root.right)) < 2 \
        and isBalanced(root.left) \
        and isBalanced(root.right)
```

## 111. [Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(logn) | Tree |

```python3
def minDepth(root):
    if not root: 
        return 0 
    
    children = [root.left, root.right]
    if not any(children):
        return 1
    
    min_depth = float('inf')
    for c in children:
        if c:
            min_depth = min(minDepth(c), min_depth)
    return min_depth + 1
```

## 112. [Path Sum](https://leetcode.com/problems/path-sum/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(logn) | Tree |

```python3
def hasPathSum(root, sum):
    if not root:
        return False

    sum -= root.val
    if not root.left and not root.right:
        return sum == 0
    return hasPathSum(root.left, sum) or hasPathSum(root.right, sum)
```


## [118. Pascal's Triangle](https://leetcode.com/problems/pascals-triangle/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Math |

```python3
def generate(numRows):
    '''
    Construct a row with the offset sum of the previous row.
        1 3 3 1 0 
    +  0 1 3 3 1
    =  1 4 6 4 1
    '''

    res = [[1]]
    for i in range(1, numRows):
        res += [map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1])]
    return res[:numRows]
```

## [119. Pascal's Triangle II](https://leetcode.com/problems/pascals-triangle-ii/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Math |

```python3
def getRow(rowIndex):
    row = [1]
    for _ in range(rowIndex):
        row = [x + y for x, y in zip([0] + row, row + [0])]
    return row
```

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



## [122. Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | Greedy |

```python3
def maxProfit(prices):
    if not prices or len(prices) is 1: return 0
        profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]: profit += prices[i] - prices[i-1]
    return profit
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

## 128. [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Hash Table |

```python3
def longest_consecutive(nums):
    longest_streak = 0
    num_set = set(nums)

    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1

            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1

            longest_streak = max(longest_streak, current_streak)

    return longest_streak
```


## 133. [Clone Graph](https://leetcode.com/problems/clone-graph/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Graph |

```python3
from collections import deque
def cloneGraph(node):
    if not node:
        return node

    visited = {}
    queue = deque([node])
    visited[node] = Node(node.val, [])

    while queue:
        n = queue.popleft()
        for neighbor in n.neighbors:
            if neighbor not in visited:
                visited[neighbor] = Node(neighbor.val, [])
                queue.append(neighbor)
            visited[n].neighbors.append(visited[neighbor])

    return visited[node]
```

## 136. [Single Number](https://leetcode.com/problems/single-number/)

```python3
def single_number(nums):
    n = 0
    for num in nums:
        n ^= num
    return n
```

## 139. [Word Break](https://leetcode.com/problems/word-break/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n<sup>3</sup>) | O(n) | Dynamic Programming, Breadth First Search |

```python3
def word_break(s, word_dict):
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(len(s)):
        if dp[i]:
            for j in range(i + 1, len(s) + 1):
                if s[i:j] in word_dict:
                    dp[j] = True     
    return dp[-1]
```

## 141. [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(n) | Linked List, Floyd's Cycle Detection, Tortoise and Hare |

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

## 143. [Reorder List](https://leetcode.com/problems/reorder-list/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(1) | Linked List |

```python3
def reorder_list(head):
    if not head:
        return
    
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
    prev, curr = None, slow
    while curr:
        curr.next, prev, curr = prev, curr, curr.next

    first, second = head, prev
    while second.next:
        first.next, first = second, first.next
        second.next, second = first, second.next
```

## 146. LRU Cache

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(n) | Data Structure |

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

```python3
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



## [157. Read N Characters Given Read4](https://leetcode.com/problems/read-n-characters-given-read4/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | Array |

```python3
def read(buf, n):
    i = 0
    while i < n:
        buf4 = [''] * 4
        count = read4(buf4)
        if not count: break
        count = min(count, n - i)
        buf[i:] = buf4[:count]
        i += count
    return i
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

## 167. [Two Sum II](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n) | O(1) | Two Pointer |

```python3
def twoSum(numbers, target):
    l, r = 0, len(numbers) - 1
    while l < r:
        s = numbers[l] + numbers[r]
        if s == target:
            return [l + 1, r + 1]
        elif s < target:
            l += 1
        else:
            r -= 1
```


## 168. [Excel Sheet Column Title](https://leetcode.com/problems/excel-sheet-column-title/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(logn) | O(logn) | String |

```python3
def convertToTitle(num):
    result = []
    while x > 0:
        result.append(string.ascii_uppercase[(x - 1) % 26])
        x = (x - 1) // 26
    return "".join(reversed(result))
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

## [170. Two Sum III](https://leetcode.com/problems/two-sum-iii-data-structure-design/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(N) | Hash Table |

```python3
class TwoSum:

    def __init__(self):
        self.counter = {}

    def add(self, number):
        if number in self.counter:
            self.counter[number] += 1
        else:
            self.counter[number] = 1

    def find(self, value):
        counter = self.counter
        for num in counter:
            if value - num in counter and (value - num != num or counter[num] > 1):
                return True
        return False
```


## [171. Excel Sheet Column Number](https://leetcode.com/problems/excel-sheet-column-number/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | Math |

```python3
def titleToNumber(s):
    result = 0
    n = len(s)
    for i in range(n):
        result = result * 26
        result += (ord(s[i]) - ord('A') + 1)
    return result
```

## [172. Factorial Trailing Zeroes](https://leetcode.com/problems/factorial-trailing-zeroes/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(logN) | O(1) | Math |

```python3
def trailingZeroes(n):
    zero_count = 0
    while n > 0:
        n //= 5
        zero_count += n
    return zero_count
```


## [175. Combine Two Tables](https://leetcode.com/problems/combine-two-tables/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(~) | O(~) | SQL |

```sql
SELECT FirstName, LastName, City, State
FROM Person LEFT JOIN Address
ON Person.PersonId = Address.PersonId
;
```


## 176. [Second Highest Salary](https://leetcode.com/problems/second-highest-salary/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(1) | O(1) | SQL |

```sql
SELECT
    (SELECT DISTINCT
            Salary
        FROM
            Employee
        ORDER BY Salary DESC
        LIMIT 1 OFFSET 1) AS SecondHighestSalary
;
```


## [181. Employees Earning More Than Their Managers](https://leetcode.com/problems/employees-earning-more-than-their-managers/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(~) | O(~) | SQL |

```sql
SELECT
     a.NAME AS Employee
FROM Employee AS a JOIN Employee AS b
     ON a.ManagerId = b.Id
     AND a.Salary > b.Salary
;
```

## [182. Duplicate Emails](https://leetcode.com/problems/duplicate-emails/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(~) | O(~) | SQL |

```sql
SELECT Email
FROM Person
GROUP BY Email
HAVING COUNT(Email) > 1;
```

## [183. Customers Who Never Order](https://leetcode.com/problems/customers-who-never-order/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(~) | O(~) | SQL |

```sql
SELECT Customers.Name AS 'Customers'
FROM Customers
WHERE Customers.Id NOT IN
(
    SELECT CustomerId FROM Orders
);
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


## [193. Valid Phone Numbers](https://leetcode.com/problems/valid-phone-numbers/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(~) | O(~) | Bash |

```bash
grep -P '^(\d{3}-|\(\d{3}\) )\d{3}-\d{4}$' file.txt
sed -n -r '/^([0-9]{3}-|\([0-9]{3}\) )[0-9]{3}-[0-9]{4}$/p' file.txt
awk '/^([0-9]{3}-|\([0-9]{3}\) )[0-9]{3}-[0-9]{4}$/' file.txt
```

## [195. Tenth Line](https://leetcode.com/problems/valid-phone-numbers/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(~) | O(~) | Bash |

```bash
awk 'NR == 10' file.txt
sed -n 10p file.txt
```

## [196. Delete Duplicate Emails](https://leetcode.com/problems/delete-duplicate-emails/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(~) | O(~) | SQL |

```sql
DELETE p1 FROM Person p1,
    Person p2
WHERE
    p1.Email = p2.Email AND p1.Id > p2.Id
```

## [197. Rising Temperature](https://leetcode.com/problems/rising-temperature/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(~) | O(~) | SQL |

```sql
SELECT
    Weather.Id AS 'Id'
FROM
    Weather
        JOIN
    Weather w ON DATEDIFF(Weather.RecordDate, w.RecordDate) = 1
        AND Weather.Temperature > w.Temperature
;
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