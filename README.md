# Interview Handbook

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


11. Container With Most Water

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

15. 3-Sum

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

23. Merge K Sorted Lists

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


33. Search in a Rotated Sorted Array

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

53. Maximum Subarray

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

70. Climbing Stairs

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

100. Same Tree

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

104. Maximum Depth of a Binary Tree

https://leetcode.com/problems/maximum-depth-of-binary-tree/

Simply go down the tree recursively in both directions on every node. Every node will add the next node to the stack. O(n) as it traverses all nodes.

```
def maxDepth(self, root):
    return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right)) if root else 0
```


121. Best Time to Buy and Sell Stock

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

152. Maximum Product Subarray

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


153. Find Minimum in Rotated Sorted Array

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


190. Reverse Bits

Pop off least siginficant bit and right shift it.

```
def reverseBits(n):
    res = 0
    for i in range(32):
        res = (res << 1) + (n & 1)
        n >>= 1
```

191. Number of 1 Bits

Pop the least significant bit and count until 0.

```
def hammingWeight(n):
    c = 0
    while n:
        n &= n - 1
        c += 1
    return c
```


217. Contains Duplicate

Given array of integers, find if array has duplicates.

Brute force - check one int across the array (n^2) or sort the array and check if any repeats one-after (nlogn)
Ideal - throw ints into set until repeat or check set length inequality.

```
def containsDuplicate(self, nums):
    return len(nums) != len(set(nums))
```

226. Invert Binary Tree

https://leetcode.com/problems/invert-binary-tree/solution/

```
def invertTree(self, root):
    if root is None:
        return None
    root.left, root.right = \
        self.invertTree(root.right), self.invertTree(root.left)
    return root
```


238. Product of Array Except Self

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

268. Missing Number

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


295. Find the Median From Data Stream

https://leetcode.com/problems/find-median-from-data-stream/

Simple sorting will be an O(nlogn) solution. Sort every time. Maintaining two heaps - one max and one min heap where each is maintained with equal sizes within 2 is the key to this problem.

322. Coin Change

This problem can be broken down into subtracting a single coin from the current amount value and checking if the subtracted amount has a number of times associated with it for all the coins in the bank.

```
def
```

347. Top K Frequent Elements

Given a non-empty array of integers, return the k most frequent elements.

https://leetcode.com/problems/top-k-frequent-elements/solution/

Using bucket sort we can maintain an array where the frequency is the index. Make a hashmap with the frequencies and add them to an array. This will take O(N). Using the bucket array we can return the top K values where K will always be less than N because the frequency can be no larger than the number of integers.

338. Counting Bits

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


342. Is Power Of Four

Powers of four in binary have a few characteristics: (1) greater than 0 (2) only have one 1 bit in their binary notation (3)) the 1 bit should be at an odd location.

```
def isPowerOfFour(n):
    return (
        n > 0 && #1
        !(n & (n - 1)) && #clear the 1 bit
        !(n & 0x55555555) #check the odd positions
    )

```

371. Sum of Two Integers Without '+'

Add the two numbers with the XOR operator. However in Binary this will take 1+1 and instead of 0 with a carry it will just be 0. We need to account for the carry with the & operator. The carry gets added to the next number so we keep adding until it becomes 0.

```
def getSum(a, b):
    c = 0
    while b != 0:
        c = a & b
        a = a ^ b
        b = c<<1 #reassign carry
    return a
```




# Project Euler

1. Multiples of 3 and 5

```
def multiples(self):
    return sum([x for x in range(1000) if ((x % 5 == 0) or (x % 3 == 0))])
```

