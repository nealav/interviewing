# LeetCode 301 - 400

## 322. [Coin Change](https://leetcode.com/problems/coin-change/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(n\*k) | O(k) | Dynamic Programming |

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

## 323. [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(V*E) | O(V) | Graph, Union-Find |

```python3
def count_components(n, edges):
    p = list(range(n))
    def find(v):
        if p[v] != v:
            p[v] = find(p[v])
        return p[v]
    for v, w in edges:
        p[find(v)] = find(w)
    return len(set(map(find, p)))
```

## 328. [Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | Linked List |

```python3
def odd_even_list(head):
    if not head:
        return head
    
    odd = head
    even = head.next
    even_head = even
    
    while even and even.next:
        odd.next = odd.next.next
        even.next = even.next.next
        odd = odd.next
        even = even.next
    
    odd.next = even_head
    return head
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

## 377. [Combination Sum IV](https://leetcode.com/problems/combination-sum-iv/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(NlogN) | O(N) | Dynamic Programming |

```python3
def combination_sum(nums, target):
    nums, combs = sorted(nums), [1] + [0] * (target)
    for i in range(target + 1):
        for num in nums:
            if num > i: break
            if num == i: combs[i] += 1
            if num < i: combs[i] += combs[i - num]
    return combs[target]
```

