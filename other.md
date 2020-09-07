# GeeksForGeeks

## [Find The Position Of Element In Sorted Array Of Infinite Numbers](https://www.geeksforgeeks.org/find-position-element-sorted-array-infinite-numbers/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(logp) | O(1) | Binary Search |

```python3
def binary_search(arr, l, r, x): 
    if r >= l: 
        mid = l + (r - l)/2
        if arr[mid] == x:
            return mid
        if arr[mid] > x:
            return binary_search(arr, l, mid - 1, x)
        return binary_search(arr, mid + 1, r, x)
    return -1
  
def findPos(a, key):
    l, h, val = 0, 1, arr[0] 
    while val < key: 
        l = h
        h = 2*h
        val = arr[h]
    return binary_search(a, l, h, key) 
```

# HackerRank

## [Merge Sort: Counting Inversions](https://www.hackerrank.com/challenges/ctci-merge-sort/problem)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
O(n^2) | O(1) | Array |

Count the number of elements that are less than the current element after it and add to swap count. Each element less than the current element will require that many swaps either to move upward, or to move the lesser element downwards.

# Other

## Delete Odd Nodes in Linked List

```python3
def delete_odd_nodes(head):
    head.next = head.next.next
    temp = head.next
    while temp != head and temp.next != head:
        temp.next = temp.next.next
        temp = temp.next
    return head
```
