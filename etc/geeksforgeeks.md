# GeeksForGeeks

## [Find The Position Of Element In Sorted Array Of Infinite Numbers](https://www.geeksforgeeks.org/find-position-element-sorted-array-infinite-numbers/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(logp) | O(1) | Binary Search |

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

## [Delete Odd Nodes in Linked List](https://www.geeksforgeeks.org/delete-all-odd-or-even-positioned-nodes-from-circular-linked-list/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(N) | O(1) | Linked List |

```python3
def delete_odd_nodes(head):
    head.next = head.next.next
    temp = head.next
    while temp != head and temp.next != head:
        temp.next = temp.next.next
        temp = temp.next
    return head
```

