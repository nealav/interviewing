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
