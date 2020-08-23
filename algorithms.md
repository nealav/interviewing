# Algorithms

* [Binary Search Tree](https://github.com/nealav/interviewing/blob/master/algorithms.md#binary-search-tree)

## Binary Search Tree

* https://en.wikipedia.org/wiki/Tree_traversal

```python3
class Node: 
    def __init__(self, key): 
        self.left = None
        self.right = None
        self.val = key
```

## Inorder Traversal (Left, Root, Right)

```python3
def inorder(node):
    if node:
        preorder(node.left)
        visit(node)
        preorder(node.right)
```

## Preorder Traversal (Root, Left, Right)

```python3
def preorder(node):
    if node:
        visit(node)
        preorder(node.left)
        preorder(node.right)
```

## Postorder Traversal (Left, Right, Root)

```python3
def postorder(node):
    if node:
        postorder(node.left)
        postorder(node.right)
        visit(node)
```

## Level-Order Traversal

```python3
def levelorder(node):
    queue = []
    queue.push(node)
    while queue:
        node = queue.pop()
        visit(node)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
```
