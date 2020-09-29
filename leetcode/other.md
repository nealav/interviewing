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

## 937. [Reorder Data In Log Files](https://leetcode.com/problems/reorder-data-in-log-files/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(m * nlogn) | O(m * n) | String |

```python3
def reorderLogFiles(logs):

    def get_key(log):
        _id, rest = log.split(" ", maxsplit=1)
        return (0, rest, _id) if rest[0].isalpha() else (1, )

    return sorted(logs, key=get_key)
```

## 953. [Verifying an Alien Dictionary](https://leetcode.com/problems/verifying-an-alien-dictionary/)

| Time    | Space    | Tags           |
|-------- | -------- | -------------- |
| O(m + n) | O(1) | String |

```python3
def isAlienSorted(words, order):
    order_index = {c: i for i, c in enumerate(order)}

    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i+1]

        for k in range(min(len(word1), len(word2))):
            if word1[k] != word2[k]:
                if order_index[word1[k]] > order_index[word2[k]]:
                    return False
                break
        else:
            if len(word1) > len(word2):
                return False

    return True
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
