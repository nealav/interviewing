# Project Euler

## 1. Multiples of 3 and 5

```
def multiples(self):
    return sum([x for x in range(1000) if ((x % 5 == 0) or (x % 3 == 0))])
```

## 2. Even Fibonacci Numbers

```
def even_fib(n):
    sum = 0
    i, j = 0, 1
    while i < n:
        if i % 2 == 0:
            sum += i
        i, j = j, i+j
    return sum
```

## 3. Largest Prime Factor

```
def largest_prime_factor(n):
    i = 2
    while i < n:
        if n % i == 0 and n / i > 1:
            n = n / i
            i = 2
        else:
            i = i + 1
    return n 
```

## 4. Largest Palindrome Product

```
def largest_palindrome_product():
    palindromes = []
    numbers = [a * b for a in range(100,999) for b in range(100, 999)]
    for num in numbers:
        potential_palindrome = str(num)
        if potential_palindrome = potential_palindrome[::-1]:
            palindromes.append(num)
    return palindromes[-1]
```
