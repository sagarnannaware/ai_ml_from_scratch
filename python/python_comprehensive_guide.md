# Complete Python Programming Guide (for AI/ML Engineering) — Detailed with Explanations & Examples

---

## 1. Python Basics

### Syntax & Data Types

#### Variables

- **Definition:** A variable is a name that refers to a value stored in memory. In Python, you do not need to declare the type; the interpreter figures it out automatically.

**Examples:**
```python
# Assigning values
x = 10           # integer
y = 3.14         # float
name = "Alice"   # string
is_valid = True  # boolean

# Using variables in expressions
z = x + y        # z = 13.14
greeting = "Hello, " + name  # greeting = "Hello, Alice"

# Reassigning variables
x = 20           # x is now 20

# Multiple assignment
a, b, c = 1, 2, 3

# Swapping values
a, b = b, a      # a = 2, b = 1

# Using variables in functions
def add_numbers(num1, num2):
    result = num1 + num2
    return result

print(add_numbers(x, y))   # Output: 23.14

# Assigning collections
numbers = [1, 2, 3]        # list
person = {"name": name, "age": x}  # dictionary
```
- **When to use:**  
  Use variables to store data for later use, make code more readable, avoid hardcoding values.

- **Naming Rules & Good Practices:**
  - Start with letter or underscore, cannot start with a number.
  - Case sensitive.
  - Descriptive names (e.g., `user_age`), use lowercase_with_underscores per PEP8.

#### Data Types

- **int:** Whole numbers (`a = 5`)
- **float:** Decimal numbers (`b = 3.14`)
- **str:** Strings (`s = "hello"`)
- **bool:** Boolean (`is_valid = True`)
- **list:** Ordered, mutable (`l = [1, 2, 3]`)
- **tuple:** Ordered, immutable (`t = (1, 2, 3)`)
- **dict:** Key-value pairs (`d = {'a': 1}`)
- **set:** Unordered, unique (`s = {1, 2, 3}`)

---

### Working with Lists, Sets, and Dictionaries

#### List

A **list** is an ordered, mutable collection.

**Sample:**
```python
fruits = ["apple", "banana", "cherry"]
print(fruits[0])            # Access: "apple"
fruits.append("orange")     # Add item
fruits.insert(1, "mango")   # Insert at index
removed = fruits.pop()      # Remove and return last item ("orange")
fruits.remove("banana")     # Remove by value
print(fruits)               # Manipulated list
print(len(fruits))          # Length of list

# Iteration
for fruit in fruits:
    print(fruit)
```

#### Set

A **set** is an unordered collection of unique elements.

**Sample:**
```python
numbers = {1, 2, 3}
numbers.add(4)              # Add an element
numbers.update([2, 5, 6])   # Add multiple elements
numbers.discard(2)          # Remove an element (no error if not found)
removed = numbers.pop()     # Remove and return an arbitrary element
print(numbers)              # Manipulated set

# Set operations
a = {1, 2, 3}
b = {3, 4, 5}
print(a | b)   # Union: {1, 2, 3, 4, 5}
print(a & b)   # Intersection: {3}
print(a - b)   # Difference: {1, 2}
print(a ^ b)   # Symmetric difference: {1, 2, 4, 5}

# Iteration
for num in numbers:
    print(num)
```

#### Dictionary

A **dictionary** is a collection of key-value pairs.

**Sample:**
```python
person = {"name": "Alice", "age": 30}
print(person["name"])           # Access value
person["email"] = "a@mail.com"  # Add new key-value
person["age"] = 31              # Update value
removed = person.pop("email")   # Remove key and return value
print(person)                   # Manipulated dict
print(len(person))              # Number of keys

# Iteration
for key, value in person.items():
    print(key, value)

# Dictionary comprehension
squared = {x: x**2 for x in range(5)}  # {0:0, 1:1, 2:4, ...}
```

---

### Lambda Functions

A **lambda function** in Python is a small, anonymous (unnamed) function defined with the `lambda` keyword.  
It can take any number of arguments, but can only have one expression (no statements).

**General syntax:**
```python
lambda arguments: expression
```

---

#### Lambda Function Samples

##### 1. Basic Usage

```python
add = lambda x, y: x + y
print(add(2, 3))      # Output: 5

square = lambda x: x * x
print(square(4))      # Output: 16
```

##### 2. Using with `map()`

Apply a function to each item in a list.

```python
numbers = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, numbers))
print(squared)        # Output: [1, 4, 9, 16]
```

##### 3. Using with `filter()`

Filter items in a list by a condition.

```python
numbers = [1, 2, 3, 4, 5, 6]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)          # Output: [2, 4, 6]
```

##### 4. Using with `sorted()` (custom sort key)

```python
pairs = [(1, 'one'), (3, 'three'), (2, 'two'), (4, 'four')]
# Sort by second item in each tuple (the string)
sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
print(sorted_pairs)
# Output: [(4, 'four'), (1, 'one'), (3, 'three'), (2, 'two')]
```

##### 5. Lambda inside functions

You can return a lambda from a function.

```python
def make_multiplier(n):
    return lambda x: x * n

double = make_multiplier(2)
print(double(5))      # Output: 10

triple = make_multiplier(3)
print(triple(4))      # Output: 12
```

##### 6. Lambda for small utility functions

```python
# No need to define a separate function for simple operations
names = ['alice', 'Bob', 'charlie']
capitalized = list(map(lambda s: s.capitalize(), names))
print(capitalized)    # Output: ['Alice', 'Bob', 'Charlie']
```

##### 7. Lambda with `reduce()`

Reduce a list to a single value (needs functools).

```python
from functools import reduce

numbers = [1, 2, 3, 4]
product = reduce(lambda x, y: x * y, numbers)
print(product)        # Output: 24
```

##### 8. Lambda with conditional expressions

```python
max_func = lambda x, y: x if x > y else y
print(max_func(10, 15))   # Output: 15
```

##### 9. Lambda with default arguments

```python
add_five = lambda x, y=5: x + y
print(add_five(10))       # Output: 15
```

**When to Use Lambda Functions?**

- For small, simple functions as arguments to functions like `map`, `filter`, `sorted`.
- When a full function with `def` would be unnecessarily verbose.
- For simple one-liners.

**When NOT to Use Lambda Functions?**

- When the logic is complex or multi-line; use `def` and a named function for clarity.

---

### Control Structures

- **if/elif/else:** Conditional execution
- **for:** Loop over iterable objects
- **while:** Loop as long as a condition is true

**Examples:**
```python
if a > 5:
    print("a is big")
elif a == 5:
    print("a is five")
else:
    print("a is small")

for i in range(5):
    print(i)

while a > 0:
    a -= 1
```
- **break/continue/pass:** Control loop flow  
  `break` exits the loop, `continue` skips to next iteration, `pass` is a no-op placeholder.

- **With/else on loops:**
  ```python
  for i in range(3):
      if i == 2:
          break
  else:
      print("Not broken!")  # Will not print because the loop broke
  ```

### Functions

- **Definition:** Use `def` to define functions.  
- **Arguments:** Can have default, variable (`*args`, `**kwargs`), keyword arguments.
- **First-class:** Can pass functions as arguments, return them, assign to variables.
- **Docstrings:** Use triple quotes for documentation.

**Examples:**
```python
def add(x, y=0):
    """Returns sum of x and y"""
    return x + y

result = add(5, 2)

def example(*args, **kwargs):
    print(args, kwargs)
example(1, 2, three=3)
```

### Comprehensions

- **List/Dict/Set comprehensions:** Concise way to create collections.
  ```python
  squares = [x**2 for x in range(5)]
  squared_dict = {x: x**2 for x in range(5)}
  evens = {x for x in range(10) if x % 2 == 0}
  ```

---

## 2. Object-Oriented Programming (OOP) in Python

### Classes & Objects

- **Class:** Blueprint for objects; defines methods and properties.
- **Object:** Instance of a class.
- **`__init__`:** Constructor called on object creation.
- **Class variables vs Instance variables:**
  ```python
  class Demo:
      count = 0  # Class variable
      def __init__(self):
          Demo.count += 1
          self.id = Demo.count  # Instance variable
  ```

- **Private & Protected:** Use `_protected` (convention), `__private` (name mangling).

**Examples:**
```python
class Person:
    def __init__(self, name):
        self.name = name
    def greet(self):
        print(f"Hello, {self.name}!")

p = Person("Alice")
p.greet()
```

### Inheritance & Polymorphism

- **Inheritance:** Child class inherits properties/methods from parent.
- **Polymorphism:** Different classes define methods with same name, can be used interchangeably.
- **Multiple inheritance:** A class can inherit from multiple parents.

**Examples:**
```python
class Animal:
    def speak(self): pass

class Dog(Animal):
    def speak(self):
        print("Woof")
a = Dog()
a.speak()
```

### Accessing Parent Class from Child Class

- **Using `super()`:**
  ```python
  class Parent:
      def __init__(self, value):
          self.value = value
          print("Parent __init__ called")
      def show(self):
          print(f"Parent value: {self.value}")

  class Child(Parent):
      def __init__(self, value, extra):
          super().__init__(value)
          self.extra = extra
          print("Child __init__ called")
      def show(self):
          super().show()
          print(f"Child extra: {self.extra}")

  c = Child(10, "extra info")
  c.show()
  ```
- **Accessing parent attributes directly:**
  ```python
  class Parent:
      def __init__(self):
          self.parent_attr = "I am parent"
  class Child(Parent):
      def display(self):
          print(self.parent_attr)
  c = Child()
  c.display()
  ```
- **Calling parent methods by name:**
  ```python
  class Parent:
      def say_hello(self):
          print("Hello from Parent")
  class Child(Parent):
      def say_hello(self):
          print("Hello from Child")
          Parent.say_hello(self)
  c = Child()
  c.say_hello()
  ```
- **Multiple inheritance:**
  ```python
  class A: 
      def hello(self):
          print("Hello from A")
  class B: 
      def hello(self): 
          print("Hello from B")
  class C(A, B):
      def hello(self):
          super().hello()
          print("Hello from C")
  obj = C()
  obj.hello()
  ```

### Abstract Classes & Methods

- **Abstract classes:** Provide common interface, enforce implementation of methods in subclasses.
  ```python
  from abc import ABC, abstractmethod

  class Animal(ABC):
      @abstractmethod
      def speak(self):
          pass

  class Dog(Animal):
      def speak(self):
          print("Woof")
  dog = Dog()
  dog.speak()
  ```
  - Cannot instantiate abstract class directly.

### Static Methods & Class Methods

- **Static methods:** No access to instance or class; utility functions.
  ```python
  class Math:
      @staticmethod
      def add(x, y):
          return x + y
  print(Math.add(3, 4))
  ```
- **Class methods:** Receive class as first argument (`cls`), can access/modify class state.
  ```python
  class Counter:
      count = 0
      @classmethod
      def increment(cls):
          cls.count += 1
          return cls.count
  print(Counter.increment())
  ```
- **Difference:**  
  - `@classmethod` can modify class variables (shared across instances).
  - `@staticmethod` is just namespaced in the class, cannot access/modify class or instance state.
  - The parameter name `cls` is a convention, not a keyword.

---

## 3. Advanced Python Concepts

### Decorators

- **Definition:** Functions that modify behavior of other functions/classes.
- **When to use:** Add logic (e.g., logging, timing) to multiple functions, separate concerns, used in frameworks.

**Examples:**
```python
def my_decorator(func):
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")
say_hello()
```
- **Timing example:**
  ```python
  import time
  def timer(func):
      def wrapper(*args, **kwargs):
          start = time.time()
          result = func(*args, **kwargs)
          end = time.time()
          print(f"{func.__name__} took {end-start} seconds")
          return result
      return wrapper

  @timer
  def compute():
      time.sleep(1)

  compute()
  ```

### Property Decorators

- **@property:** Make method accessible as attribute (getter/setter).
  ```python
  class Circle:
      def __init__(self, radius):
          self._radius = radius
      @property
      def radius(self):
          return self._radius
      @radius.setter
      def radius(self, value):
          if value < 0:
              raise ValueError("Negative radius not allowed")
          self._radius = value

  c = Circle(5)
  c.radius = 10
  print(c.radius)
  ```

### Generators & Iterators

- **Iterator:** Object with `__iter__()` and `__next__()` methods.
- **Generator:** Use `yield` to return sequence of values, saves memory.
- **Generator expressions:** Like list comprehensions but lazy.

**Examples:**
```python
def gen():
    for i in range(5):
        yield i

g = gen()
print(next(g))  # 0
print(list(g))  # [1, 2, 3, 4]
squares = (x*x for x in range(10))
for s in squares:
    print(s)
```

### Context Managers

- **Definition:** Manage resources with `with` block, guarantee cleanup.
- **Example:**
  ```python
  with open('file.txt', 'w') as f:
      f.write("Hello")
  # File automatically closed

  class MyResource:
      def __enter__(self):
          print("Acquire resource")
          return self
      def __exit__(self, exc_type, exc_value, traceback):
          print("Release resource")

  with MyResource():
      print("Using resource")
  ```

---

## 4. Modules & Packages

- **Module:** `.py` file of code; import via `import module`.
- **Package:** Directory with `__init__.py`, contains modules.

**Example:**
```python
# In my_module.py
def greet(name):
    print(f"Hello, {name}!")

# In main.py
import my_module
my_module.greet("Alice")
```

---

## 5. Exception Handling

- **Purpose:** Manage errors gracefully, avoid crashes.
- **Syntax:**
  ```python
  try:
      1 / 0
  except ZeroDivisionError:
      print("Cannot divide by zero")
  finally:
      print("Done")
  ```
- **Custom Exceptions:**
  ```python
  class MyError(Exception):
      pass
  try:
      raise MyError("Something went wrong")
  except MyError as e:
      print(e)
  ```

---

## 6. Pythonic Practices & Idioms

- List comprehensions, `enumerate`, `with`, unpacking (`a, b = b, a`), swap values, docstrings, PEP8.
- **Always strive for code that is clean, readable, idiomatic.**

---

## 7. Type Hints & Annotations

- **Purpose:** Indicate expected types for clarity, static analysis.
- **Examples:**
  ```python
  def add(x: int, y: int) -> int:
      return x + y
  from typing import List, Dict
  nums: List[int] = [1, 2, 3]
  mapping: Dict[str, float] = {'a': 1.0}
  ```

---

## 8. Virtual Environments & Dependency Management

- **Virtual environments:** Isolate dependencies per project.
  - `python -m venv venv`
  - `source venv/bin/activate` (Linux/Mac), `venv\Scripts\activate` (Windows)
- **pip:** Install packages.
- **requirements.txt:** List for reproducibility (`pip freeze > requirements.txt`).
- **pipenv/poetry:** More advanced environment/dependency managers.

---

## 9. Testing

- **Unit testing:** Isolated code tests.
  ```python
  import unittest

  class TestAdd(unittest.TestCase):
      def test_add(self):
          self.assertEqual(add(1, 2), 3)

  if __name__ == '__main__':
      unittest.main()
  ```
- **pytest:** Popular third-party tool for testing.

---

## 10. Performance Tips

- Prefer built-ins, use **NumPy** for heavy computation, profile with `cProfile`/`timeit`, avoid globals, use generators for large data.

---

## 11. Useful Built-in Libraries

- **os:** File/directory ops
- **sys:** System functions
- **json:** JSON parsing
- **datetime:** Dates/times
- **re:** Regex
- **logging:** Logging

---

## 12. Python for AI/ML

- Use **NumPy**, **Pandas** for data; **Matplotlib/Seaborn** for visualization; **scikit-learn**, **TensorFlow**, **PyTorch** for ML.
- Write modular code, use version control, document experiments.

---

## 13. Best Practices for Competitive Python

- Master the standard library, write modular/readable code, use efficient data structures, understand Big O, practice coding challenges, write tests/assertions, use Python features (f-strings, context managers, type hints).

---

## 14. Example: End-to-End AI/ML Workflow

```python
# 1. Environment setup (done in terminal)
# python -m venv venv
# source venv/bin/activate
# pip install numpy pandas scikit-learn matplotlib

# 2. Data loading & exploration
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
print(df.head())
df.hist()
plt.show()

# 3. Data preparation
from sklearn.model_selection import train_test_split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Modeling
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Evaluation
print("Accuracy:", model.score(X_test, y_test))

# 6. Saving model
import joblib
joblib.dump(model, 'model.pkl')
```

---

**Tip:**  
For AI/ML engineering, combine strong Python fundamentals, write clear and modular code, and be familiar with the scientific Python ecosystem. Practice regularly, explore open-source projects, and build real-world projects to deepen your expertise.