from typing import List, Any, Dict, Set, Generator

class StaticArray:
    def __init__(self, capacity: int):
        self.array = [None] * capacity
        self.capacity = capacity

    def set(self, index: int, value: int) -> None:
        if 0 <= index < self.capacity:
            self.array[index] = value
        else:
            raise IndexError("Index out of bounds.")

    def get(self, index: int) -> int:
        if 0 <= index < self.capacity:
            return self.array[index]
        else:
            raise IndexError("Index out of bounds.")

class DynamicArray:
    def __init__(self):
        self.array = []

    def append(self, value: int) -> None:
        self.array.append(value)

    def insert(self, index: int, value: int) -> None:
        if 0 <= index <= len(self.array):
            self.array.insert(index, value)
        else:
            raise IndexError("Index out of bounds.")

    def delete(self, index: int) -> None:
        if 0 <= index < len(self.array):
            self.array.pop(index)
        else:
            raise IndexError("Index out of bounds.")

    def get(self, index: int) -> int:
        if 0 <= index < len(self.array):
            return self.array[index]
        else:
            raise IndexError("Index out of bounds.")

class Node:
    def __init__(self, value: int):
        self.value = value
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, value: int) -> None:
        new_node = Node(value)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def insert(self, position: int, value: int) -> None:
        new_node = Node(value)
        if position == 0:
            new_node.next = self.head
            self.head = new_node
            return
        current = self.head
        for _ in range(position - 1):
            if current is None:
                raise IndexError("Position out of bounds.")
            current = current.next
        new_node.next = current.next
        current.next = new_node

    def delete(self, value: int) -> None:
        if not self.head:
            return
        if self.head.value == value:
            self.head = self.head.next
            return
        current = self.head
        while current.next and current.next.value != value:
            current = current.next
        if current.next:
            current.next = current.next.next

    def find(self, value: int) -> Node:
        current = self.head
        while current:
            if current.value == value:
                return current
            current = current.next
        return None

    def size(self) -> int:
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count

    def is_empty(self) -> bool:
        return self.head is None

    def print_list(self) -> None:
        current = self.head
        while current:
            print(current.value, end=" -> ")
            current = current.next
        print("None")

    def reverse(self) -> None:
        prev = None
        current = self.head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev

    def get_head(self) -> Node:
        return self.head

    def get_tail(self) -> Node:
        current = self.head
        while current and current.next:
            current = current.next
        return current


class DoubleNode:
    def __init__(self, value: int, next_node=None, prev_node=None):
        self.value = value
        self.next = next_node
        self.prev = prev_node

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, value: int) -> None:
        new_node = DoubleNode(value)
        if not self.head:
            self.head = self.tail = new_node
            return
        self.tail.next = new_node
        new_node.prev = self.tail
        self.tail = new_node

    def insert(self, position: int, value: int) -> None:
        new_node = DoubleNode(value)
        if position == 0:
            new_node.next = self.head
            if self.head:
                self.head.prev = new_node
            self.head = new_node
            if not self.tail:
                self.tail = new_node
            return
        current = self.head
        for _ in range(position - 1):
            if not current:
                raise IndexError("Position out of bounds.")
            current = current.next
        new_node.next = current.next
        new_node.prev = current
        if current.next:
            current.next.prev = new_node
        current.next = new_node
        if new_node.next is None:
            self.tail = new_node

    def delete(self, value: int) -> None:
        current = self.head
        while current:
            if current.value == value:
                if current.prev:
                    current.prev.next = current.next
                if current.next:
                    current.next.prev = current.prev
                if current == self.head:
                    self.head = current.next
                if current == self.tail:
                    self.tail = current.prev
                return
            current = current.next

    def find(self, value: int) -> DoubleNode:
        current = self.head
        while current:
            if current.value == value:
                return current
            current = current.next
        return None

    def size(self) -> int:
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count

    def is_empty(self) -> bool:
        return self.head is None

    def print_list(self) -> None:
        current = self.head
        while current:
            print(current.value, end=" <-> ")
            current = current.next
        print("None")

    def reverse(self) -> None:
        current = self.head
        prev = None
        while current:
            next_node = current.next
            current.next = prev
            current.prev = next_node
            prev = current
            current = next_node
        self.head, self.tail = self.tail, self.head

    def get_head(self) -> DoubleNode:
        return self.head

    def get_tail(self) -> DoubleNode:
        return self.tail


class Queue:
    def __init__(self):
        self.queue = []

    def enqueue(self, value: int) -> None:
        self.queue.append(value)

    def dequeue(self) -> int:
        if not self.is_empty():
            return self.queue.pop(0)
        raise IndexError("Queue is empty.")

    def peek(self) -> int:
        if not self.is_empty():
            return self.queue[0]
        raise IndexError("Queue is empty.")

    def is_empty(self) -> bool:
        return len(self.queue) == 0

class TreeNode:
    def __init__(self, value: int):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, value: int) -> None:
        def _insert(node, value):
            if not node:
                return TreeNode(value)
            if value < node.value:
                node.left = _insert(node.left, value)
            else:
                node.right = _insert(node.right, value)
            return node
        self.root = _insert(self.root, value)

    def delete(self, value: int) -> None:
        def _delete(node, value):
            if not node:
                return None
            if value < node.value:
                node.left = _delete(node.left, value)
            elif value > node.value:
                node.right = _delete(node.right, value)
            else:
                if not node.left:
                    return node.right
                if not node.right:
                    return node.left
                temp = node.right
                while temp.left:
                    temp = temp.left
                node.value = temp.value
                node.right = _delete(node.right, temp.value)
            return node
        self.root = _delete(self.root, value)

    def search(self, value: int) -> TreeNode:
        def _search(node, value):
            if not node or node.value == value:
                return node
            if value < node.value:
                return _search(node.left, value)
            return _search(node.right, value)
        return _search(self.root, value)

    def inorder_traversal(self) -> List[int]:
        result = []
        def _inorder(node):
            if node:
                _inorder(node.left)
                result.append(node.value)
                _inorder(node.right)
        _inorder(self.root)
        return result

    def size(self) -> int:
        def _size(node):
            if not node:
                return 0
            return 1 + _size(node.left) + _size(node.right)
        return _size(self.root)

    def is_empty(self) -> bool:
        return self.root is None

    def height(self) -> int:
        def _height(node):
            if not node:
                return 0  
            left_height = _height(node.left)
            right_height = _height(node.right)
            return 1 + max(left_height, right_height)
        return _height(self.root)


    def preorder_traversal(self) -> List[int]:
        result = []
    
        def _preorder(node):
            if node:
                result.append(node.value)  
                _preorder(node.left)     
                _preorder(node.right)  
    
        _preorder(self.root)
        return result


    def postorder_traversal(self) -> List[int]:
        result = []
    
        def _postorder(node):
            if node:
                _postorder(node.left)     
                _postorder(node.right)     
                result.append(node.value) 
    
        _postorder(self.root)
        return result


    def level_order_traversal(self) -> List[int]:
        if not self.root:
            return []
    
        from collections import deque
        queue = deque([self.root])
        result = []
    
        while queue:
            current = queue.popleft()
            result.append(current.value)
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
    
        return result


    def minimum(self) -> TreeNode:
        if not self.root:
            return None
    
        current = self.root
        while current.left:
            current = current.left
    
        return current


    def maximum(self) -> TreeNode:
        if not self.root:
            return None
    
        current = self.root
        while current.right:
            current = current.right
    
        return current

    
    def is_valid_bst(self) -> bool:
        def _is_valid(node, lower=float('-inf'), upper=float('inf')):
            if not node:
                return True
            if not (lower < node.value < upper):
                return False
            return (_is_valid(node.left, lower, node.value) and
                    _is_valid(node.right, node.value, upper))
    
        return _is_valid(self.root)


def insertion_sort(lst: List[int]) -> List[int]:
    for i in range(1, len(lst)):
        key = lst[i]
        j = i - 1
        while j >= 0 and lst[j] > key:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = key
    return lst

def selection_sort(lst: List[int]) -> List[int]:
    for i in range(len(lst)):
        min_idx = i
        for j in range(i + 1, len(lst)):
            if lst[j] < lst[min_idx]:
                min_idx = j
        lst[i], lst[min_idx] = lst[min_idx], lst[i]
    return lst

def bubble_sort(lst: List[int]) -> List[int]:
    for i in range(len(lst)):
        for j in range(0, len(lst) - i - 1):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    return lst


def shell_sort(lst: List[int]) -> List[int]:
    gap = len(lst) // 2
    while gap > 0:
        for i in range(gap, len(lst)):
            temp = lst[i]
            j = i
            while j >= gap and lst[j - gap] > temp:
                lst[j] = lst[j - gap]
                j -= gap
            lst[j] = temp
        gap //= 2
    return lst

def merge_sort(lst: List[int]) -> List[int]:
    if len(lst) <= 1:
        return lst
    mid = len(lst) // 2
    left = merge_sort(lst[:mid])
    right = merge_sort(lst[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(lst: List[int]) -> List[int]:
    if len(lst) <= 1:
        return lst
    pivot = lst[0]
    less = [x for x in lst[1:] if x <= pivot]
    greater = [x for x in lst[1:] if x > pivot]
    return quick_sort(less) + [pivot] + quick_sort(greater)

