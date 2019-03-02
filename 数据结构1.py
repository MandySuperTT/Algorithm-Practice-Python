'''
 Implement Three Stacks by Single Array
Implement three stacks by single array.

You can assume the three stacks has the same size and big enough, you don't need to care about how to extend it if one of the stack is full.
'''
class ThreeStacks:
    """
    @param: size: An integer
    """
    def __init__(self, size):
        # do intialization if necessary
        self.size = size
        self.stack = [[],[],[]]

    """
    @param: stackNum: An integer
    @param: value: An integer
    @return: nothing
    """
    def push(self, stackNum, value):
        # Push value into stackNum stack
        self.stack[stackNum].append(value)

    """
    @param: stackNum: An integer
    @return: the top element
    """
    def pop(self, stackNum):
        # Pop and return the top element from stackNum stack
        return self.stack[stackNum].pop()

    """
    @param: stackNum: An integer
    @return: the top element
    """
    def peek(self, stackNum):
        # Return the top element
        return self.stack[stackNum][-1]

    """
    @param: stackNum: An integer
    @return: true if the stack is empty else false
    """
    def isEmpty(self, stackNum):
        # write your code here
        return len(self.stack[stackNum]) == 0



 '''
 Implement Queue by Circular Array
Implement queue by circulant array. You need to support the following methods:

CircularQueue(n): initialize a circular array with size n to store elements
boolean isFull(): return true if the array is full
boolean isEmpty(): return true if there is no element in the array
void enqueue(element): add an element to the queue
int dequeue(): pop an element from the queue
'''
class CircularQueue:
    def __init__(self, n):
        # do intialization if necessary
        self.size = n
        self.queue = []
    """
    @return:  return true if the array is full
    """
    def isFull(self):
        # write your code here
        return len(self.queue) == self.size

    """
    @return: return true if there is no element in the array
    """
    def isEmpty(self):
        # write your code here
        return len(self.queue) == 0

    """
    @param element: the element given to be added
    @return: nothing
    """
    def enqueue(self, element):
        # write your code here
        self.queue.append(element)

    """
    @return: pop an element from the queue
    """
    def dequeue(self):
        # write your code here
        return self.queue.pop(0)

'''
Moving Average from Data Stream
Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.
'''
class MovingAverage:
    """
    @param: size: An integer
    """
    def __init__(self, size):
        # do intialization if necessary
        self.size = size
        self.queue = []
        self.add = 0

    """
    @param: val: An integer
    @return:  
    """
    def next(self, val):
        # write your code here
        self.queue.append(val)
        self.add += val
        if len(self.queue) < self.size:
            return self.add/len(self.queue)
        elif len(self.queue) > self.size:
            self.add -= self.queue.pop(0)
            return self.add/len(self.queue)
        else:
            return self.add/self.size


# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param = obj.next(val)


'''
LRU Cache
Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and set.

get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
set(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.
'''
from collections import OrderedDict

class LRUCache:
    """
    @param: capacity: An integer
    """
    def __init__(self, capacity):
        # do intialization if necessary
        self.capacity = capacity
        self.cache = OrderedDict()

    """
    @param: key: An integer
    @return: An integer
    """
    def get(self, key):
        # write your code here
        if key not in self.cache:
            return -1
        ## pop value and insert to the bottom of queue
        value = self.cache.pop(key)
        self.cache[key] = value
        return value
        
        

    """
    @param: key: An integer
    @param: value: An integer
    @return: nothing
    """
    def set(self, key, value):
        # write your code here
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) == self.capacity:
            ## last = True时pop规则为FILO, last = False时pop规则为FIFO
            self.cache.popitem(last = False)
        self.cache[key] = value


'''
Insert Delete GetRandom O(1)
Design a data structure that supports all following operations in average O(1) time.

insert(val): Inserts an item val to the set if not already present.
remove(val): Removes an item val from the set if present.
getRandom: Returns a random element from current set of elements. Each element must have the same probability of being returned.
'''
from random import choice
class RandomizedSet:
    '''
    def __init__(self):
        # do intialization if necessary
        self.d = {}

    """
    @param: val: a value to the set
    @return: true if the set did not already contain the specified element or false
    """
    def insert(self, val):
        # write your code here
        if val in self.d:
            return False
        self.d[val] =val
        #print(self.d)
        return True

    """
    @param: val: a value from the set
    @return: true if the set contained the specified element or false
    """
    def remove(self, val):
        # write your code here
        if val not in self.d:
            return False 
        del self.d[val]
        #print(self.d)
        return True

    """
    @return: Get a random element from the set
    """
    def getRandom(self):
        # write your code here
        value = self.d.popitem()[0]
        self.d[value] = value
        return value
        


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param = obj.insert(val)
# param = obj.remove(val)
# param = obj.getRandom()
'''
    def __init__(self):
        self.s = set()
    def insert(self, n):
        self.s.add(n)
    def remove(self, n):
        self.s.discard(n)
    def getRandom(self):
        n = self.s.pop()
        self.s.add(n)
        return n


'''
Insert Delete GetRandom O(1) - Duplicates allowed
Design a data structure that supports all following operations in average O(1) time.
'''
class RandomizedCollection(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.num = {}
        self.l = []
        

    def insert(self, val):
        """
        Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        self.l.append(val)
        #if val in self.l:
         #   return False 
        #return True
        

    def remove(self, val):
        """
        Removes a value from the collection. Returns true if the collection contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.l:
            self.l.remove(val)
        #return False
        
        
        

    def getRandom(self):
        """
        Get a random element from the collection.
        :rtype: int
        """
        from random import choice
        
        return choice(self.l)
        


# Your RandomizedCollection object will be instantiated and called as such:
# obj = RandomizedCollection()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

'''
First Unique Character in a String
Find the first unique character in a given string. You can assume that there is at least one unique character in the string.

'''
class Solution:
    """
    @param str: str: the given string
    @return: char: the first unique character in a given string
    """
    def firstUniqChar(self, str):
        # Write your code here
        
        num = {}
        for each in str:
            if each not in num and str.count(each) == 1:
                return each
            num[each] = 1
        
        '''
        counter = {}

        for c in str:
            counter[c] = counter.get(c, 0) + 1

        for c in str:
            if counter[c] == 1:
                return c
                '''
            
'''
First Unique Number in a Stream II
We need to implement a data structure named DataStream. There are two methods required to be implemented:

void add(number) // add a new number
int firstUnique() // return first unique number
'''
class DataStream:

    def __init__(self):
        # do intialization if necessary
        self.d={}
    """
    @param num: next number in stream
    @return: nothing
    """
    def add(self, num):
        # write your code here
        if num not in self.d:
            self.d[num] = 1 
        else:
            self.d[num] += 1 

    """
    @return: the first unique number in stream
    """
    def firstUnique(self):
        # write your code here
        print(self.d)
        for each in self.d:
            if self.d[each] == 1:
                return each


'''
Subarray Sum
Given an integer array, find a subarray where the sum of numbers is zero. Your code should return the index of the first number and the index of the last number.
'''
class Solution:
    """
    @param nums: A list of integers
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def subarraySum(self, nums):
        # write your code here
        index = {0:-1}
        s = 0
        for i,each in enumerate(nums):
            s += each
            if s in index:
                return index[s] + 1,i 
            index[s] = i 
        return -1,-1


'''
Copy List with Random Pointer
A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.

Return a deep copy of the list.
'''
"""
Definition for singly-linked list with a random pointer.
class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None
"""


class Solution:
    # @param head: A RandomListNode
    # @return: A RandomListNode
    def copyRandomList(self, head):
        # write your code here
        if head == None:
            return None
            
        myMap = {}
        nHead = RandomListNode(head.label)
        myMap[head] = nHead
        p = head
        q = nHead
        while p != None:
            q.random = p.random
            if p.next != None:
                q.next = RandomListNode(p.next.label)
                myMap[p.next] = q.next
            else:
                q.next = None
            p = p.next
            q = q.next
        
        p = nHead
        while p!= None:
            if p.random != None:
                p.random = myMap[p.random]
            p = p.next
        return nHead


'''
 Longest Consecutive Sequence
Given an unsorted array of integers, find the length of the longest consecutive elements sequence.
'''
class Solution:
    """
    @param num: A list of integers
    @return: An integer
    """
    def longestConsecutive(self, num):
        # write your code here
        if len(num) == 0:
            return 0
        num.sort()
        #num = set(num)
        helper = 0
        
        i=0
        while i < len(num):
            count = 0
            while i < len(num)-1:
                #count = 0
                if num[i] == num[i+1]:
                    i+=1
                    continue
                elif num[i] + 1 == num[i+1]:
                   # print(num[i])
                    count += 1 
                    i += 1
                    print(count)
                else:
                    break
               
            helper = max(helper,count+1)
            i+=1 
        return helper
                
            
            
'''
Ugly Number II
Ugly number is a number that only have factors 2, 3 and 5.

Design an algorithm to find the nth ugly number. The first 10 ugly numbers are 1, 2, 3, 4, 5, 6, 8, 9, 10, 12...
'''
import heapq
class Solution:
    """
    @param n: An integer
    @return: the nth prime number as description.
    """
    def nthUglyNumber(self, n):
        # write your code here
        heap = [1]
        visited = set([1])
        
        val = None
        for i in range(n):
            val = heapq.heappop(heap)
            for multi in [2, 3, 5]:
                if val * multi not in visited:
                    visited.add(val * multi)
                    heapq.heappush(heap, val * multi)
            print(heap)
            
        return val


'''
Merge K Sorted Lists
Merge k sorted linked lists and return it as one sorted list.

Analyze and describe its complexity.

'''
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""
class Solution:
    """
    @param lists: a list of ListNode
    @return: The head of one sorted list.
    """
    '''
    def mergeKLists(self, lists):
        # write your code here
        self.heap = [[i, lists[i].val] for i in range(len(lists)) if lists[i] != None]
        self.hsize = len(self.heap)
        for i in range(self.hsize - 1, -1, -1):
            self.adjustdown(i)
        nHead = ListNode(0)
        head = nHead
        while self.hsize > 0:
            ind, val = self.heap[0][0], self.heap[0][1]
            head.next = lists[ind]
            head = head.next
            lists[ind] = lists[ind].next
            if lists[ind] is None:
                self.heap[0] = self.heap[self.hsize-1]
                self.hsize = self.hsize - 1
            else:
                self.heap[0] = [ind, lists[ind].val]
            self.adjustdown(0)
        return nHead.next

    def adjustdown(self, p):
        lc = lambda x: (x + 1) * 2 - 1
        rc = lambda x: (x + 1) * 2
        while True:
            np, pv = p, self.heap[p][1]
            if lc(p) < self.hsize and self.heap[lc(p)][1] < pv:
                np, pv = lc(p), self.heap[lc(p)][1]
            if rc(p) < self.hsize and self.heap[rc(p)][1] < pv:
                np = rc(p)
            if np == p:
                break
            else:
                self.heap[np], self.heap[p] = self.heap[p], self.heap[np]
                p = np
    '''
    def mergeKLists(self, lists):
        # write your code here

        if len(lists) < 1:
            return None
            
        return self.helper(lists, 0, len(lists) - 1)
            
    def helper(self,lists, start, end):
        if start == end:
            return lists[start]
            
        mid = start + (end - start) / 2
        left = self.helper(lists, start, mid)
        right = self.helper(lists, mid + 1, end)
        
        return self.mergeTwolist(left, right)
        
    
    def mergeTwolist(self, list1, list2):
        dummy = tail = ListNode(None)
        
        while list1 and list2:
            if list1.val < list2.val:
                tail.next = list1
                tail = tail.next
                list1 = list1.next
            else:
                tail.next = list2
                tail = tail.next
                list2 = list2.next
                
        if list1:
            tail.next = list1
        else:
            tail.next = list2
        
        return dummy.next



#  解法2： 堆！堆！堆！堆！堆！堆！
class Solution:
    """
    @param lists: a list of ListNode
    @return: The head of one sorted list.
    """
    def mergeKLists(self, lists):
        # write your code here
        if len(lists) < 1:
            return None
        heap = []
        dummy = tail = ListNode(-1)
        
        for node in lists:
            if node:
                heappush(heap, (node.val, node))
        
        while len(heap) > 0:
            curr = heappop(heap)[1]
            tail.next = curr
            tail = tail.next
            if curr.next:
                heappush(heap, (curr.next.val,curr.next))
        
        return dummy.next
        
'''
K Closest Points
Given some points and an origin point in two-dimensional space, find k points which are nearest to the origin.
Return these points sorted by distance, if they are same in distance, sorted by the x-axis, and if they are same in the x-axis, sorted by y-axis.
'''
import heapq
"""
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
"""

class Solution:
    """
    @param points: a list of points
    @param origin: a point
    @param k: An integer
    @return: the k closest points
    """
    def kClosest(self, points, origin, k):
        # write your code here
        
        if not points:
            return

        heap = []
        for point in points:
            distance = self.get_distance(origin, point)
            heapq.heappush(heap, (-distance, -point.x,-point.y))
            print(heap)

            if len(heap) > k:
                heapq.heappop(heap)

        ret = []
        while len(heap) > 0:
            _, x, y = heapq.heappop(heap)
            ret.append(Point(-x, -y))

        ret.reverse()
        return ret

    def get_distance(self, a, b):

        return (a.x - b.x)**2 + (a.y - b.y)**2
            
'''
Top k Largest Numbers II
Implement a data structure, provide two interfaces:

add(number). Add a new number in the data structure.
topk(). Return the top k largest numbers in this data structure. k is given when we create the data structure.
'''
import heapq
class Solution:
    """
    @param: k: An integer
    """
    def __init__(self, k):
        # do intialization if necessary
        self.size = k 
        self.heap = []

    """
    @param: num: Number to be added
    @return: nothing
    """
    def add(self, num):
        # write your code here
        heapq.heappush(self.heap, num)

    """
    @return: Top k element
    """
    def topk(self):
        # write your code here
        return heapq.nlargest(self.size, self.heap)


'''
Implement Queue by Two Stacks
As the title described, you should only use two stacks to implement a queue's actions.

The queue should support push(element), pop() and top() where pop is pop the first(a.k.a front) element in the queue.

Both pop and top methods should return the value of first element.

'''

class MyQueue:
    
    def __init__(self):
        # do intialization if necessary
        self.head = []

    """
    @param: element: An integer
    @return: nothing
    """
    def push(self, element):
        # write your code here
       
        self.head.append(element)

    """
    @return: An integer
    """
    def pop(self):
        # write your code here
        
        i = self.head.pop(0)
        return i

    """
    @return: An integer
    """
    def top(self):
        # write your code here
        return self.head[0]

