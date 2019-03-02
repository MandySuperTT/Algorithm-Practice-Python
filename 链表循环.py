#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 00:18:26 2018

@author: xuexudong
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 21:08:12 2018

@author: xuexudong
"""

class Node(object):
    def __init__(self,elem):
        self.elem = elem
        self.next = None
        
    
class SingleLinkList(object):
    def __init__(self,node=None):
        self._head = node
        if node:
            node.next = node
        
    def is_empty(self):
        return self._head == None
    
    def length(self):
        if self.is_empty():
            return 0
        cur = self._head
        count = 1
        while cur.next != self._head:
            count += 1
            cur = cur.next
        return count   
      
    def travel(self):
        if self.is_empty():
            return 
        cur = self._head
        while cur.next != self._head:
            print(cur.elem)
            cur = cur.next
        print(cur.elem)

    def add(self,elem):
        node = Node(elem)
        if self.is_empty():
            self._head = node
            node.next = node
            return
        
        cur = self._head
        while cur.next != self._head:
            cur = cur.next
        node.next = self._head
        self._head = node
        cur.next = node
        
        
    def append(self,elem):
        node = Node(elem)
        if self.is_empty():
            self._head = node
            node.next = node
            return
        else:    
            cur = self._head
            while cur.next != self._head:
                cur = cur.next
            cur.next = node
            node.next = self._head
    def insert(self,pos,elem):       
        if pos <= 0:
            self.add(elem)
        elif pos > (self.length() -1):
            self.append(elem)
        else:
            prior = self._head
            count = 0
            while count < (pos - 1):
                prior = prior.next
                count += 1
            node = Node(elem)
            node.next = prior.next
            prior.next = node
            
        
    def remove(self,item):
        if self.is_empty():
            return None
        cur = self._head
        pre = None
        while cur.next != self._head:
            if cur.elem == item:
                if cur == self._head:
                    rear = self._head
                    while rear.next != self._head:
                        rear = rear.next
                    self._head = cur.next
                    rear.next = self._head
                    #self._head = cur.next              
                else:
                    pre.next = cur.next
                return
            else:
                pre = cur
                cur = cur.next
         if cur.elem == item: 
             if cur == self._head:
                 self._head = None
             else:
                 pre.next = cur.next
            
        
    def search(self,elem):
        if self.is_empty():
            return None
        cur = self._head
        
        while cur.next != self._head:
            if cur.elem == elem:
                return True
            else:
                cur = cur.next
        if cur.elem == elem:
            return True
        return None
if __name__ == "__main__":
    ll = SingleLinkList()
    print(ll.is_empty())
    print(ll.length())
    
    ll.append(1)
    print(ll.is_empty())
    print(ll.length())
    
    ll.append(2)
    ll.append(3)
    ll.add(8)
    ll.append(4)
    ll.append(5)
    ll.append(6)
    ll.append(7)
    ll.travel()
    ll.insert(-1,9)
    ll.travel()
    ll.insert(4,10)
    ll.travel()
    ll.insert(20,11)
    ll.travel()
    print('')
    ll.remove(9)
    ll.travel()
    ll.remove(11)
    ll.travel()

    