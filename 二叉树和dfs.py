# Binary Tree Preorder Traversal
# Given a binary tree, return the preorder traversal of its nodes' values.
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: A Tree
    @return: Preorder in ArrayList which contains node values.
    """
    def preorderTraversal(self, root):
        # write your code here
        self.results = []
        self.traverse(root)
        return self.results
        
    def traverse(self, root):
        if root is None:
            return
        self.results.append(root.val)
        self.traverse(root.left)
        self.traverse(root.right)


# Binary Tree Inorder Traversal
# Given a binary tree, return the inorder traversal of its nodes' values.

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: A Tree
    @return: Inorder in ArrayList which contains node values.
    """
    def inorderTraversal(self, root):
        # write your code here
        self.result = []
        self.traversal(root)
        return self.result
    def traversal(self,root):
        if root is None:
            return
        self.traversal(root.left)
        self.result.append(root.val)
        self.traversal(root.right)

# Binary Tree Postorder Traversal
# Given a binary tree, return the postorder traversal of its nodes' values.
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: A Tree
    @return: Postorder in ArrayList which contains node values.
    """
    def postorderTraversal(self, root):
        # write your code here
        self.result = []
        self.traversal(root)
        return self.result
        
    def traversal(self,root):
        if root is None:
            return
        self.traversal(root.left)
        self.traversal(root.right)
        self.result.append(root.val)


# Maximum Depth of Binary Tree
# Given a binary tree, find its maximum depth.
# The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """
    def maxDepth(self, root):
        # write your code here
        '''
        if root is None:
            return 0 
        
        l = self.maxDepth(root.left)
        r = self.maxDepth(root.right)
        
        return max(l,r) + 1 
        '''
        self.depth = 0
        self.traversal(root,1)
        return self.depth
        
    def traversal(self,root,curdepth):
        if root is None: return
        self.depth = max(self.depth,curdepth)
        self.traversal(root.left,curdepth+1)
        self.traversal(root.right,curdepth+1)


# Balanced Binary Tree
# Given a binary tree, determine if it is height-balanced.
# For this problem, a height-balanced binary tree is defined as a binary tree 
# in which the depth of the two subtrees of every node never differ by more than 1.
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """
    def isBalanced(self, root):
        # write your code here
        self.balance = True
        self.balance,_ = self.valid(root)
        return self.balance 
    
    def valid(self,root):
        if root is None:
            return True,0
        
        self.banlance,left = self.valid(root.left)
        if self.banlance == False:
            return False,0
        self.banlance,right = self.valid(root.right)
        if self.banlance == False:
            return False,0
        
        return abs(left-right)<=1, max(left,right) + 1   


# Validate Binary Search Tree
# Given a binary tree, determine if it is a valid binary search tree (BST).
'''
Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
A single node tree is a BST
'''
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of binary tree.
    @return: True if the binary tree is BST, or false
    """
    def isValidBST(self, root):
        # write your code here
        self.result = []
        self.valid = True
        self.inorder(root)
        return self.valid
    def inorder(self,root):
        if root is None:
            return
        self.inorder(root.left)
        if self.result == [] or self.result[-1] < root.val:
            self.result.append(root.val)
        else:
            self.valid = False
            return
        self.inorder(root.right)

'''
Subtree with Maximum Average
Given a binary tree, find the subtree with maximum average. Return the root of the subtree.
'''
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the maximum average of subtree
    """

    average, node = 0, None

    def findSubtree2(self, root):
        # Write your code here
        self.helper(root)
        return self.node

    def helper(self, root):
        if root is None:
            return 0, 0

        left_sum, left_size = self.helper(root.left)
        right_sum, right_size = self.helper(root.right)

        sum, size = left_sum + right_sum + root.val, \
                    left_size + right_size + 1

        if self.node is None or sum * 1.0 / size > self.average:
            self.node = root
            self.average = sum * 1.0 / size

        return sum, size

'''
 Invert Binary Tree
Invert a binary tree.
'''
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def invertBinaryTree(self, root):
        # write your code here
        if root is None:
            return
        self.invertBinaryTree(root.left)
        self.invertBinaryTree(root.right)
        root.left,root.right = root.right,root.left
        return root

'''
Minimum Subtree
Given a binary tree, find the subtree with minimum sum. Return the root of the subtree.
'''
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    
    sum,node = 100,None
    """
    @param root: the root of binary tree
    @return: the root of the minimum subtree
    """
    def findSubtree(self, root):
        # write your code here
        self.helper(root)
       # print(self.node.val)
        return self.node
        
    def helper(self,root):
        if root is None:
            return 0
        leftsum = self.helper(root.left)
        rightsum = self.helper(root.right)
        a = leftsum+rightsum+root.val

        if self.node is None or self.sum > a:
            self.node = root
            self.sum = leftsum+rightsum+root.val
            #print(self.sum)
            
        return a

'''
 Binary Tree Paths
 Given a binary tree, return all root-to-leaf paths.
'''
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of the binary tree
    @return: all root-to-leaf paths
    """
    def binaryTreePaths(self, root):
        if root is None:
            return []
            
        if root.left is None and root.right is None:
            return [str(root.val)]

        paths = []
        for path in self.binaryTreePaths(root.left):
            #print(path)
            paths.append(str(root.val) + '->' + path)
        
        for path in self.binaryTreePaths(root.right):
            #print(path)
            paths.append(str(root.val) + '->' + path)
        #print(paths)   
        return paths

'''
Lowest Common Ancestor of a Binary Tree
Given the root and two nodes in a Binary Tree. Find the lowest common ancestor(LCA) of the two nodes.

The lowest common ancestor is the node with largest depth which is the ancestor of both nodes.
'''
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: A: A TreeNode in a Binary.
    @param: B: A TreeNode in a Binary.
    @return: Return the least common ancestor(LCA) of the two nodes.
    """
    node,d = None,100
    def lowestCommonAncestor(self, root, A, B):
        
        if root is None:
            return None
            
        if root is A or root is B:
            return root
            
        left = self.lowestCommonAncestor(root.left, A, B)
        right = self.lowestCommonAncestor(root.right, A, B)
        
        if left is not None and right is not None:
            return root
        if left is not None:
            return left
        if right is not None:
            return right
        return None

'''
Flatten Binary Tree to Linked List
Flatten a binary tree to a fake "linked list" in pre-order traversal.

Here we use the right pointer in TreeNode as the next pointer in ListNode.

'''
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def flatten(self, root):
        # write your code here
        self.result = []
        self.preorder(root)
        #self.result.sort()
        if len(self.result) == 0:
            return {}
        for i in range(len(self.result)-1):
            self.result[i].left,self.result[i].right = None,self.result[i+1]
        
        return self.result[0]
        
    def preorder(self,root):
        if root is None:
            #self.result.append('#')
            return 
        self.result.append(root)
        self.preorder(root.left)
        self.preorder(root.right)
    

'''
Kth Smallest Element in a BST
Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.
'''
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the given BST
    @param k: the given k
    @return: the kth smallest element in BST
    """
    def kthSmallest(self, root, k):
        # write your code here
        self.result = []
        self.inorder(root)
        return self.result[k-1]
        
    def inorder(self,root):
        if root is None:
            return
        self.inorder(root.left)
        self.result.append(root.val)
        self.inorder(root.right)

# Version 1. Count Children of each node. 
# O(n) of time. 
class Solution:
    """
    @param root: the given BST
    @param k: the given k
    @return: the kth smallest element in BST
    """
    def kthSmallest(self, root, k):
        # write your code here
        nChildren = {}
        self.countChildren(root, nChildren)
        return self.quickSelect(root, k, nChildren)
    
    def countChildren(self, root, nChildren):
        
        if root is None: return 0
        left = self.countChildren(root.left, nChildren)
        right = self.countChildren(root.right, nChildren)
        
        # using the node as the key, instead of using the node value. in case of conflict. 
        nChildren[root] = left + right + 1
        
        return left + right + 1
        
    def quickSelect(self, root, k, nChildren):
        
        if root.left: 
            left = nChildren[root.left]
        else:
            left = 0
        
        if left >= k:
            return self.quickSelect(root.left, k, nChildren)
        elif left + 1 == k:
            return root.val
        else:
            return self.quickSelect(root.right, k-left-1, nChildren)

'''
Binary Search Tree Iterator
Design an iterator over a binary search tree with the following rules:

Elements are visited in ascending order (i.e. an in-order traversal)
next() and hasNext() queries run in O(1) time in average.
'''
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

Example of iterate a tree:
iterator = BSTIterator(root)
while iterator.hasNext():
    node = iterator.next()
    do something for node 
"""


class BSTIterator:
    """
    @param: root: The root of binary tree.
    """
    def __init__(self, root):
        self.stack = []
        while root != None:
            self.stack.append(root)
            root = root.left

    """
    @return: True if there has next node, or false
    """
    def hasNext(self):
        return len(self.stack) > 0

    """
    @return: return next node
    """
    def next(self):
        node = self.stack[-1]
        if node.right is not None:
            n = node.right
            while n != None:
                self.stack.append(n)
                n = n.left
        else:
            n = self.stack.pop()
            while self.stack and self.stack[-1].right == n:
                n = self.stack.pop()
        
        return node


'''
Closest Binary Search Tree Value
Given a non-empty binary search tree and a target value, find the value in the BST that is closest to the target.
'''
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the given BST
    @param target: the given target
    @return: the value in the BST that is closest to the target
    """
    node = None
    def closestValue(self, root, target):
        # write your code here
        self.m = target
        self.inorder(root,target)
        
        return self.node
        
    def inorder(self,root,target):
        if root is None:
            return
        self.inorder(root.left,target)
        #print(abs(root.val-target),self.m)
        if abs(root.val-target) < self.m:
            self.m = abs(root.val-target)
            self.node = root.val
        self.inorder(root.right,target)


class Solution:
    """
    @param root: the given BST
    @param target: the given target
    @param k: the given k
    @return: k values in the BST that are closest to the target
    """
    def closestKValues(self, root, target, k):
        # write your code here
        prev = []
        next = []
        
        while root:
            if root.val > target:
                next.append(root)
                root = root.left
            elif root.val < target:
                prev.append(root)
                root = root.right
            else:
                next.append(root)
                break
        
        res = []
        while k:
            next_val = sys.maxsize if len(next) == 0 else abs(next[-1].val - target)
            prev_val = sys.maxsize if len(prev) == 0 else abs(prev[-1].val - target)
            
            if next_val == sys.maxsize and prev_val == sys.maxsize:
                break
            if next_val < prev_val:
                res.append(next[-1].val)
                self.getNext(next)
                k -= 1
            else:
                res.append(prev[-1].val)
                self.getPrev(prev)
                k -= 1
        
        return res
    
    def getNext(self, next):
        node = next.pop()
        node = node.right
        while node:
            next.append(node)
            node = node.left
    
    def getPrev(self, prev):
        node = prev.pop()
        node = node.left
        while node:
            prev.append(node)
