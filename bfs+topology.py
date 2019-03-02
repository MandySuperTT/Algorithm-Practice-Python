'''
433. Number of Islands
Given a boolean 2D matrix, 0 is represented as the sea, 1 is represented as the island. If two 1 is adjacent, we consider them in the same island. We only consider up/down/left/right adjacent.

Find the number of islands.

Example
Given graph:

[
  [1, 1, 0, 0, 0],
  [0, 1, 0, 0, 1],
  [0, 0, 0, 1, 1],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1]
]
return 3.
'''
class Solution:
    """
    @param grid: a boolean 2D matrix
    @return: an integer
    """
    def numIslands(self, grid):
        # write your code here
        if grid == [] or grid[0] == []:
            return 0
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]:
                    count += 1 
                    self.bfs(grid,i,j)
        return count
                    
    def bfs(self,grid,x,y):
        queue = collections.deque([[x,y]])
        choose = [(1,0),(-1,0),(0,1),(0,-1)]
        
        while queue:
            [i,j] = queue.popleft()
            grid[i][j] = 0
            for each in choose:
                if self.valid(grid,i+each[0],j+each[1]):
                    queue.append([i+each[0],j+each[1]])
                    grid[i+each[0]][j+each[1]] = 0
                    
    def valid(self,grid,x,y):
        return 0<=x<len(grid) and 0<=y<len(grid[0]) and grid[x][y]
            
                    
'''
69. Binary Tree Level Order Traversal
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

Example
Given binary tree {3,9,20,#,#,15,7},

    3
   / \
  9  20
    /  \
   15   7
 

return its level order traversal as:

[
  [3],
  [9,20],
  [15,7]
]
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
    @param root: A Tree
    @return: Level order a list of lists of integer
    """
    def levelOrder(self, root):
        # write your code here
        
        if root is None:
            return []
        queue = collections.deque([root])
        order = []
        while queue:
            temp = []
            for i in range(len(queue)):
                node = queue.popleft()
                temp += [node.val]
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            order.append(temp)
        return order
            
            
'''
615. Course Schedule
There are a total of n courses you have to take, labeled from 0 to n - 1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?

Example
Given n = 2, prerequisites = [[1,0]]
Return true

Given n = 2, prerequisites = [[1,0],[0,1]]
Return false
'''
class Solution:
    """
    @param: numCourses: a total of n courses
    @param: prerequisites: a list of prerequisite pairs
    @return: true if can finish all courses or false
    """
    def canFinish(self, numCourses, prerequisites):
        # write your code here
        nodes,neibor = self.getnode(prerequisites,numCourses)
        start = [each for each in nodes if nodes[each] == 0]
        if start == []:
            return False
        order = []
        #process = [each for each in nodes if nodes[each] != 0]
        queue = collections.deque(start)
        while queue:
            node = queue.popleft()
            order += [node]
            #print(neibor[node])
            for nei in neibor[node]:
                nodes[nei] -= 1
                if nodes[nei] == 0:
                    queue.append(nei)
        return len(order) == numCourses
        
    def getnode(self,prerequisites,numCourses):
        node = {x:0 for x in range(numCourses)}
        neibor = {nei:[] for nei in range(numCourses)}
        for each in prerequisites:
            neibor[each[1]].append(each[0])
            node[each[0]] += 1 
        return node,neibor


'''
616. Course Schedule II
There are a total of n courses you have to take, labeled from 0 to n - 1.
Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.

There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.

Example
Given n = 2, prerequisites = [[1,0]]
Return [0,1]

Given n = 4, prerequisites = [1,0],[2,0],[3,1],[3,2]]
Return [0,1,2,3] or [0,2,1,3]
'''
class Solution:
    """
    @param: numCourses: a total of n courses
    @param: prerequisites: a list of prerequisite pairs
    @return: the course order
    """
    def findOrder(self, numCourses, prerequisites):
        # write your code here
        neibor = {x:[] for x in range(numCourses)}
        course = {x:0 for x in range(numCourses)}
        
        for each in prerequisites:
            course[each[0]] += 1
            neibor[each[1]].append(each[0])
        
        start = [x for x in range(len(course)) if course[x] == 0]
        queue = collections.deque(start)
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for nei in neibor[node]:
                course[nei] -= 1 
                if course[nei] == 0:
                    queue.append(nei)
        if len(order) == numCourses:
            return order 
        else:
            return []
           

   '''
   611. Knight Shortest Path
Given a knight in a chessboard (a binary matrix with 0 as empty and 1 as barrier) with a source position, find the shortest path to a destination position, return the length of the route.
Return -1 if knight can not reached.

Example
[[0,0,0],
 [0,0,0],
 [0,0,0]]
source = [2, 0] destination = [2, 2] return 2

[[0,1,0],
 [0,0,0],
 [0,0,0]]
source = [2, 0] destination = [2, 2] return 6

[[0,1,0],
 [0,0,1],
 [0,0,0]]
source = [2, 0] destination = [2, 2] return -1
Clarification
If the knight is at (x, y), he can get to the following positions in one step:

(x + 1, y + 2)
(x + 1, y - 2)
(x - 1, y + 2)
(x - 1, y - 2)
(x + 2, y + 1)
(x + 2, y - 1)
(x - 2, y + 1)
(x - 2, y - 1)
'''
"""
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
"""

DIRECTIONS = [
    (-2, -1), (-2, 1), (-1, 2), (1, 2),
    (2, 1), (2, -1), (1, -2), (-1, -2),
]
h = set()
class Solution:
    """
    @param grid: a chessboard included 0 (false) and 1 (true)
    @param source: a point
    @param destination: a point
    @return: the shortest path 
    """
   
    def shortestPath(self, grid, source, destination):
        # write your code here
        h = set()
        count = 0
        i = source.x 
        j = source.y 
        d_i = destination.x 
        d_j = destination.y 
        queue = [[i,j]]
        h.add((i,j))
        #h.add((i-1,j+1))
        #print(h)
        while queue:
            for _ in range(len(queue)):
                node = queue.pop(0)
                #print(node,[d_i,d_j])
                if node == [d_i,d_j]:
                    return count
                    '''
                new_node = self.getnode(grid,node)
                queue = queue + new_node
                '''
                #print(queue)
                choose = [[1,2],[1,-2],[-1,2],[-1,-2],[2,1],[2,-1],[-2,-1],[-2,1]]
                for each in choose:
            #print(Point(source.x))
                    x = node[0] + each[0]
                    y = node[1] + each[1]
                    #print(source[0],each[0])
                    if 0<=x<len(grid) and 0<=y<len(grid[0]) and grid[x][y] == 0 and (x,y) not in h:
                        queue.append([x,y])
                        h.add((x,y))
            count += 1 
        return -1
            
'''
    def getnode(self,grid,source):
        choose = [[1,2],[1,-2],[-1,2],[-1,-2],[2,1],[2,-1],[-2,-1],[-2,1]]
        node = []
        for each in choose:
            #print(Point(source.x))
            x = source[0] + each[0]
            y = source[1] + each[1]
            #print(source[0],each[0])
            if 0<=x<len(grid) and 0<=y<len(grid[0]) and grid[x][y] == 0 and (x,y) not in h:
                node.append([x,y])
                h.add((x,y))
               # print(h)
        return node
        '''

'''
        queue = collections.deque([(source.x, source.y)])
        distance = {(source.x, source.y): 0}

        while queue:
            x, y = queue.popleft()
            if (x, y) == (destination.x, destination.y):
                return distance[(x, y)]
            for dx, dy in DIRECTIONS:
                next_x, next_y = x + dx, y + dy
                if (next_x, next_y) in distance:
                    continue
                if not self.is_valid(next_x, next_y, grid):
                    continue
                distance[(next_x, next_y)] = distance[(x, y)] + 1
                queue.append((next_x, next_y))
        return -1
        
    def is_valid(self, x, y, grid):
        n, m = len(grid), len(grid[0])

        if x < 0 or x >= n or y < 0 or y >= m:
            return False
            
        return not grid[x][y]
        '''

        '''
        605. Sequence Reconstruction
Check whether the original sequence org can be uniquely reconstructed from the sequences in seqs. The org sequence is a permutation of the integers from 1 to n, with 1 ≤ n ≤ 10^4. Reconstruction means building a shortest common supersequence of the sequences in seqs (i.e., a shortest sequence so that all sequences in seqs are subsequences of it). Determine whether there is only one sequence that can be reconstructed from seqs and it is the org sequence.

Example
Given org = [1,2,3], seqs = [[1,2],[1,3]]
Return false
Explanation:
[1,2,3] is not the only one sequence that can be reconstructed, because [1,3,2] is also a valid sequence that can be reconstructed.

Given org = [1,2,3], seqs = [[1,2]]
Return false
Explanation:
The reconstructed sequence can only be [1,2].

Given org = [1,2,3], seqs = [[1,2],[1,3],[2,3]]
Return true
Explanation:
The sequences [1,2], [1,3], and [2,3] can uniquely reconstruct the original sequence [1,2,3].

Given org = [4,1,5,2,6,3], seqs = [[5,2,6,3],[4,1,5,2]]
Return true
'''

class Solution:
    """
    @param org: a permutation of the integers from 1 to n
    @param seqs: a list of sequences
    @return: true if it can be reconstructed only one or false
    """
    def sequenceReconstruction(self, org, seqs):
        # write your code here
        '''
        course = {x:[] for x in org}
        nodes = {x:0 for x in org}
        
        for each in seqs:
            for i in range(1,len(each)):
                nodes[each[i]] += 1 
                course[each[i-1]].append(each[i])
                
        start = [x for x in range(len(nodes)) if nodes[x] == 0]
        queue = collections.deque(start)
        order = []
        while queue:
            node = queue.popleft()
            order += [node]
            for each in course[node]:
                nodes[each] -= 1 
                if nodes[each] == 0:
                    queue.append(each)
        
        if order == org:
            return True
        else:
            False
            '''
        edges = {}
        degrees = {}
        nodes = set()
        for x in org:
            edges[x] = []
            degrees[x] = 0
        for s in seqs:
            nodes |= set(s)
            for i in range(len(s) - 1):
                edges[s[i]].append(s[i+1])
                if s[i+1] in degrees:
                    degrees[s[i+1]] += 1
                else:
                    return False

        # push 0 indegree to queue
        queue = []
        answer = []
        for k, v in degrees.items():
            if v == 0:
                queue.append(k)
        #BFS
        while len(queue) == 1:
            num = queue.pop(0)
            answer.append(num)
            for e in edges[num]:
                degrees[e] -= 1 
                if degrees[e] == 0:
                    queue.append(e)
        return answer == org and len(nodes) == len (org)
            
                
'''
137. Clone Graph
Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors.

How we serialize an undirected graph:

Nodes are labeled uniquely.

We use # as a separator for each node, and , as a separator for node label and each neighbor of the node.

As an example, consider the serialized graph {0,1,2#1,2#2,2}.

The graph has a total of three nodes, and therefore contains three parts as separated by #.

First node is labeled as 0. Connect node 0 to both nodes 1 and 2.
Second node is labeled as 1. Connect node 1 to node 2.
Third node is labeled as 2. Connect node 2 to node 2 (itself), thus forming a self-cycle.
Visually, the graph looks like the following:

   1
  / \
 /   \
0 --- 2
     / \
     \_/
Example
return a deep copied graph.
'''
"""
Definition for a undirected graph node
class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
"""


class Solution:
    """
    @param: node: A undirected graph node
    @return: A undirected graph node
    """
    def cloneGraph(self, node):
        # write your code here
        root = node
        if node is None:
            return node
            
        # use bfs algorithm to traverse the graph and get all nodes.
        nodes = self.getNodes(node)
        
        # copy nodes, store the old->new mapping information in a hash map
        mapping = {}
        for node in nodes:
            mapping[node] = UndirectedGraphNode(node.label)
        
        # copy neighbors(edges)
        for node in nodes:
            new_node = mapping[node]
            for neighbor in node.neighbors:
                new_neighbor = mapping[neighbor]
                new_node.neighbors.append(new_neighbor)
        
        return mapping[root]
        
    def getNodes(self, node):
        q = collections.deque([node])
        result = set([node])
        while q:
            head = q.popleft()
            for neighbor in head.neighbors:
                if neighbor not in result:
                    result.add(neighbor)
                    q.append(neighbor)
        return result


'''
127. Topological Sorting
Given an directed graph, a topological order of the graph nodes is defined as follow:

For each directed edge A -> B in graph, A must before B in the order list.
The first node in the order can be any node in the graph with no nodes direct to it.
Find any topological order for the given graph.

Example
For graph as follow:

picture

The topological order can be:

[0, 1, 2, 3, 4, 5]
[0, 2, 3, 1, 5, 4]
...
'''
"""
Definition for a Directed graph node
class DirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
"""


class Solution:
    """
    @param: graph: A list of Directed graph node
    @return: Any topological order for the given graph.
    """
    def topSort(self, graph):
        # write your code here
        indegree_node = self.indegree(graph)
        #bfs
        order = []
        start = [x for x in indegree_node if indegree_node[x] == 0]
        queue = collections.deque(start)
        while queue:
            node = queue.popleft()
            order += [node]
            for nei in node.neighbors:
                indegree_node[nei] -= 1 
                if indegree_node[nei] == 0:
                    queue.append(nei)
        return order
            
        
    def indegree(self,graph):
        node = {x:0 for x in graph}
        
        for nodes in graph:
            print('node',node)
            for nei in nodes.neighbors:
                print(nei)
                node[nei] += 1 
        
        return node 


'''
7. Serialize and Deserialize Binary Tree
Design an algorithm and write code to serialize and deserialize a binary tree. Writing the tree to a file is called 'serialization' and reading back from the file to reconstruct the exact same binary tree is 'deserialization'.

Example
An example of testdata: Binary tree {3,9,20,#,#,15,7}, denote the following structure:

  3
 / \
9  20
  /  \
 15   7
Our data serialization use bfs traversal. This is just for when you got wrong answer and want to debug the input.

You can use other method to do serializaiton and deserialization.

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
    @param root: An object of TreeNode, denote the root of the binary tree.
    This method will be invoked first, you should design your own algorithm 
    to serialize a binary tree which denote by a root node to a string which
    can be easily deserialized by your own "deserialize" method later.
    """
    def serialize(self, root):
        # write your code here
        if not root:
            return ['#']
        q = [root]
        ans = []
        while q:
            temp = q.pop(0)
            if not temp:
                ans.append('#')
            else:
                ans.append(str(temp.val))
                q.append(temp.left)
                q.append(temp.right)
        return ans
                
            

    """
    @param data: A string serialized by your serialize method.
    This method will be invoked second, the argument data is what exactly
    you serialized at method "serialize", that means the data is not given by
    system, it's given by your own serialize method. So the format of data is
    designed by yourself, and deserialize it here as you serialize it in 
    "serialize" method.
    """
    def deserialize(self, data):
        # write your code here
        if data[0] == '#':
            return None
        root = TreeNode(int(data.pop(0)))
        q = [root]
        isLeft = True
        while data:
            ch = data.pop(0)
            if ch != '#':
                node = TreeNode(int(ch))
                q.append(node)
                if isLeft:
                    q[0].left = node
                else:
                    q[0].right = node
            if not isLeft:
                q.pop(0)
            isLeft = not isLeft
        return root


'''
120. Word Ladder
Given two words (start and end), and a dictionary, find the length of shortest transformation sequence from start to end, such that:

Only one letter can be changed at a time
Each intermediate word must exist in the dictionary
Example
Given:
start = "hit"
end = "cog"
dict = ["hot","dot","dog","lot","log"]
As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
return its length 5.

'''
class Solution:
    """
    @param: start: a string
    @param: end: a string
    @param: dict: a set of string
    @return: An integer
    """
    def ladderLength(self, start, end, dict):
        # write your code here
        dict.add(end)
        queue = collections.deque([start])
        #print(queue)
        visited = set([start])
        #print(visited,[start])
        distance = 0
        while queue:
            distance += 1
            for i in range(len(queue)):
                word = queue.popleft()
                if word == end:
                    return distance
                
                for next_word in self.get_next_words(word):
                    if next_word not in dict or next_word in visited:
                        continue
                    queue.append(next_word)
                    visited.add(next_word) 

        return 0
        
    # O(26 * L^2)
    # L is the length of word
    def get_next_words(self, word):
        words = []
        for i in range(len(word)):
            left, right = word[:i], word[i + 1:]
            for char in 'abcdefghijklmnopqrstuvwxyz':
                if word[i] == char:
                    continue
                words.append(left + char + right)
        #print(words)
        return words
