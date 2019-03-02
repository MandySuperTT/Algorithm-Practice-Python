'''
115. Unique Paths II
中文English
Follow up for "Unique Paths":

Now consider if some obstacles are added to the grids. How many unique paths would there be?

An obstacle and empty space is marked as 1 and 0 respectively in the grid.

Example
Example 1:
	Input: [[0]]
	Output: 1


Example 2:
	Input:  [[0,0,0],[0,1,0],[0,0,0]]
	Output: 2
	
	Explanation:
	Only 2 different path.
	

Notice
m and n will be at most 100.
'''
class Solution:
    """
    @param obstacleGrid: A list of lists of integers
    @return: An integer
    """
    def uniquePathsWithObstacles(self, obstacleGrid):
        # write your code here
       
        m,n = len(obstacleGrid),len(obstacleGrid[0])
        grid = [[0 for _ in range(n)] for _ in range(m)]
        grid[0][-1] = 1
        for i in range(m):
            for j in range(n):
                if i == 0:
                    if obstacleGrid[i][j] == 0 and grid[i][j-1] != 0:
                        grid[i][j] = 1 
                    else:
                        grid[i][j] = 0
                    #print('1',grid[i][j])
                elif j == 0:
                    if obstacleGrid[i][j] != 0 or grid[i-1][j] == 0:
                        grid[i][j] = 0
                    else:
                        grid[i][j] = 1
                    #print('2',grid[i][j])
                else:
                    if obstacleGrid[i][j] == 0:
                        grid[i][j] = grid[i-1][j] + grid[i][j-1]
                    else:
                        grid[i][j] = 0
                    #print(i,j,grid[i][j])
        #print(grid[-1])
        return grid[-1][-1]

'''
114. Unique Paths
中文English
A robot is located at the top-left corner of a m x n grid.

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid.

How many possible unique paths are there?

Example
Example 1:
	Input: n = 1, m = 3
	Output: 1
	
	Explanation:
	Only one path to target position.

Example 2:
	Input:  n = 3, m = 3
	Output: 6
	
	Explanation:
	D : Down
	R : Right
	1) DDRR
	2) DRDR
	3) DRRD
	4) RRDD
	5) RDRD
	6) RDDR
Notice
m and n will be at most 100.


'''
class Solution:
    """
    @param m: positive integer (1 <= m <= 100)
    @param n: positive integer (1 <= n <= 100)
    @return: An integer
    """
    """
    斜对角相加（左下+右上）
    sum over left-bottom and top-right
    """
    def uniquePaths(self, m, n):
        # write your code here
        matrix = [[1 for _ in range(n)] for _ in range(m)]
        for a in range(1,m):
            for b in range(1,n):
                matrix[a][b] = matrix[a-1][b] + matrix[a][b-1]
        return matrix[m-1][n-1]


'''
111. Climbing Stairs
中文English
You are climbing a stair case. It takes n steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Example
Example 1:
	Input:  n = 3
	Output: 3
	
	Explanation:
	1) 1, 1, 1
	2) 1, 2
	3) 2, 1
	total 3.


Example 2:
	Input:  n = 1
	Output: 1
	
	Explanation:  
	only 1 way.

'''
#import numpy as np
class Solution:
    """
    @param n: An integer
    @return: An integer
    """
    
    def climbStairs(self, n):
        # write your code here
        if n == 0:
            return 0
        if n == 1:
            return 1 
        if n == 2:
            return 2 
        opt = [0]*3
        opt[0] = 1 
        opt[1] = 2
        for i in range(2,n):
            opt[i%3] = opt[(i-1)%3] + opt[(i-2)%3]
        return opt[(n-1)%3]

'''
110. Minimum Path Sum
中文English
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

Example
Example 1:
	Input:  [[1,3,1],[1,5,1],[4,2,1]]
	Output: 7
	
	Explanation:
	Path is: 1 -> 3 -> 1 -> 1 -> 1


Example 2:
	Input:  [[1,3,2]]
	Output: 6
	
	Explanation:  
	Path is: 1 -> 3 -> 2

Notice
ou can only go right or down in the path..


'''
class Solution:
    """
    @param grid: a list of lists of integers
    @return: An integer, minimizes the sum of all numbers along its path
    """
    def minPathSum(self, grid):
        # write your code here
        for a in range(len(grid)):
            for b in range(len(grid[0])):
                if a == 0 and b !=0:
                    grid[a][b] += grid[a][b-1]
                    
                elif b == 0 and a != 0:
                    grid[a][b] += grid[a-1][b]
                    
                elif a != 0 and b != 0:
                    grid[a][b] += min(grid[a-1][b],grid[a][b-1]) 
                    
        return grid[-1][-1]

'''
272. Climbing Stairs II
中文English
A child is running up a staircase with n steps, and can hop either 1 step, 2 steps, or 3 steps at a time. Implement a method to count how many possible ways the child can run up the stairs.

Example
n=3
1+1+1=2+1=1+2=3=3

return 4
'''
class Solution:
    """
    @param n: An integer
    @return: An Integer
    """
    def climbStairs2(self, n):
        # write your code here
        if n == 1 or n == 0:
            return 1 
        if n == 2:
            return 2 
        grid = [0 for _ in range(n+1)]
        grid[0],grid[1],grid[2]=1,1,2
        for i in range(3,n+1):
            grid[i] = grid[i-1] + grid[i-2] + grid[i-3]
        return grid[-1]

'''
254. Drop Eggs
中文English
There is a building of n floors. If an egg drops from the k th floor or above, it will break. If it's dropped from any floor below, it will not break.

You're given two eggs, Find k while minimize the number of drops for the worst case. Return the number of drops in the worst case.

Example
Example 1:

Input: 100
Output: 14
Example 2:

Input: 10
Output: 4
Clarification
For n = 10, a naive way to find k is drop egg from 1st floor, 2nd floor ... kth floor. But in this worst case (k = 10), you have to drop 10 times.

Notice that you have two eggs, so you can drop at 4th, 7th & 9th floor, in the worst case (for example, k = 9) you have to drop 4 times.


'''
class Solution:
    """
    @param n: An integer
    @return: The sum of a and b
    """
    def dropEggs(self, n):
        # write your code here
        import math
        x = int(math.sqrt(n * 2))
        while x * (x + 1) / 2 < n:
            x += 1
        return x

'''
76. Longest Increasing Subsequence
中文English
Given a sequence of integers, find the longest increasing subsequence (LIS).

You code should return the length of the LIS.

Example
Example 1:
	Input:  [5,4,1,2,3]
	Output:  3
	
	Explanation:
	LIS is [1,2,3]


Example 2:
	Input: [4,2,4,5,3,7]
	Output:  4
	
	Explanation: 
	LIS is [2,4,5,7]
Challenge
Time complexity O(n^2) or O(nlogn)
'''
class Solution:
    """
    @param nums: An integer array
    @return: The length of LIS (longest increasing subsequence)
    """
    def longestIncreasingSubsequence(self, nums):
        # write your code here
        if nums is None or not nums:
            return 0
        dp = [1] * len(nums)
        for curr, val in enumerate(nums):
            for prev in range(curr):
                if nums[prev] < val:
                    dp[curr] = max(dp[curr], dp[prev] + 1)
        return max(dp)

'''
109. Triangle
中文English
Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.

Example
example 1
Given the following triangle:

[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).

example 2
Given the following triangle:

[
     [2],
    [3,2],
   [6,5,7],
  [4,4,8,1]
]
The minimum path sum from top to bottom is 12 (i.e., 2 + 2 + 7 + 1 = 12).

Notice
Bonus point if you are able to do this using only O(n) extra space, where n is the total number of rows in the triangle.

'''
class Solution:
    """
    @param triangle: a list of lists of integers
    @return: An integer, minimum path sum
    """
    def minimumTotal(self, triangle):
        # write your code here
        if not triangle or not triangle[0]:
            return 0
            
        n, m = len(triangle), len(triangle[-1])
        if n == 0 or m == 0:
            return 0

        path_sum = triangle[-1][::]
        
        for i in range(n - 2, -1, -1):
            for j in range(i + 1):
                path_sum[j] = min(path_sum[j], path_sum[j + 1]) + triangle[i][j]
        
        return path_sum[0]

'''
116. Jump Game
中文English
Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

Example
A = [2,3,1,1,4], return true.

A = [3,2,1,0,4], return false.

Notice
This problem have two method which is Greedy and Dynamic Programming.

The time complexity of Greedy method is O(n).

The time complexity of Dynamic Programming method is O(n^2).

We manually set the small data set to allow you pass the test in both ways. This is just to let you learn how to use this problem in dynamic programming ways. If you finish it in dynamic programming ways, you can try greedy method to make it accept again.


'''
class Solution:
    """
    @param A: A list of integers
    @return: A boolean
    """
    def canJump(self, A):
        # write your code here
        p = 0
        ans = 0
        for item in A[:-1]:
            ans = max(ans, p + item)
            #print(ans)
            if(ans <= p):
                return False
            p += 1
            #print(item,p)
        return True


'''
513. Perfect Squares
中文English
Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.

Example
Example 1:

Input: 12
Output: 3
Explanation: 4 + 4 + 4
Example 2:

Input: 13
Output: 2
Explanation: 4 + 9
'''
class Solution:
    # @param {int} n a positive integer
    # @return {int} an integer
    def numSquares(self, n):
        # Write your code here
        while n % 4 == 0:
            n /= 4
        if n % 8 == 7:
            return 4

        for i in xrange(n+1):
            temp = i * i
            if temp <= n:
                if int((n - temp)** 0.5 ) ** 2 + temp == n: 
                    return 1 + (0 if temp == 0 else 1)
            else:
                break
        return 3


'''
611. Knight Shortest Path
中文English
Given a knight in a chessboard (a binary matrix with 0 as empty and 1 as barrier) with a source position, find the shortest path to a destination position, return the length of the route.
Return -1 if destination cannot be reached.

Example
Example 1:

Input:
[[0,0,0],
 [0,0,0],
 [0,0,0]]
source = [2, 0] destination = [2, 2] 
Output: 2
Explanation:
[2,0]->[0,1]->[2,2]
Example 2:

Input:
[[0,1,0],
 [0,0,1],
 [0,0,0]]
source = [2, 0] destination = [2, 2] 
Output:-1
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
Notice
source and destination must be empty.
Knight can not enter the barrier.
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
603. Largest Divisible Subset
中文English
Given a set of distinct positive integers, find the largest subset such that every pair (Si, Sj) of elements in this subset satisfies: Si % Sj = 0 or Sj % Si = 0.

Example
Example 1:

Input: nums =  [1,2,3], 
Output: [1,2] or [1,3]
Example 2:

Input: nums = [1,2,4,8], 
Output: [1,2,4,8]
Notice
If there are multiple solutions, return any subset is fine.
'''
class Solution:
    # @param {int[]} nums a set of distinct positive integers
    # @return {int[]} the largest subset 
    def largestDivisibleSubset(self, nums):
        # Write your code here
        n = len(nums)
        dp = [1] * n
        father = [-1] * n

        nums.sort()
        m, index = 0, -1
        for i in xrange(n):
            for j in xrange(i):
                if nums[i] % nums[j] == 0:
                    if 1 + dp[j] > dp[i]:
                        dp[i] = dp[j] + 1
                        father[i] = j

            if dp[i] >= m:
                m = dp[i]
                index = i

        result = []
        for i in xrange(m):
            result.append(nums[index])
            index = father[index]

        return result