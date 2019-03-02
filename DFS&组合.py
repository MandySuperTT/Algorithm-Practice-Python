'''
Subsets
Given a set of distinct integers, return all possible subsets.
'''
class Solution:
    """
    @param nums: A set of numbers
    @return: A list of lists
    """
    def search(self, nums, S, index):
        # write your code here
        if index == len(nums):
            print(list(S))
            self.results.append(list(S))
            return
        
        S.append(nums[index])
        print(S,index)
        self.search(nums, S, index + 1)
        a=S.pop()
        print(a,index)
        self.search(nums, S, index + 1)
        
    def subsets(self, nums):
        self.results = []
        self.search(sorted(nums), [], 0)
        return self.results


    def helper2(self,subset,res,index,nums):
        res.append(subset[:])
        for i in range(index,len(nums)):
            subset.append(nums[i])
            self.helper2(subset,res,i+1,nums)
            subset.pop(-1)
        
    def helper3(self,res,nums):
        q = []
        q.append([])
        while q:
            subset = q.pop()[:]
            res.append(subset)
            for i in range(len(nums)):
                if not subset or subset[-1] < nums[i]:
                    newSubset = subset[:]
                    newSubset.append(nums[i])
                    q.append(newSubset)
        return res
        
    def helper4(self,res,nums):
        n = len(nums)
        for i in range(1<<n):
            subset = []
            for j in range(n):
                if i & 1 << j:
                    subset.append(nums[j])
            res.append(subset)
        return res


'''
Combination Sum
Given a set of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.

The same repeated number may be chosen from C unlimited number of times.
'''
class Solution:
    """
    @param candidates: A list of integers
    @param target: An integer
    @return: A list of lists of integers
    """
    def combinationSum(self, candidates, target):
        # write your code here
        candidates = sorted(list(set(candidates)))
        #print(list(set(candidates)).sort())
        self.result = []
        self.subset = []
        self.helper(candidates,0,target)
        return self.result
        
    def helper(self,candidates,startindex,target):
        if target == 0:
            self.result.append(self.subset[:])
        for i in range(startindex,len(candidates)):
            if target < candidates[i]:
                return
            self.subset.append(candidates[i])
            self.helper(candidates,i,target-candidates[i])
            self.subset.pop()


'''
Combination Sum II
Given a collection of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.

Each number in C may only be used once in the combination.
'''
class Solution:
    """
    @param num: Given the candidate numbers
    @param target: Given the target number
    @return: All the combinations that sum to target
    """
    def combinationSum2(self, num, target):
        # write your code here
        num.sort()
        self.result = []
        self.subset = []
        self.helper(num,target,0)
        return self.result
      
    def helper(self,num,target,startindex):
        if target == 0 and self.subset[:] not in self.result:
            self.result.append(self.subset[:])
            
        for i in range(startindex,len(num)):
            if target < num[i]: 
                return
            self.subset.append(num[i])
            print(self.subset)
            self.helper(num,target-num[i],i+1)
            self.subset.pop()



'''
k Sum II
Given n unique integers, number k (1<=k<=n) and target.

Find all possible k integers where their sum is target.
'''
class Solution:
    """
    @param: A: an integer array
    @param: k: a postive integer <= length(A)
    @param: targer: an integer
    @return: A list of lists of integer
    """
    def kSumII(self, A, k, target):
        # write your code here
        self.result = []
        self.subset = []
        self.helper(A,k,target,0)
        return self.result
        
    def helper(self,A,k,target,startindex):
        if target == 0 and k == 0:
            self.result.append(self.subset[:])
            
        for i in range(startindex,len(A)):
            if k == 0 or target < A[i]:
                return
            self.subset.append(A[i])
            self.helper(A,k-1,target-A[i],i+1)
            self.subset.pop()


'''
Palindrome Partitioning
Given a string s, partition s such that every substring of the partition is a palindrome.

Return all possible palindrome partitioning of s.
'''
class Solution:
    """
    @param: s: A string
    @return: A list of lists of string
    """
    def partition(self, s):
        results = []
        self.dfs(s, [], results)
        return results
    
    def dfs(self, s, stringlist, results):
        if len(s) == 0:
            results.append(stringlist)
            # results.append(list(stringlist))
            return
            
        for i in range(1, len(s) + 1):
            prefix = s[:i]
            print(prefix,i)
            if self.is_palindrome(prefix):
                self.dfs(s[i:], stringlist + [prefix], results)
                

    def is_palindrome(self, s):
        return s == s[::-1]

class Solution:
    """
    @param: s: A string
    @return: A list of lists of string
    """
    results = []
    isPalindrome = None 
    def partition(self, s):
        # write your code here
        # results = []
        if s is None or len(s) == 0:
            return self.results
            
        self.getIsPalindrome(s)
        self.helper(s, 0, [])
        return self.results
        
    def getIsPalindrome(self, s):
        size = len(s)
        self.isPalindrome = [[0]*size for i in range(size)]
        #print(self.isPalindrome)
        for i in range(size):
            self.isPalindrome[i][i] = True
            #print(self.isPalindrome)
        for i in range(size - 1):
            if s[i] == s[i+1]:
                self.isPalindrome[i][i + 1] = True
            else:
                self.isPalindrome[i][i + 1] = False 
            #print(self.isPalindrome)    
        for i in range(size-3, -1, -1):
            for j in range(i+2, size):
                self.isPalindrome[i][j] = self.isPalindrome[i + 1][j - 1] and s[i] == s[j]
            #print(self.isPalindrome)    
    def helper(self, s, startIndex, subset):
        if startIndex == len(s):
            self.addResult(s, subset)
            return 
        
        for i in range(startIndex, len(s)):
            if not self.isPalindrome[startIndex][i]:
                continue
            
            subset.append(i)
            print(subset)
            self.helper(s, i + 1, subset)
            subset.pop()
            
    def addResult(self, s, subset):
        result = []
        startIndex = 0
        for i in range(len(subset)):
            result.append(s[startIndex:subset[i]+1])
            startIndex = subset[i] + 1 
        
        self.results.append(result)
        print(self.results)



'''
Wildcard Matching
Implement wildcard pattern matching with support for '?' and '*'.

'?' Matches any single character.
'*' Matches any sequence of characters (including the empty sequence).
The matching should cover the entire input string (not partial).
'''
def isMatch(self, source, pattern):
        return self.is_match_helper(source, 0, pattern, 0, {})
        
        
    # source 从 i 开始的后缀能否匹配上 pattern 从 j 开始的后缀
    # 能 return True
    def is_match_helper(self, source, i, pattern, j, memo):
        if (i, j) in memo:
            return memo[(i, j)]
            
        # source is empty
        if len(source) == i:
            # every character should be "*"
            for index in range(j, len(pattern)):
                if pattern[index] != '*':
                    return False
            return True
            
        if len(pattern) == j:
            return False
            
        if pattern[j] != '*':
            matched = self.is_match_char(source[i], pattern[j]) and \
                self.is_match_helper(source, i + 1, pattern, j + 1, memo)
        else:                
            matched = self.is_match_helper(source, i + 1, pattern, j, memo) or \
                self.is_match_helper(source, i, pattern, j + 1, memo)
        
        memo[(i, j)] = matched
        return matched



 '''
 Regular Expression Matching
Implement regular expression matching with support for '.' and '*'.

'.' Matches any single character.
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).
'''
class Solution:
    """
    @param s: A string 
    @param p: A string includes "." and "*"
    @return: A boolean
    """
    def isMatch(self, s, p):
        # write your code here
        return self.helper(s,0,p,0,{})
        
    def helper(self,s,i,p,j,memo):
       
        if (i,j) in memo:
            return memo[(i,j)]
            
        if len(s) == i:
            if len(p[j:]) % 2 == 1:
                return False
        
            for i in range(len(p[j:]) // 2):
                if p[i * 2 + 1] != '*':
                    return False
            return True
            
        if len(p) == j:
            return False
        print(s[i],p[j],i,j)   
        if j < len(p) - 1 and p[j+1] == '*':
            match = self.match(s[i],p[j]) and self.helper(s,i+1,p,j,memo) or self.helper(s,i,p,j+2,memo)
            #print(False and False or True)
        else:
            match = self.match(s[i],p[j]) and self.helper(s,i+1,p,j+1,memo)
        memo[(i,j)] = match
        return match
        
    def match(self,a,b):
        return a==b or b == '.'
            

'''
Split String
Give a string, you can choose to split the string after one character or two adjacent characters, and make the string to be composed of only one character or two characters. Output all possible results.
'''
class Solution:
    """
    @param: : a string to be split
    @return: all possible split string array
    """
    
    def splitString(self, s):
        # write your code here
        self.result = []
        
        self.dfs(s,[])
        return self.result
        
    def dfs(self,s,subset):
        if len(s) == 0:
            self.result.append(subset)
            return
        if len(s) == 1:
            subset+=s[0]
            self.result.append(subset)
            return
        for i in range(1,3):
            com = s[:i]
            self.dfs(s[i:],subset+[com])
    '''       
    def splitString(self, s):
        if len(s) == 0:
            return [[]]
        if len(s) == 1:
            return [[s]]
        result1 = self.splitString(s[1:])
        #print('1',result1)
        result2 = self.splitString(s[2:])
       # print('2',result2)
        result = []
        for r1 in result1:
            #print(r1)
            result.append([s[0]] + r1)
            #print(result)
        for r2 in result2:
           # print(r2)
            result.append([s[:2]] + r2)
        return result
'''


'''
Word Break II
Given a string s and a dictionary of words dict, add spaces in s to construct a sentence where each word is a valid dictionary word.

Return all such possible sentences.
'''
class Solution:
    """
    @param: s: A string
    @param: wordDict: A set of words.
    @return: All possible sentences.
    """
    def wordBreak(self, s, wordDict):
        # write your code here
        return self.dfs(s, wordDict, {})
    
    # 找到 s 的所有切割方案并 return
    def dfs(self, s, wordDict, memo):
        if s in memo:
            return memo[s]
            
        if len(s) == 0:
            return []
            
        partitions = []
        
        for i in range(1, len(s)):
            prefix = s[:i]
            if prefix not in wordDict:
                continue
            
            sub_partitions = self.dfs(s[i:], wordDict, memo)
            for partition in sub_partitions:
                partitions.append(prefix + " " + partition)
                
        if s in wordDict:
            partitions.append(s)
            
        memo[s] = partitions
        return partitions

'''
Subsets II
Given a collection of integers that might contain duplicates, nums, return all possible subsets (the power set).
'''
class Solution:
    """
    @param nums: A set of numbers.
    @return: A list of lists. All valid subsets.
    """
    def subsetsWithDup(self, nums):
        # write your code here
        self.result = []
        self.subset = []
        self.helper(sorted(nums),0)
        return self.result
        
    def helper(self,nums,startindex):
        if self.subset in self.result:
            return
        if startindex == len(nums):
            self.result.append(self.subset[:])
            return
        self.subset.append(nums[startindex])
        self.helper(nums,startindex + 1)
        self.subset.pop()
        self.helper(nums,startindex + 1)


'''
Combinations
Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.
'''
class Solution:
    """
    @param n: Given the range of numbers
    @param k: Given the numbers of combinations
    @return: All the combinations of k numbers out of 1..n
    """
    def combine(self, n, k):
        # write your code here
        self.result = []
        self.subset = []
        self.helper(n,k,1)
        return self.result
        
    def helper(self,n,k,startindex):
       
        if len(self.subset) == k:
            self.result.append(self.subset[:])
            
        for i in range(startindex,n+1):
            self.subset.append(i)
            self.helper(n,k,i+1)
            self.subset.pop()



