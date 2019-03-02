‘’‘
Last Position of Target
Find the last position of a target number in a sorted array. Return -1 if target does not exist.
‘’‘
class Solution:
    """
    @param nums: An integer array sorted in ascending order
    @param target: An integer
    @return: An integer
    """
    '''
    def lastPosition(self, nums, target):
        # write your code here
        if len(nums) == 0:
            return -1
            
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] > target:
                end = mid
            else:
                start = mid
        if nums[end] == target:
            return end        
        if nums[start] == target:
            return start
        
        return -1
        '''
    def lastPosition(self, A, target):
        # Write your code here
        if len(A) == 0 or A == None:
            return -1
        
        start = 0
        end = len(A) - 1
        
        if target < A[start] or target > A[end]:
            return -1
        
        while start + 1 < end:
            mid = start + (end - start) / 2
            if A[mid] > target:
                end = mid
            else:
                start = mid
        
        if target == A[end]:
            return end
        elif target == A[start]:
            return start
        else:
            return -1

‘’‘
585. Maximum Number in Mountain Sequence
Given a mountain sequence of n integers which increase firstly and then decrease, find the mountain top.
‘’‘
class Solution:
    """
    @param nums: a mountain sequence which increase firstly and then decrease
    @return: then mountain top
    """
    def mountainSequence(self, nums):
        # write your code here
        start,end = 0,len(nums)-1
        while start + 1 < end:
            mid = (end+start) // 2
            if nums[start] < nums[mid]:
                if nums[mid-1] < nums[mid]:
                    start = mid
                else:
                    end = mid
            else:
                end = mid
        return max(nums[start],nums[end])
                

’‘’
460. Find K Closest Elements
Given a target number, a non-negative integer k and an integer array A sorted in ascending order, find the k closest numbers to target in A, sorted in ascending order by the difference between the number and target. Otherwise, sorted in ascending order by number if the difference is same.

‘’‘
class Solution:
    """
    @param A: an integer array
    @param target: An integer
    @param k: An integer
    @return: an integer array
    """
    
    def kClosestNumbers(self, A, target, k):
        # write your code here
        if len(A) == 0 or k > len(A):
            return -1
        start,end = 0, len(A)-1
        while start + 1 < end:
            mid = start + (end-start)//2
            if A[mid] < target:
                start = mid
            else:
                end = mid
        if A[end] < target:
            left= end
        if A[start] < target:
            left= start
        else:
            left = -1
        right = left + 1
        r=[]
        for i in range(k):
            if left< 0:
                #return A[:k-1]
                r.append(A[right])
                right += 1 
            elif right >= len(A):
                #return A[-k:-1]
                r.append(A[left])
                left -= 1
            elif A[right]-target < target-A[left]:
                r.append(A[right])
                right += 1 
            else:
                r.append(A[left])
                left -= 1
        return r
                
                
‘’‘
447. Search in a Big Sorted Array
Given a big sorted array with positive integers sorted by ascending order. The array is so big so that you can not get the length of the whole array directly, and you can only access the kth number by ArrayReader.get(k) (or ArrayReader->get(k) for C++). Find the first index of a target number. Your algorithm should be in O(log k), where k is the first index of the target number.

Return -1, if the number doesn't exist in the array.
‘’‘
class Solution:
    """
    @param: reader: An instance of ArrayReader.
    @param: target: An integer
    @return: An integer which is the first index of target.
    """
    def searchBigSortedArray(self, reader, target):
        # write your code here
        index = 0
        while reader.get(index) < target:
            index = index * 2 + 1 
        start, end = 0, index
        while start + 1 < end:
            mid = start + (end-start) // 2
            if reader.get(mid) < target:
                start = mid
            else:
                end = mid
        if reader.get(start) == target:
            return start
        elif reader.get(end) == target:
            return end
        else:
            return -1


‘’‘
428. Pow(x, n)
Implement pow(x, n).
‘’‘
class Solution:
    """
    @param x: the base number
    @param n: the power number
    @return: the result
    """
    def myPow(self, x, n):
        # write your code here
        if x == 0:
            return 0 
        
        if n == 0:
            return 1 
            
        if n < 0:
            x = 1 / x
            n = -n 
        
        if n == 1:
            return x
            
        if n % 2 == 0:
            temp = self.myPow(x, n // 2)
            return temp * temp
            
        else:
            temp = self.myPow(x, n // 2)
            return temp * temp * x

‘’‘
159. Find Minimum in Rotated Sorted Array
Suppose a sorted array is rotated at some pivot unknown to you beforehand.

(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

Find the minimum element.
’‘’
class Solution:
    """
    @param nums: a rotated sorted array
    @return: the minimum number in the array
    """
    def findMin(self, nums):
        # write your code here
        start,end = 0, len(nums)-1
        while start + 1 < end:
            mid = (end+start) // 2
            if nums[start] < nums[mid]:
                start = mid
            else:
                end = mid
        if (nums[0] < nums[start]) and (nums[0] < nums[end]):
            return nums[0]
        elif nums[start] < nums[end]:
            return nums[start]
        else:
            return nums[end]
                
‘’‘
140. Fast Power
Calculate the an % b where a, b and n are all 32bit positive integers.

’‘’
class Solution:
    """
    @param a: A 32bit integer
    @param b: A 32bit integer
    @param n: A 32bit integer
    @return: An integer
    """
    def fastPower(self, a, b, n):
        # write your code here
        '''
        ans=1 
       # a = a % b 
        while n > 0:
            if n % 2 == 1:
                ans = ans * a % b 
            a = a*a % b 
            n = n/2
        return ans % b     
        
        ans = 1
        while n > 0:
            if n % 2==1:
                ans = ans * a % b
            a = a * a % b
            n = n / 2
        return ans % b
        '''
        
        # write your code here
        if n == 0:
            return 1 % b
        if n % 2 == 0:
            tmp = self.fastPower(a, b, n / 2)
            return tmp * tmp % b
        else:
            tmp = self.fastPower(a, b, n / 2)
            return tmp * tmp * a % b

‘’‘
75. Find Peak Element
There is an integer array which has the following features:

The numbers in adjacent positions are different.
A[0] < A[1] && A[A.length - 2] > A[A.length - 1].
We define a position P is a peak if:

A[P] > A[P-1] && A[P] > A[P+1]
Find a peak element in this array. Return the index of the peak.
‘’‘
class Solution:
    """
    @param A: An integers array.
    @return: return any of peek positions.
    """
    def findPeak(self, A):
        # write your code here
        start,end = 0,len(A)-1
        while start + 1 < end:
            mid = (start+end)//2
            if A[mid+1] < A[mid]:
                end = mid
            else:
                start = mid
        if A[start] > A[end]:
            return start
        return end


‘’‘
74. First Bad Version
The code base version is an integer start from 1 to n. One day, someone committed a bad version in the code case, so it caused this version and the following versions are all failed in the unit tests. Find the first bad version.

You can call isBadVersion to help you determine which version is the first bad one. The details interface can be found in the code's annotation part.

‘’‘
#class SVNRepo:
#    @classmethod
#    def isBadVersion(cls, id)
#        # Run unit tests to check whether verison `id` is a bad version
#        # return true if unit tests passed else false.
# You can use .SVNRepo.isBadVersion(10) to check whether version 10 is a 
# bad version.
class Solution:
    """
    @param n: An integer
    @return: An integer which is the first bad version.
    """
    def findFirstBadVersion(self, n):
        # write your code here
        start,end = 1,n 
        while start + 1 < end:
            mid = start + (end-start)//2
            if SVNRepo.isBadVersion(mid):
                end = mid
            else:
                start = mid
        if SVNRepo.isBadVersion(start):
            return start
        return end

’‘’
62. Search in Rotated Sorted Array
Suppose a sorted array is rotated at some pivot unknown to you beforehand.

(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.
‘’‘
class Solution:
    """
    @param A: an integer rotated sorted array
    @param target: an integer to be searched
    @return: an integer
    """
    def search(self, A, target):
        # write your code here
        start,end = 0,len(A)-1
        if len(A) == 0:
            return -1
        while start+1<end:
            mid = (end+start)//2
            if A[start] < A[mid]:
                if A[mid] > target and A[start] <= target:
                    end = mid
                else:
                    start = mid
            else:
                if (A[mid] > target) or (A[start]<target):
                    end = mid
                else:
                    start = mid
        if A[start] == target:
            return start
        elif A[end] == target:
            return end
        else:
            return -1
            

