'''
228. Middle of Linked List
Find the middle node of a linked list.

Example
Given 1->2->3, return the node with value 2.

Given 1->2, return the node with value 1.
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
    @param head: the head of linked list.
    @return: a middle node of the linked list
    """
    def middleNode(self, head):
        # write your code here
        if head == None:
            return None
        
        a=head
        b=head.next
        
        while b != None and b.next != None:
            b=b.next.next
            a=a.next
          
        return a 

'''
607. Two Sum III - Data structure design
Design and implement a TwoSum class. It should support the following operations: add and find.

add - Add the number to an internal data structure.
find - Find if there exists any pair of numbers which sum is equal to the value.

Example
add(1); add(3); add(5);
find(4) // return true
find(7) // return false
'''
class TwoSum:
    """
    @param number: An integer
    @return: nothing
    """
    num = []
    def add(self, number):
        # write your code here
       
        self.num.append(number)
        #print(self.num)

    """
    @param value: An integer
    @return: Find if there exists any pair of numbers which sum is equal to the value.
    """
    
    def find(self, value):
        # write your code here
        hashh={}
        for key in self.num:
            #print(value - key,self.num[key],hashh)
            if (value -key) in hashh:
                return True
            hashh[key]=key
        return False


'''
539. Move Zeroes
Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.

Example
Given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].
'''
class Solution:
    """
    @param nums: an integer array
    @return: nothing
    """
    def moveZeroes(self, nums):
        # write your code here
        '''
        start = 0
        end = len(nums) - 1
        
        while start <= end:
            while nums[end] == 0:
                end -= 1
            if nums[start] == 0:
                temp = nums[start]
                nums[start] = nums[end]
                nums[end] = temp
                start += 1 
                end -= 1 
            
        return nums
        '''
        
        if 0 not in nums:
            return nums
        start,end = 0,1 
        n = len(nums)
        while end < n:
            while nums[end] == 0:
                end += 1 
                if (end == n-1) and (nums[end] == 0):
                    return nums
            if nums[start] == 0:
                nums[start],nums[end] = nums[end],nums[start]
                start += 1 
                end += 1 
            else:
                start += 1
                if start == end:
                    end += 1 
        return nums
        

'''
521. Remove Duplicate Numbers in Array
Given an array of integers, remove the duplicate numbers in it.

You should:

Do it in place in the array.
Move the unique numbers to the front of the array.
Return the total number of the unique numbers.
Example
Given nums = [1,3,1,4,4,2], you should:

Move duplicate integers to the tail of nums => nums = [1,3,4,2,?,?].
Return the number of unique integers in nums => 4.
Actually we don't care about what you place in ?, we only care about the part which has no duplicate integers.
'''
class Solution:
    """
    @param nums: an array of integers
    @return: the number of unique integers
    """
    def deduplication(self, nums):
        # write your code here
        '''
        hash = {}
        i=0
        count = 0
        while i < len(nums):
            if nums[i] not in hash:
                hash[nums[i]] = i
                nums[count] = nums[i]
                count += 1 
            i += 1
                
        return len(hash)
        '''
        if len(nums) == 0:
            return 0
        point1 = 1 
        nums.sort()
        for i in range(1,len(nums)):
            if nums[i-1] != nums[i]:
                nums[point1] = nums[i]
                point1 += 1
        return point1
            
                
'''
464. Sort Integers II
Given an integer array, sort it in ascending order. Use quick sort, merge sort, heap sort or any O(nlogn) algorithm.

Example
Given [3, 2, 1, 4, 5], return [1, 2, 3, 4, 5].
'''
class mergeSort(object):
    def __init__(self,nums):
        self.A = nums
        size = len(self.A)
        self.temp = [0 for _ in range(size)]
        self.sort(0,size-1)

        
    def sort(self,start,end):
        if start >= end:
            return 
        mid = start + (end-start)/2
        self.sort(start,mid)
        self.sort(mid+1,end)
        self.merge(start,end)
        
    def merge(self,start,end):
        mid = start + (end-start)/2
        l,r = start,mid+1
        index = start
        while l<=mid and r<=end:
            if self.A[l] < self.A[r]:
                self.temp[index] = self.A[l]
                index += 1
                l += 1
            else:
                self.temp[index] = self.A[r]
                index += 1
                r += 1
        while l<=mid:
            self.temp[index] = self.A[l]
            index += 1
            l += 1
        while r<=end:
            self.temp[index] = self.A[r]
            index += 1
            r += 1
        for i in range(start,end+1):
            self.A[i] = self.temp[i]


class Solution:
    """
    @param A: an integer array
    @return: nothing
    """
    def sortIntegers2(self, A):
        # write your code here
        self.sort(A,0,len(A)-1)
    def sort(self,A,start,end):
        if start >= end:
            return
        #start,end = 0,len(A)-1
        l,r = start,end
        mid = A[(start+end)//2]
        while l<=r:
            while l<=r and A[l] < mid:
                l+=1 
            while l<=r and A[r] > mid:
                r-=1 
            if l<=r:
                A[l],A[r] = A[r],A[l]
                l+=1 
                r-=1 
        self.sort(A,start,r)
        self.sort(A,l,end)


'''
608. Two Sum II - Input array is sorted
Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.

The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.

Example
Given nums = [2, 7, 11, 15], target = 9
return [1, 2]
'''
class Solution:
    """
    @param nums: an array of Integer
    @param target: target = nums[index1] + nums[index2]
    @return: [index1 + 1, index2 + 1] (index1 < index2)
    """
    def twoSum(self, nums, target):
        # write your code here
        start,end = 0,len(nums)-1
        while start < end:
            if nums[start] + nums[end] == target:
                return [start+1,end+1]
            elif nums[start] + nums[end] > target:
                end -= 1 
            else:
                start += 1 

'''
143. Sort Colors II
Given an array of n objects with k different colors (numbered from 1 to k), sort them so that objects of the same color are adjacent, with the colors in the order 1, 2, ... k.

Example
Given colors=[3, 2, 2, 1, 4], k=4, your code should sort colors in-place to [1, 2, 2, 3, 4].
'''
class Solution:
    """
    @param colors: A list of integer
    @param k: An integer
    @return: nothing
    """
    '''
    def sortColors2(self, colors, k):
        # write your code here
        self.partition(colors,1,k,0,len(colors)-1)
        
    def partition(self,colors,cf,ct,inf,into):
        if cf == ct or inf == into:
            return
        l,r = inf,into
        mid = (cf+ct)//2
        while l <= r:
            while l<=r and colors[l] < mid:
                l += 1 
            while l <= r and colors[r] > mid:
                r -= 1 
            if l <= r:
                colors[l],colors[r] = colors[r],colors[l]
                l+=1 
                r-=1 
        
        self.partiton(colors,cf,mid,inf,r)
        self.partiton(colors,mid+1,ct,l,into)
    '''
        
    def sortColors2(self, colors, k):
        self.sort(colors, 1, k, 0, len(colors) - 1)
        
    def sort(self, colors, color_from, color_to, index_from, index_to):
        if color_from == color_to or index_from == index_to:
            return
            
        color = (color_from + color_to) // 2
        
        left, right = index_from, index_to
        while left <= right:
            while left <= right and colors[left] <= color:
                left += 1
            while left <= right and colors[right] > color:
                right -= 1
            if left <= right:
                colors[left], colors[right] = colors[right], colors[left]
                left += 1
                right -= 1
        
        self.sort(colors, color_from, color, index_from, right)
        self.sort(colors, color + 1, color_to, left, index_to)
    

'''
57. 3Sum
Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Example
For example, given array S = {-1 0 1 2 -1 -4}, A solution set is:

(-1, 0, 1)
(-1, -1, 2)
Notice
Elements in a triplet (a,b,c) must be in non-descending order. (ie, a ≤ b ≤ c)
'''
class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @return: Find all unique triplets in the array which gives the sum of zero.
    """
    def threeSum(self, numbers):
        # write your code here
        '''
        numbers.sort()
         
        r = []
        for a in range(len(numbers)):
            
            b,c = a+1, len(numbers)-1
            while b < c:
                if numbers[a] + numbers[b] + numbers[c] == 0:
                    r.append((numbers[a],numbers[b],numbers[c]))
                    b += 1 
                    c-= 1 
                elif numbers[a] + numbers[b] + numbers[c] > 0:
                    c -= 1 
                else:
                    #a += 1 
                    b += 1 
        return list(set(r)) 
        '''
        self.results = []
        if not numbers or len(numbers) < 3:
            return self.results
        
        numbers.sort()
        for i in range(len(numbers)-1, 1, -1):
            if i <= len(numbers) - 2 and numbers[i] == numbers[i+1]:
                continue
            
            left, right = 0, i - 1
            self.two_sum(numbers[left:right+1], left, right, -numbers[i])
        
        return self.results
    
    def two_sum(self, nums, left, right, target):
        while left < right:
            if left >= 1 and nums[left] == nums[left-1]:
                left += 1
                continue
            if right <= len(nums) - 2 and nums[right] == nums[right+1]:
                right -= 1
                continue
            
            if nums[left] + nums[right] == target:
                self.results.append([nums[left], nums[right], -target])
                left += 1
                right -= 1
            elif nums[left] + nums[right] > target:
                right -= 1
            else:
                left += 1

'''
31. Partition Array
Given an array nums of integers and an int k, partition the array (i.e move the elements in "nums") such that:

All elements < k are moved to the left
All elements >= k are moved to the right
Return the partitioning index, i.e the first index i nums[i] >= k.

Example
If nums = [3,2,2,1] and k=2, a valid answer is 1.
'''
class Solution:
    """
    @param nums: The integer array you should partition
    @param k: An integer
    @return: The index after partition
    """
    def partitionArray(self, nums, k):
        # write your code here
        start, end = 0, len(nums) - 1
        while start <= end:
            while start <= end and nums[start] < k:
                start += 1
            while start <= end and nums[end] >= k:
                end -= 1
            if start <= end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1
        return start


'''
5. Kth Largest Element
Find K-th largest element in an array.

Example
In array [9,3,2,4,8], the 3rd largest element is 4.

In array [1,2,3,4,5], the 1st largest element is 5, 2nd largest element is 4, 3rd largest element is 3 and etc.
'''
class Solution:
    """
    @param n: An integer
    @param nums: An array
    @return: the Kth largest element
    """
    
    def kthLargestElement(self, n, nums):
        # write your code here
        if not nums or n < 1 or n > len(nums):
            return None
        return self.partition(nums,0,len(nums)-1,len(nums)-n)
    
    def partition(self,nums,l,r,k):
        if l == r:
            return nums[k]
        start,end = l,r
        mid = nums[(l+r)//2]
        while start <= end :
            while start<=end and nums[start] < mid:
                start += 1
            while start<=end and nums[end] > mid:
                end -= 1 
            if  start <=end:
                nums[start],nums[end]=nums[end],nums[start]
                start += 1 
                end -= 1 
        
        if k <= end:
            return self.partition(nums,l,end,k)
        if k >= start:
            return self.partition(nums,start,r,k)
        
        return nums[k]
        '''
    def kthLargestElement(self, k, A):
        A.sort()
        return A[-k]
        '''
        
        
        


