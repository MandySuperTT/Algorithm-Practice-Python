'''
Median of two Sorted Arrays
There are two sorted arrays A and B of size m and n respectively. Find the median of the two sorted arrays.
'''
class Solution:
    """
    @param: A: An integer array
    @param: B: An integer array
    @return: a double whose format is *.5 or *.0
    """
    #find Kth
    def findMedianSortedArrays(self, A, B):
        n = len(A) + len(B)
        if n % 2 == 1:
            return self.findKth(A, B, n // 2 + 1)
        else:
            smaller = self.findKth(A, B, n // 2)
            bigger = self.findKth(A, B, n // 2 + 1)
            return (smaller + bigger) / 2.0

    def findKth(self, A, B, k):
        if len(A) == 0:
            return B[k - 1]
        if len(B) == 0:
            return A[k - 1]
        if k == 1:
            return min(A[0], B[0])
        
        a = A[k // 2 - 1] if len(A) >= k // 2 else None
        b = B[k // 2 - 1] if len(B) >= k // 2 else None
        
        if b is None or (a is not None and a < b):
            return self.findKth(A[k // 2:], B, k - k // 2)
        return self.findKth(A, B[k // 2:], k - k // 2)
            
    '''
    def findMedianSortedArrays(self, A, B):
        #binary search
        # write your code here
        c = len(A) + len(B) 
        k = c // 2
        if c % 2 == 1:
            return self.findkth(A,B,k + 1)
        else:
            small = self.findkth(A,B,k)
            big = self.findkth(A,B,k + 1) 
            return (small + big)/2.0 
            
    def findkth(self, A, B, k):
        
        if len(A) == 0:
            return B[k - 1] 
        if len(B) == 0:
            return A[k - 1]
        
        start = int (min (A[0], B[0]))
        end = max(A[-1], B[-1]) 
        
        while start + 1 < end:
            mid =  (start + end) // 2 
            k1 = self.countsmallerequ(A, mid) 
            k2 = self.countsmallerequ(  B, mid) 
            if k1 + k2 < k:
                start = mid                                                                      
            else:
                end = mid 
        if self.countsmallerequ(A,start) + self.countsmallerequ(B, start) >= k:
            return start 
        return end 
                
                
    def countsmallerequ(self, A, num):
        left, right = 0, len(A) - 1  
        
        while left + 1 < right:
            mid = int((right + left)/2)
            if A[mid] <= num:
                left = mid 
            else:
                right = mid
                
        if A[left] > num:
            return left 
        if A[right] > num:
            return right 
            
        return len(A)
        
         '''   
            

'''
Insert Interval
Given a non-overlapping interval list which is sorted by start point.

Insert a new interval into it, make sure the list is still in order and non-overlapping (merge intervals if necessary).
'''
"""
Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    """
    @param intervals: Sorted interval list.
    @param newInterval: new interval.
    @return: A new interval list.
    """
    def insert(self, intervals, newInterval):
        # write your code here
        answer = []
        insertindex = 0 
        for each in intervals:
            if each.end < newInterval.start:
                answer.append(each)
                insertindex += 1 
            elif each.start > newInterval.end:
                answer.append(each)
            else:
                newInterval.start = min(newInterval.start,each.start)
                newInterval.end = max(newInterval.end,each.end)
        answer.insert(insertindex,newInterval)
        return answer



'''
Intersection of Arrays
Give a number of arrays, find their intersection, and output their intersection size.
'''
class Solution:
    """
    @param arrs: the arrays
    @return: the number of the intersection of the arrays
    """
    def intersectionOfArrays(self, arrs):
        # write your code here
        answer = {}
        for each in arrs:
            for i in each:
                if i in answer:
                    answer[i] += 1 
                else:
                    answer[i] = 1 
        r = 0 
        for k,v in answer.items():
            if v == len(arrs):
                r += 1 
        return r 


'''
Merge K Sorted Arrays
Given k sorted integer arrays, merge them into one sorted array.
'''
import heapq
class Solution:
    """
    @param arrays: k sorted integer arrays
    @return: a sorted array
    """
    def mergekSortedArrays(self, arrays):
        # write your code here
        heap = []
        result = []
       # heap = []
        for index, array in enumerate(arrays):
            if len(array) == 0:
                continue
            heapq.heappush(heap, (array[0], index, 0))
            
        while len(heap):
            val, x, y = heap[0]
            heapq.heappop(heap)
            result.append(val)
            if y + 1 < len(arrays[x]):
                heapq.heappush(heap, (arrays[x][y + 1], x, y + 1))
        return result
                


'''
Count 1 in Binary
Count how many 1 in binary representation of a 32-bit integer.
'''
class Solution:
    """
    @param: num: An integer
    @return: An integer
    """
    def countOnes(self, num):
        # write your code here
        total = 0
        for i in range(32):
            total += num & 1
            num >>= 1
        return total




'''
Merge Sorted Array
Given two sorted integer arrays A and B, merge B into A as one sorted array.
'''
class Solution:
    """
    @param: A: sorted integer array A which has m elements, but size of A is m+n
    @param: m: An integer
    @param: B: sorted integer array B which has n elements
    @param: n: An integer
    @return: nothing
    """
    def mergeSortedArray(self, A, m, B, n):
        # write your code here
        i,j=m-1,n-1
        index = n+m-1
        while i >= 0 and j >= 0:
            if A[i] > B[j]:
                A[index] = A[i]
                index = index - 1
                i = i - 1
            else:
                A[index] = B[j]
                index = index - 1
                j = j - 1

        while j >= 0:
            A[index] = B[j]
            index = index - 1
            j = j - 1

        return A



'''
Merge Two Sorted Interval Lists
Merge two sorted (ascending) lists of interval and return it as a new sorted list. The new sorted list should be made by splicing together the intervals of the two lists and sorted in ascending order.
'''
"""
Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    """
    @param list1: one of the given list
    @param list2: another list
    @return: the new sorted list of interval
    """
    def mergeTwoInterval(self, list1, list2):
        # write your code here
        i, j = 0, 0
       
        intervals = []
        while i < len(list1) and j < len(list2):
            if list1[i].start < list2[j].start:
                self.push_back(intervals, list1[i])
                i += 1
            else:
                self.push_back(intervals, list2[j])
                j += 1
        while i < len(list1):
            self.push_back(intervals, list1[i])
            i += 1
        while j < len(list2):
            self.push_back(intervals, list2[j])
            j += 1
        
        return intervals
        
    def push_back(self, intervals, interval):
        if not intervals:
            intervals.append(interval)
            return
        
        last_interval = intervals[-1]
        if last_interval.end < interval.start:
            intervals.append(interval)
            return
        
        intervals[-1].end = max(intervals[-1].end, interval.end)


'''
Merge K Sorted Interval Lists
Merge K sorted interval lists into one sorted interval list. You need to merge overlapping intervals too.
'''
"""
Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""
import heapq
class Solution:
    """
    @param intervals: the given k sorted interval lists
    @return:  the new sorted interval list
    """
    def mergeKSortedIntervalLists(self, intervals):
        # write your code here
        data = []
        for i in intervals:
            data += i
        #print(data)
        data.sort(key= lambda t:t.start)
        res = [data[0]]
        for d in data:
            if res[-1].end < d.start:
                res += [d]
            else:
                res[-1].end = max(res[-1].end, d.end)
        return res


'''
 Intersection of Two Arrays
Given two arrays, write a function to compute their intersection.
'''
class Solution:
    """
    @param nums1: an integer array
    @param nums2: an integer array
    @return: an integer array
    """
    def intersection(self, nums1, nums2):
        # write your code here
        '''
        result = []
        num1 = set(nums1)
        num2 = set(nums2)
        if len(num1) > len(num2):
            for each in num2:
                if each in num1:
                    result.append(each)
        else:
            for each in num1:
                if each in num2:
                    result.append(each)
        return result
        '''
        return list(set(nums1) & set(nums2))
    
    '''
    hash set.
    python build in set operation.
    binary search + hash set.
    sort + two pointers
    def intersection_1(self, nums1, nums2):
        # idea: hash set
        unique = set(nums1)
        result = []
        for num in nums2:
            if num in unique:
                result.append(num)
                unique.discard(num)
        return result


    def intersection_2(self, nums1, nums2):
        return list(set(nums1) & set(nums2))


    def intersection_3(self, nums1, nums2):
        # idea: binary search + hash set
        result = set()
        if len(nums1) > len(nums2):
            s_nums = nums2
            b_nums = nums1
        else:
            s_nums = nums1
            b_nums = nums2

        s_nums.sort()
        for num in b_nums:
            if self.binary_seach(s_nums, num):
                result.add(num)
        return list(result)


    def binary_seach(self, nums, target):
        if not nums or len(nums) == 0:
            return False

        start, end = 0, len(nums) -1
        while start + 1 < end:
            middle = start + (end - start) // 2

            if nums[middle] <= target:
                start = middle
            elif nums[middle] > target:
                end = middle

        if nums[end] == target or nums[start] == target:
            return True
        return False


    def intersection_4(self, nums1, nums2):
        # idea: two pointers 
        nums1.sort()
        nums2.sort()
        result = []

        i, j = 0, 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] == nums2[j]:
                result.append(nums1[i])
                i += 1
                j += 1
                while i < len(nums1) and nums1[i] == nums1[i - 1]:
                    i += 1
                while j < len(nums2) and nums2[j] == nums2[j - 1]:
                    j += 1
                    
            elif nums1[i] > nums2[j]:
                j += 1
            else:
                i += 1

        return result
    '''

'''
Intersection of Two Arrays II
Given two arrays, write a function to compute their intersection.
'''
class Solution:
    """
    @param nums1: an integer array
    @param nums2: an integer array
    @return: an integer array
    """
    def intersection(self, nums1, nums2):
        # write your code here
        counts = collections.Counter(nums1)
        print(counts)
        result = []
        for each in nums2:
            if counts[each]>0:
                result.append(each)
                counts[each] -= 1 
                
        return result
        
'''
Sparse Matrix Multiplication
Given two Sparse Matrix A and B, return the result of AB.

You may assume that A's column number is equal to B's row number.
'''
class Solution:
    """
    @param A: a sparse matrix
    @param B: a sparse matrix
    @return: the result of A * B
    """
    def multiply(self, A, B):
        # write your code here
        
        AB = [[0 for _ in range(len(B[0]))] for i in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B)):
                if A[i][j] !=0:
                    for l in range(len(B[0])):
                        AB[i][l] += A[i][j] * B[j][l]
        return AB
                        
        
