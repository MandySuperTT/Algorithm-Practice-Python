#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:05:42 2018

@author: xuexudong
"""
'''
一个合法的括号匹配序列有以下定义:
1、空串""是一个合法的括号匹配序列
2、如果"X"和"Y"都是合法的括号匹配序列,"XY"也是一个合法的括号匹配序列
3、如果"X"是一个合法的括号匹配序列,那么"(X)"也是一个合法的括号匹配序列
4、每个合法的括号序列都可以由以上规则生成。
例如: "","()","()()","((()))"都是合法的括号序列
对于一个合法的括号序列我们又有以下定义它的深度:
1、空串""的深度是0
2、如果字符串"X"的深度是x,字符串"Y"的深度是y,那么字符串"XY"的深度为max(x,y) 3、如果"X"的深度是x,那么字符串"(X)"的深度是x+1
例如: "()()()"的深度是1,"((()))"的深度是3。牛牛现在给你一个合法的括号序列,需要你计算出其深度。
'''
while 1:
    s = input()
    if s == '':
        print(0)
    m = 0
    n=0
    for each in s:
        if each == '(':
            n += 1
        else:
            n -= 1
        print(n,m)
        m = max(n,m)
    print(m)
        

'''
牛牛养了n只奶牛,牛牛想给每只奶牛编号,这样就可以轻而易举地分辨它们了。 每个奶牛对于数字都有自己的喜好,第i只奶牛想要一个1和x[i]之间的整数(其中包含1和x[i])。
牛牛需要满足所有奶牛的喜好,请帮助牛牛计算牛牛有多少种给奶牛编号的方法,输出符合要求的编号方法总数。 
输入描述:
输入包括两行,第一行一个整数n(1 ≤ n ≤ 50),表示奶牛的数量 第二行为n个整数x[i](1 ≤ x[i] ≤ 1000)


输出描述:
输出一个整数,表示牛牛在满足所有奶牛的喜好上编号的方法数。因为答案可能很大,输出方法数对1,000,000,007的模。

输入例子1:
4
4 4 4 4

输出例子1:
24
'''
l1= int(input())
l2 = input()
l = list(map(int,l2.split()))
l.sort()
m=1
for each in range(l1):
    m *= (l[each] - each) 
    
print(m%1000000007)
    
'''
如果一个字符串S是由两个字符串T连接而成,即S = T + T, 我们就称S叫做平方串,例如"","aabaab","xxxx"都是平方串.
牛牛现在有一个字符串s,请你帮助牛牛从s中移除尽量少的字符,让剩下的字符串是一个平方串。换句话说,就是找出s的最长子序列并且这个子序列构成一个平方串。 
输入描述:
输入一个字符串s,字符串长度length(1 ≤ length ≤ 50),字符串只包括小写字符。


输出描述:
输出一个正整数,即满足要求的平方串的长度。

输入例子1:
frankfurt

输出例子1:
4
'''
s = input()
#m = [each if each not in m else m.pop(each) for each in s]  
m=[]
for each in s:
    if each not in m:
        #m.append(each)
        #print(m)
        m += [each]
        print(m)
    else:
        m.remove(each)
print(len(s)-len(m))

def max1(s1,s2):
    lens=[[0]*(len(s2)+1) for _ in range(len(s1)+1)]
    #print(lens)
    for i in range(1,len(s1)+1):
        for j in range(1,len(s2)+1):
            if s1[i-1]==s2[j-1]:
                lens[i][j]=lens[i-1][j-1]+1
                #print(lens[i][j])
            else:
                lens[i][j]=max(lens[i-1][j],lens[i][j-1])
                #print('else:',lens[i][j])
    return lens[len(s1)][len(s2)]
s=input()
max2=0
for i in range(1,len(s)):
    max2=max(max1(s[:i],s[i:]),max2)
    #print(max2)
print(max2*2)
    
s=input()
new = []
for each in range(len(s)):
    if s[each] == '(':
        new += s[each]
        print(new)
    else:
        if each != 0 and len(new) != 0:
            if new[-1] == '(':
                new.pop(-1)
                print(new)
            else:
                new += s[each]
        else:
            new += s[each]
            print(new)
print(len(new))



def solve(eq,num,var='x'):
    #eq1 = eq.replace("=","-(")+")"
    #print(eq1)
    eq1 = eq + '-' + num
    print(eq1)
    c = eval(eq1,{var:1j})
    print(c)
    return -c.real/c.imag

a=int(input())
def xx(x):
    b = x
    while b//10 != 0:
        x += b//10
        b=b//10
    return x

low = 0
high = a
while low <=high:
    mid = (low + high)/2
    if xx(mid) < a:
        low = mid
    elif xx(mid) > a:
        high = mid
    else:
        print(mid)
        break
if low > high:
    print(-1)

a=input()
a=list(map(int,a.split()))
b=input()
b=list(map(int,b.split()))
b.sort()
def long(x):
    long = 0
    for each in range(a[0]):
        if b[each] > x:
            long += b[each]-x
    
    return long
low = b[0]
high = b[-1]
m=[]
while (high-low) > 1:
    mid = (low+high)//2
    if long(mid) < a[1]:
        high = mid       
    elif long(mid) > a[1]:
        low = mid        
    else:
        m.append(mid)
        low = mid
        print(mid)
        
    

s=int(input())
add1 = s
add2 = 1
l=0
ll=0
for each in range(s-1):
    #print(s,each,l)
    l = s-(each+2)
    print(l)
    ll += l
    #print(l)
add3 = 1
print(add1+add2+ll+add3)   



'''
Longest Palindromic Substring
Manacher's
'''
s = 'abcdzdcab'
s = '#'+'#'.join(s)+'#'
pos = 0
i = 0 
MaxR = 0
p = [0]*len(s)
L = 0
d1={}
for i in range(len(s)):
    i_mirror = pos - (i - pos)
    if MaxR > i:
        p[i] = min(MaxR - i, p[i_mirror])
    else:
        p[i] = 0
    while i- p[i] -1>= 0 and i + p[i] +1< len(s) and s[i+p[i]+1] == s[i-p[i]-1]:
        p[i] += 1
    if i + p[i] >= MaxR:
        MaxR = i + p[i]
        pos = i
    
    ss = s[i-p[i]:i+p[i]+1]
    L = max(L,len(ss))
    if L not in d1:
        d1[L] = ss
d = ''.join(d1[L].split('#'))
                         
        
s='#'+'#'.join(s)+'#'

RL=[0]*len(s)
MaxRight=0
pos=0
MaxLen=0
for i in range(len(s)):
    if i<MaxRight:
        RL[i]=min(RL[2*pos-i], MaxRight-i)
    else:
        RL[i]=1
    #尝试扩展，注意处理边界
    while i-RL[i]>=0 and i+RL[i]<len(s) and s[i-RL[i]]==s[i+RL[i]]:
        RL[i]+=1
    #更新MaxRight,pos
    if RL[i]+i-1>MaxRight:
        MaxRight=RL[i]+i-1
        pos=i
    #更新最长回文串的长度
    MaxLen=max(MaxLen, RL[i])


'''
分解质因数
‘’‘
'''
    
import math
n=10
up = int(math.sqrt(n))
k=2
r=[]
while k<=up and n>1:
    while(n % k == 0):
        n/=k
        print(n)
        r.append(k)
    k += 1
if n >1:
    r.append(k)

'''
迭代二分法
'''
nums= [1,2,2,4,5,5]
target = 2
def Binary(nums,start,end,target):
    if start > end:
        return -1
    mid = int((start + end)/2)           
    if nums[mid] < target:
        return Binary(nums,mid+1,end,target)
    if nums[mid] > target:
        return Binary(nums,start,mid-1,target)
    return mid
a=Binary(nums,0,len(nums)-1,target)


class mine:
    def recur(self, num):
        print(num, end="")
        if num > 1:
            print(" * ",end="")
            return num * self.recur(self, num-1)
        print(" =")
        return 1
a = mine()
print(mine.recur(mine,10))


# 本参考程序来自九章算法，由 @九章算法 提供。版权所有，转发请注明出处。
# - 九章算法致力于帮助更多中国人找到好的工作，教师团队均来自硅谷和国内的一线大公司在职工程师。
# - 现有的面试培训课程包括：九章算法班，系统设计班，算法强化班，Java入门与基础算法班，Android 项目实战班，
# - Big Data 项目实战班，算法面试高频题班, 动态规划专题班
# - 更多详情请见官方网站：http://www.jiuzhang.com/?source=code

'''
二分法模版
'''
class Solution:
    # @param nums: The integer array
    # @param target: Target number to find
    # @return the first position of target in nums, position start from 0 
    #找到第一个index
    def binarySearch(self, nums, target):
        if len(nums) == 0:
            return -1
            
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            mid = start + (end - start) / 2
            if nums[mid] < target:
                start = mid
            else:
                end = mid
                
        if nums[start] == target:
            return start
        if nums[end] == target:
            return end
        return -1

#找到index
def findPosition(self, nums, target):
        start = 0
        end = len(nums)-1
        if len(nums) == 0 or nums == None:
            return -1
        if target < nums[start] or target > nums[end]:
            return -1
        while start + 1 < end:
            mid = int((start + end)/2)  
            if nums[mid] == target:
                return mid
            if nums[mid] > target:
                end = mid
            start = mid
        if nums[start] == target:
            return start
        if nums[end] == target:
            return end
        return -1

'''
k nn
'''
def kClosestNumbers(A, target, k):
    # write your code here
    if len(A) == 0 or k > len(A):
        return -1
    start,end = 0, len(A)-1
    while start + 1 < end:
        mid = start + (end-start)//2
        if A[mid] < target:
            start = mid
            print('start:',start)
        else:
            end = mid
            print('end:',end)
    if A[end] < target:
        left= end
    elif A[start] < target:
        left= start
    else:
        left = -1
    print(left)
    right = left + 1
    print(right)
    r=[]
    for i in range(k):
        if left< 0:
            #return A[:k-1]
            r.append(A[right])
            right += 1 
            print('left<0',r)
        elif right >= len(A):
            #return A[-k:-1]
            r.append(A[left])
            left -= 1
            print('right >',r)
        elif A[right]-target < target-A[left]:
            r.append(A[right])
            right += 1 
            print('compare:',r)
        else:
            r.append(A[left])
            left -= 1
            print('else',r)
    return r

'''
找revered sort里最小值
例子:[1,1,1,1,1....,1] 里藏着一个0
最坏情况下需要把每个位置上的1都看一遍，才能找到最后一个有0 的位置
考点:能否想到这个最坏情况的例子，而不是写代码!
'''
nums = [4,5,6,7,0,1,2,3]
start,end = 0, len(nums)-1
while start + 1 < end:
    mid = (end+start) // 2
    if nums[start] < nums[mid]:
        start = mid
        print(start,mid)
    else:
        end = mid
        print(end)
if (nums[0] < nums[start]) and (nums[0] < nums[end]):
    print(nums[0],nums[start],nums[end])
elif nums[start] < nums[end]:
    print(nums[start])
else:
    print(nums[end])

def findMin(self, nums):
    if len(nums) == 0:
        return 0
        
    start, end = 0, len(nums) - 1
    target = nums[-1]
    while start + 1 < end:
        mid = (start + end) / 2
        if nums[mid] <= target:
            end = mid
        else:
            start = mid
    return min(nums[start], nums[end])


'''
先增后减
'''
 def mountainSequence(self, nums):
    # write your code here
    left, right = 0, len(nums)-1
    while left + 1 < right:
        mid = (left + right) // 2
        if nums[mid] < nums[mid+1]:
            left = mid
        else:
            right = mid
    if nums[right] > nums[left]:
        return nums[right]
    else:
        return nums[left]
  
def power(x,n):
    ans = 1
    base = x
    while (n != 0):
        if (n % 2 == 1):
            ans *= base      
        base *= base
        n = n // 2
        print(n,base,ans)   
   print(ans)
   
    
start,end = 0,len(A)-1
if len(A) == 0:
    print(-1)
while start+1<end:
    mid = (end+start)//2
    if A[start] < A[mid]:
        if A[mid] > target and A[start] > target:
            start = mid
        else:
            end = mid
        print(A[mid],start,end)
    else:
        if A[mid] > target:
            end = mid
        else:
            start = mid
        print(A[mid],start,end)
if A[start] == target:
     print(start)
elif A[end] == target:
    print(end)
else:
    print(-1)  
    
    
    

def recoverRotatedSortedArray(nums):
        # write your code here
    for i in range(len(nums)):
        if nums[i]>nums[i+1]:
            a=nums[:i+1]
            a.reverse()
            b=nums[i+1:]
            b.reverse()
            nums = a+b
            nums = nums[::-1]
            return nums
def rotateString(str1, offset):
        # write your code here
    offset = -offset
    str1 = str1[offset:] + str1[:offset]
    return str1


s=[4,5,1,2,3]
left,right = 0,len(s)-1
while(left<right):
    temp = s[left]
    s[left] = s[right]
    s[right] = temp
    left += 1
    right -= 1
print(s)

def deduplication(nums):
    '''
    # write your code here
    hash = {}
    #start,end=0,len(nums)-1
    i=0
    count = 0
    while i < len(nums)-1 and count < len(nums):
        if nums[i] not in hash:
            hash[nums[i]] = i
            i += 1
            count += 1
            print(i,hash)
        else:
            nums.append(nums.pop(i))
            count += 1
            print(i,nums)
            '''
    point1,point2 = 0,0
    nums.sort()
    for i in range(len(nums)-1):
        if nums[i] != nums[i+1]:
            nums[point1] = nums[i]
            point1 += 1
            point2 = point1 
            print(i,point1,nums)
            
            
        
def twoSum7(nums, target):
    # write your code here
    start=0 
    while start < len(nums)-1:
        end=start+1
        while end < len(nums):
            if abs(nums[end]-nums[start]) != abs(target):
                print(end,nums[start],nums[end],abs(nums[end]-nums[start]),abs(target))
                end += 1 
                
            else:
                #abs(nums[end]-nums[start]) == abs(target):
                print(nums[start],nums[end],abs(nums[end]-nums[start]),abs(target))
                print([start+1,end+1])
        start+=1   
            