#贪心算法
#贪心算法没有普遍的模式和规律，比较复杂
#分发饼干 https://leetcode.cn/problems/assign-cookies/submissions/567034018/
class Solution(object):
    def findContentChildren(self, g, s):
        """
        :type g: List[int]
        :type s: List[int]
        :rtype: int
        """
        #或者 g.sort(reverse=True)
        g = sorted(g,reverse=True)
        s = sorted(s,reverse=True)
        gl = len(g)
        sl = len(s)
        i = 0
        j = 0
        r = 0
        while i < gl and j <sl:
            #注意这里是<= 刚好等于也算
            if g[i]<=s[j]:
                r+=1
                i+=1
                j+=1
            else:
                i+=1

        return r



#376. 摆动序列 https://leetcode.cn/problems/wiggle-subsequence/
#错误解：计算差序列计数摆动序列长度
#边界计算不明
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)<2:
            return len(nums)
        
        elif len(nums)==2 and nums[1]==nums[0]:
            return 1
        r = 0
        m = []
        for i in range(1,len(nums)):
            m.append(nums[i]-nums[i-1])
        
        c= 1
        lefti  = 0
        
        for i in range(1,len(m)):
            
            if m[i]*m[lefti]<0:
                
                lefti+=1
                c+=1
            else:
                continue  
        
        r = max(c,r)   
        r+=1
          
        return r
#正解
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
	#小于2直接返回长度
        if len(nums)<2:
            return len(nums)
	#prediff 记录之前差异
        prediff = 0
	#currdiff记录当前差异
        currdiff = 0
	#记录震荡，单独元素也算一个，因此先计1
        p  = 1
        for i in range(1,len(nums)):
		
            currdiff = nums[i]-nums[i-1]
            #用pre 处在0 的闭区间，巧妙的解决了初始的边界问题
	    #而当currdiff==0时，pre又是不会更新的，因此prediff==0只有初始状态一种情况
	    #	而差值变动的情况 因为仅要求计数，所以可以不考虑（相当于平移计算长度）
            if (prediff>=0 and currdiff<0) or (prediff<=0 and currdiff>0):
                p+=1
                prediff = currdiff
        return p
#修正解
#证明还是判断边界的问题，并非是不可以用差序列
class Solution(object):
    
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)<2:
            return len(nums)
        
        elif len(nums)==2 and nums[1]==nums[0]:
            return 1
      
        m = []
        for i in range(1,len(nums)):
            m.append(nums[i]-nums[i-1])
        
        c= 1
        lefti  = 0
        if m[0]!=0:
            c+=1
        for i in range(1,len(m)):
            prediff  =m[lefti]
            currdiff = m[i]
		#这里的判断逻辑整体上没问题，但是边界上不正确
		#边界上少算了初始为0 的情况
            #if m[i]*m[lefti]<0:
            if (prediff>=0 and currdiff<0) or (prediff<=0 and currdiff>0):

                
                lefti=i
                c+=1
            else:
                continue  
        
        
          
        return c


#53. 最大子序和 https://leetcode.cn/problems/maximum-subarray/
#贪心算法，贪心在于若当前和小于0，那么从下一个数重新计起
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
	#这里设为无穷小，方便边界设置，和值为负数的时候
        r = float('-inf')
        c = 0
     
            
        for i in range(len(nums)):
		#三步走，顺序很重要
		#先计和
            c += nums[i]
		#再对比赋值记录
            r = max(r,c)
		#若当前和小于0，重置和
            if c <0:
                
                c = 0
            
        return r
#暴力破解， 遇到长序列无法通过
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        r = float('-inf')
        c = 0
     
            
        for i in range(len(nums)):#i 为头目录
            c= 0
            for j in range(i,len(nums)):#j 末尾
                c+=nums[j]
                r = max(r,c)
        return r


#122.买卖股票的最佳时机 II https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/submissions/567933967/
#初始构想
#冗余较多，不过思路比较顺
class Solution(object):

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        r= 0
        c=0
        bd= 0
        for i in range(1,len(prices)):
            cc = prices[i]-prices[bd]
            if cc<c :
                
                bd = i 
                continue
            else:
                c = cc
                bd = i
                r = r+c
                c = 0
            
        return r
#简化逻辑
class Solution(object):

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        r= 0
      #只要把正收益全部收集即可
        for i in range(1,len(prices)):
            cc = prices[i]-prices[i-1]
            if cc>0 :
                r+=cc
            
        return r



#55. 跳跃游戏 https://leetcode.cn/problems/jump-game/submissions/568312978/
#这里需要注意的是元素为0 的情况， 当元素为0 且最远辐射小于等于为0的位置时，循环需要结束
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
	#这一步其实可以不写，写了可以加快一点效率
        if len(nums)<2:
            return True
	#初始化最远辐射距离
        m = 0
        #len-1，到了最后一位，不管是几都算true
        for i in range(len(nums)-1):
		#当元素为0且最远到这里时，遍历结束
            if nums[i]==0 and m<=i:
                break
		#否则位置+元素值为辐射距离
            n = nums[i]+i
		#维护最远辐射距离
            m = max(n,m)
	#这里要加1，因为i是从0计数的，对比的是数组长
        return m+1>=len(nums)
#优化解，效率更高
#在循环中返回True
#循环结束返回False
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        
        cover = 0
        if len(nums) == 1: return True
        i = 0
        # python不支持动态修改for循环中变量,使用while循环代替
	#当 遍历元素在 覆盖范围中时
        while i <= cover:
		#记录覆盖范围
            cover = max(i + nums[i], cover)
		#若范围到达末尾则可以
            if cover >= len(nums) - 1: return True
            i += 1
        return False


#45.跳跃游戏 II https://leetcode.cn/problems/jump-game-ii/submissions/568572763/
#这里求最小步数，很容易感觉需要递归
#其实不用，只需要计量这一步可到的范围和下一步可到的范围
#当下一步可到的范围覆盖了末端，就是最短的步数：要从覆盖范围出发，不管怎么跳，覆盖范围内一定是可以跳到的，以最小的步数增加覆盖范围，覆盖范围一旦覆盖了终点，得到的就是最少步数

class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        n  = len(nums)
	#小于2的数组有边界问题
        if n<2:
            return 0
	#步数记录
        r= 0
	#当前最大范围：默认从0位置开始
        curr = 0
	#下一步最大范围初始化为0
        nextr = 0
        for i in range(n):
		#计量下一步可达的最远范围
            nextr = max(i+nums[i],nextr)
		#如果达到了当前最远范围
            if i == curr:
		#步数+1
                r+=1
		#当前最远更新到下一步
                curr = nextr
		#若已经覆盖，则跳出循环
                if nextr >= n-1:
                    break
        return r



#1005.K次取反后最大化的数组和 https://leetcode.cn/problems/maximize-sum-of-array-after-k-negations/submissions/568793654/
#本题有两种情况：首先要把负的变成正的
#然后尽量变小的成负的
#思考后硬解，不够优雅
class Solution(object):
    def largestSumAfterKNegations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        nums.sort()
        r =0
	#for循环把负变正，然后根据剩余的变化数返回值
        for i in range(len(nums)):
            if k>0:
                if nums[i]<0:
                    nums[i] = -nums[i]
                    k-=1
                elif nums[i] == 0:
                    return sum(nums)
                elif nums[i]>0 and k%2==0:
                    return sum(nums)
                elif  nums[i]>0 and k%2!=0:
                    nums.sort()
                    nums[0]= -nums[0]
                    return sum(nums)
            else:
                break
        #注意，当全负数组长度小于k时需要进一步判断
        if k>0:
            if k%2==0:
                return sum(nums)
            else:
                nums.sort()
                nums[0]= -nums[0]
                return sum(nums)
        return sum(nums)
#正解，效率略低，但很优雅
class Solution(object):
    def largestSumAfterKNegations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        #排序表达式
	#nums.sort(ket=lambda x:abs(x), reverse=True)
	#传入 x 的绝对值 作为排序的依据 key, reverse = True 表示降序排列
        nums.sort(key=lambda x: abs(x), reverse=True)
        for i in range(len(nums)):
            if nums[i]<0 and k>0:
                nums[i] = -nums[i]
                k-=1
        if k%2!=0:
            nums[-1]= -nums[-1]
            
        return sum(nums)

            


