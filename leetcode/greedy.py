#贪心算法
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

