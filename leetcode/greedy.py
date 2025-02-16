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


#134. 加油站  https://leetcode.com/problems/gas-station/submissions/1405920057/
#贪心算法考的都是思路
#本题有两个点：首先是总消耗大于总储备，则无法抵达
#第二个是当前累计的留存量小于0 时，那么起始点应设在后一个点
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        #起始点
        start = 0
	#总留存量
        ts = 0
	#当前起始点累计留存量
        cur =0
        for i in range(len(gas)):
            cur+= gas[i]-cost[i]
            ts += gas[i]-cost[i]
		#若累计留存量小于0，则重置起始点为下一站
		#这里假如起始点在第一个，即循环开始，则会在i=0时通过判定
		#假如最后也没找到起始点，start也不会越界，因为后边有判定总留存量
            if cur < 0:
                cur = 0
                start = i+1
	#判定总留存量
        if ts <0: return -1
        return start


#135. 分发糖果 https://leetcode.cn/problems/candy/submissions/569294786/
#本题思路在于相邻有左右邻居
#因此要设一个分发糖果的数组，从左往右看一遍找到满足的情况，再从右往左看一遍再找到满足的情况，取二者中大的那个即可同时满足左右
class Solution(object):
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
	#分发数组初始化为1
        candyvec = [1] * len(ratings)
        #从左往右看，满足右大于左的 +1
	for i in range(1,len(ratings)):
            if ratings[i]>ratings[i-1]:
                candyvec[i] = candyvec[i-1]+1
	#从右往左看， 因为左边的元素比较和右边的有关，因此要逆序遍历，左大于右时的情况
        #这里是len -2 到 -1，即倒数第二个元素 到 0， -1 表示逆序 
        for i in range(len(ratings)-2,-1,-1):
	#这里左大于右时，取左右逻辑的最大值 max(candyvec[i],candyvec[i+1]+1)
            if ratings[i]>ratings[i+1]:
                candyvec[i] = max(candyvec[i],candyvec[i+1]+1)
        return sum(candyvec)


#860.柠檬水找零 https://leetcode.cn/problems/lemonade-change/submissions/569426432/
#本题在于要建立一个基于面值的计数
#然后分情况找零
#逻辑很简单，只不过情况优点多
class Solution(object):
    def lemonadeChange(self, bills):
        """
        :type bills: List[int]
        :rtype: bool
        """
	# 5 10 20 计数
        rest = [0,0,0]
        
        for i in range(len(bills)):
		#收到5刀
            if bills[i] == 5:
                rest[0] +=1
		#收到10刀找5刀不够了就找不开
            if bills[i] == 10:
                if rest[0]>0:
                    rest[0] -= 1
                    rest[1] +=1
                else:
                    return False
		#收到20 先找10块，没有10块找三个5
            if bills[i] == 20:
                if rest[0]>0 and rest[1]>0:
                    rest[0]-=1
                    rest[1]-=1
                    rest[2]+=1
                elif rest[0]>=3:
                    rest[0] -=3
                    rest[2]+=1
                else:
                    return False
        return True       


#406.根据身高重建队列 https://leetcode.cn/problems/queue-reconstruction-by-height/submissions/569610215/
#本题比较重要：
#首先思路上，是从大到小排人，从小到大排阶层
#然后按阶层插入数组对应位置即可，因为后插入的都比先插入的小，所以都是合法的
#其次是代码细节上
class Solution(object):
    
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
	#排序 数组.sort(key  = lambda x: (-x[0],x[1]))
	#当数组要自定义简单排序逻辑，且有两重排序规则时，括号里按顺序编写规则
        people.sort(key  = lambda x: (-x[0],x[1]))
        r = []
        for i in range(len(people)):
		# insert 函数再次出现，表示插入到指定位置
            r.insert(people[i][1],people[i])
        return r


#452. 用最少数量的箭引爆气球 https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/submissions/570011194/
#本题在于想清楚左右界
#然后就往前推，假如不排序，肯定是乱计的，因此要先排序
#排序以后维持交集，新点出了交集，那么count+1
#for循环和赋值有些冗余，可以精炼，不过性能没有提升
class Solution(object):
    def findMinArrowShots(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        if len(points)<2:
            return len(points)
	#排序，这个地方也可以不用双重排序，不影响逻辑，但是思路就是感觉这样更可能
        points.sort(key = lambda x: (x[0],-x[1]))
        count = 1
	#初始化交集
        lm=points[0][0]
        rm = points[0][1]
	

        for i in points:
            l = i[0]
            r = i[1]
		#遍历的逻辑就是左界超出了交集右界，就是说无交集了，则需要新箭，而因为是排序过的，所以以后的左界越来越右
            if l > rm:
                count +=1
                lm = i[0]
                rm = i[1]
		#否则有交集，更新交集上下界
            else:
                rm = min(r,rm)
                l = max(l,lm)
        return count

#435. 无重叠区间 https://leetcode.cn/problems/non-overlapping-intervals/submissions/570212976/
#原始解 有错误且效率低
class Solution(object):
    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        if len(intervals)<2:
            return 0
        c = 0
	#这里排序可以单层排序，因为双重排序也弥补不了逻辑上的失误，且效率很低
        intervals.sort(key = lambda x:(x[0],x[1]-x[0]))
	#这里cl其实是不必要的，因为cl已经排序过了，而且判断只需要cr
        cl,cr  = intervals[0][0],intervals[0][1]
        for i in range(1,len(intervals)):
		#这个地方有个逻辑错误，就是当有覆盖是不仅要计数加1，还需要判断右界是否变更，
		#因为靠右（下界更大的）可能会有更短的（但是上界更小）区间出现，因此就可以换右界更小的区间，找到计数
            if intervals[i][0]<cr:
                c+=1
            else:
                cl,cr  = intervals[i][0],intervals[i][1]
        return c
#修正解
class Solution(object):
    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        if len(intervals)<2:
            return 0
        c = 0
	#单排，快了很多
        intervals.sort(key = lambda x: x[0])
	#单变量计上界
        cr  = intervals[0][1]
        for i in range(1,len(intervals)):
            if intervals[i][0]<cr:
		#取靠左的（小的）上界
                cr  = min(cr,intervals[i][1])
                c+=1
            else:
                cr  =intervals[i][1]
        return c


#763.划分字母区间 https://leetcode.cn/problems/partition-labels/submissions/570461983/
#本题有些复杂，因为首先要找到字母的最大坐标
#然后要理清返回结果的逻辑
#细节很多
class Solution(object):
    
    def partitionLabels(self, s):
        """
        :type s: str
        :rtype: List[int]
        """
        r= []
        d = defaultdict()
        for i in range(len(s)):
		#这里用了字典，有个方法叫setdefault,可以初始化key 对应的value值和数据类型
		#和之前的d[i] = d.get(i, 0) +1 和这个get 设置初始值的 操作类似
            d.setdefault(s[i],[]).append(i)
	#记录当前字段最大坐标，初始化为-1
        m = -1
        for i in range(len(s)):
            #获得当前字母最大坐标
            a= max(d[s[i]])
		#更新当前字段最大坐标
            m = max(m,a)
		#把判断条件放在后面，以应对第一个字母独成一段的情况
            if i == m:
                if not r: r.append(i+1)
                else: r.append(i+1-sum(r))
        return r
#优化解
#仅用字典记录字母最大坐标，且用start 和 end 提升计算效率
class Solution(object):
    
    def partitionLabels(self, s):
        """
        :type s: str
        :rtype: List[int]
        """
        last_occurrence = {}  # 存储每个字符最后出现的位置
	#enumerate 获取索引 i 和对应值 ch
        for i, ch in enumerate(s):
            last_occurrence[ch] = i

        result = []
        start = 0
        end = 0
        for i, ch in enumerate(s):
            end = max(end, last_occurrence[ch])  # 找到当前字符出现的最远位置
            if i == end:  # 如果当前位置是最远位置，表示可以分割出一个区间
                result.append(end - start + 1)
                start = i + 1

        return result

#56. 合并区间 https://leetcode.cn/problems/merge-intervals/submissions/570638993/         
#本题逻辑简单，但是有很多细节
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        r = []
	#左界排序
        intervals.sort(key = lambda x: x[0])
	#初始化当前合并区间
        s  = intervals[0][0]
        e = intervals[0][1]
        for i in range(len(intervals)):
	#元素超界则将当前区间存入解，并重新初始化区间
            if intervals[i][0] > e :
                r.append([s,e])
                s = intervals[i][0]
                e = intervals[i][1]
	#否则右界取max。这里注意要取max，否则可能会缩短区间
            else:
                e = max(intervals[i][1],e)
	#遍历完记得把最后一个区间存入
        r.append([s,e])
        return r 

#738.单调递增的数字 https://leetcode.cn/problems/monotone-increasing-digits/submissions/570901898/
#本题思路是：如果左边的一位比右边大，那么找到最左边出现这种情况的地方，把原位-1，后边填9
#但是仍然有很多细节，比如数字的位数操作，或者数字转字符串
#操作数字位数属于暴力解，用循环递减数值判断数值是否符合递增的标准，判断方法如下
    """
def checkNum(self, num):
	#当前最大值，标记最右位的值
        max_digit = 10
        while num:
		#取模后依次取最右
            digit = num % 10
            if max_digit >= digit:
                max_digit = digit
            else:
                return False
		#取除数直到位0
            num //= 10
        return True
    """
#操作字符串，因为字符串在python和大多数语言中是写在底层不可更改的，因此每操作一次，都得重新赋值
#比如让前一位减一的操作： strNum = strNum[:i - 1] + str(int(strNum[i - 1]) - 1) + strNum[i:]
#和让某一位变成9的操作：strNum = strNum[:i] + '9' + strNum[i + 1:]
#值得注意的是纯数字的字符之间还是可以比较大小的

#以下为比较优质的解法
class Solution(object):
    def monotoneIncreasingDigits(self, n):
        """
        :type n: int
        :rtype: int
        """
	#这里使用把字符串变成数组，这样就可以直接修改字符了
        s = list(str(n))
	#记录要变成9的最大位，即不单调递增从左往右第一次发生的地方
        flag = len(s)
	#区间永远包前不包后，因此是从len-1到0， step是-1
        for i in range(len(s)-1,0,-1):
            if s[i]<s[i-1]:
		#从右往左遍历，当出现了乱序，则重新标记flag，直到最左
                flag = i
		#将前一位-1，因为假如不是最左都会变成9，因此这里和上面flag一样，都是一个浮动的标记
                s[i-1] = str(int(s[i-1])-1)
        for i in range(flag,len(s)):
            s[i] = '9'
	#注意这里s是一个list，需要 ''.join(s)转化成字符串再转化成数字
        return int(''.join(s))

#968.监控二叉树 https://leetcode.cn/problems/binary-tree-cameras/
#本题思路很难临场想到
#首先是怎么遍历二叉树，其次是逻辑是什么样，最后是二叉树节点状态怎么标记
#因为从下往上，不在叶子安摄像头总体优于从上往下安（从上往下仅节省一个根节点，从下往上可能少安指数个叶子节点），所以从下往上遍历
#要做逻辑，先得标记状态，总共有三种状态，即：未被覆盖，有摄像头以及被覆盖。因为要算最小摄像头数量所以要把有摄像头单做一种状态，因此被覆盖就仅剩一种与摄像头相邻的情况
#那么1，当左右都被覆盖时，父节点为了节省可以不被覆盖，即返回0
#2当左右有未被覆盖的节点时，父节点需要安装摄像头，返回1
#3最后当左右节点中有摄像头时，父节点被覆盖了，返回2
#最后，因为情况1也涵盖了根节点，而根节点之上则没有节点了，所以要根据最终的返回值判断是否要在根节点上加摄像头
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def __init__(self):
        self.r = 0
    def sbt(self, node):
	#到了空节点，因为不想让叶子安装所以空节点计为2
        if not node:
            return 2
	#从下往上遍历
        l = self.sbt(node.left)
        r = self.sbt(node.right)
	#情况1
        if l==2 and r==2: return 0
	#情况2
        elif l==0 or r==0:
            self.r+=1
            return 1
	#情况3
        elif l==1 or r==1: return 2
        else:return -1
        
    def minCameraCover(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
	#根节点是否覆盖
        if self.sbt(root) == 0:
            self.r+=1
        return self.r
            

            


