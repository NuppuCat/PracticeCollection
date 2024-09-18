#第77题. 组合 https://leetcode.cn/problems/combinations/
#本题思路在于，当k很大时，for循环将难以书写，因此使用递归
#把组合看成一个遍历树的问题
#核心思想在于，用递归控制深度，用循环控制广度
class Solution(object):
    def __init__(self):
	#q存储当前路径
        self.q = []
        self.r = []
	#搜索方法需要起始数字id，来控制遍历不重复
    def sbt(self,n,k,id):
	#若路径深度达到了k，就可以返回
        if len(self.q)==k:
		#注意要[:]
            self.r.append(self.q[:])
            return
	#从起始到n+1，因为range包前不包后
        for i in range(id,n+1):
            #先放入路径
            self.q.append(i)
	    #再向深处理
            self.sbt(n,k,i+1)
	    #然后弹出这支，即回溯一步
            self.q.pop()
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        self.sbt(n,k,1)
        return self.r


#优化横向循环提升效率

class Solution(object):
    def __init__(self):
        self.q = []
        self.r = []
    def sbt(self,n,k,id):
        if len(self.q)==k:
            self.r.append(self.q[:])
            return
	#主要思想是看剩余长度n-i 是否够所需的长度k-len(q)
#得到关于i的不等式。重定边界+2
        for i in range(id,n-(k-len(self.q))+2):
            
            self.q.append(i)
            self.sbt(n,k,i+1)
            self.q.pop()
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        self.sbt(n,k,1)
        return self.r

#216.组合总和III https://leetcode.cn/problems/combination-sum-iii/
#这就多了点条件，使用剪枝就可以解决
class Solution(object):
    def sbt(self,k,n,q,r,s):
        if sum(q)>n or len(q)>k:
            return
        if len(q)==k:
            if sum(q)==n:
                r.append(q[:])
            else:
                return
        
        for i in range(s,9 - (k - len(q)) + 2):
            if i in q:
                continue
            q.append(i)
            self.sbt(k,n,q,r,i+1)
            q.pop()
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        q = []
        r = []
        self.sbt(k,n,q,r,1)
        return r


#17.电话号码的字母组合 https://leetcode.cn/problems/letter-combinations-of-a-phone-number/
#这里主要是建一个数组来存储数字对应的字母，以方便查表
#剩余思想和组合一致
class Solution(object):
    def __init__(self):
        self.letterMap = [
            "",     # 0
            "",     # 1
            "abc",  # 2
            "def",  # 3
            "ghi",  # 4
            "jkl",  # 5
            "mno",  # 6
            "pqrs", # 7
            "tuv",  # 8
            "wxyz"  # 9
        ]
        self.result = []
        self.s = ""
    def sbt(self,digits,start):
        if len(self.s)==len(digits):
		#这里由于字符串存储在底层而非对象，所以可以直接append
            self.result.append(self.s)
            return
	#注意这里digits是字符串，因此要转换成int才能作为index
        i  = int(digits[start])
        for j in self.letterMap[i]:
         
            self.s+=j
            self.sbt(digits,start+1)
		#巧用[:-1]删除最后一位字符
            self.s =self.s[:-1]

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
	#这里需要判空，否则空组中会放一个空字符串
        if not digits:
            return []
        self.sbt(digits,0)
        return self.result


#39. 组合总和 https://leetcode.cn/problems/combination-sum/
class Solution(object):
    def __init__(self):
        self.q = []
        self.r = []
    def sbt(self,candidates,target,s):
        c = self.q[:]
        if sum(c)>target:
            return
        if sum(c)==target:
            self.r.append(c)
	#由于组合不能重复，所以仍然需要初始目录，来防止组合变序重复
        for i in range(s,len(candidates)):
            self.q.append(candidates[i])
	#由于元素可以重复，所以初始从i开始，而非i+1
            self.sbt(candidates,target,i)
            self.q.pop()

    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        self.sbt(candidates,target,0)
        return self.r

#40.组合总和II https://leetcode.cn/problems/combination-sum-ii/
#这里多了一个关于重复元素的问题，之前的题目中candidates元素可以复用，且元素无重复
#本题核心问题在于，会有重复的元素出现， 因此同元素在同一层的组合就会重复
#因此，就是先将candidates排序，使重复的元素挨在一起，然后判断当前元素是否和前一元素相同
#若相同则跳过组合，即可避免重复
class Solution(object):
    def __init__(self):
        self.q = []
        self.r = []
    def sbt(self, s , candidates, target):
        c  = self.q[:]
        if sum(c) > target:
            return
        if sum(c)== target:
            self.r.append(c)
            return
        for i  in range(s,len(candidates)):
		#判断是否相同元素出现
            if i > s and candidates[i] == candidates[i - 1]:
                continue
		#这一步是为了效率，也可以不写
            if  sum(c) +candidates[i]>target:
                break
		
            self.q.append(candidates[i])
		#元素不能复用，因此递归初始目录为i+1
            self.sbt(i+1,candidates,target)
            self.q.pop()
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
	#最重要的一步：排序为了使重复元素相邻
        candidates.sort()
        self.sbt(0,candidates,target)
        return self.r


#131.分割回文串 https://leetcode.cn/problems/palindrome-partitioning/submissions/563355593/
#所谓回文串就是从左右读起都一样的字符串
#这里有两个问题，第一个是需要判断是否回文
#第二个是终止条件和运行逻辑
class Solution(object):
    def __init__(self):
        self.r = []
        self.q = []
	#写方法判断是否回文
    def isPalindrome(self,s,start,end):
	#用while 循环不用考虑上下界，更简单
        i  = start
        j = end
        while i<j:
            if s[i]!=s[j]:
                return False
            i+=1
            j-=1
        return True
    def sbt(self,start,s):
	#终止条件为：如果初始目录已经比字符串长，则说明这轮搜索已经结束
        if start>= len(s):
            self.r.append(self.q[:])
            return
	#单层逻辑为，假如start 到 i 是回文串，则放入q，然后继续搜索下一层
	#若不是则无处理
	#单字母不会遗漏
        for i in range(start,len(s)):
            if self.isPalindrome(s,start,i):
                self.q.append(s[start:i+1])
                self.sbt(i+1,s)
                self.q.pop()
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
	#判空
        if not s:
            return []
        self.sbt(0,s)
        return self.r


#93.复原IP地址 https://leetcode.cn/problems/restore-ip-addresses/
#本题的暴力解法也有很多细节
#需要一个判断是否合法ip的方法
#需要判定临时串q是否为空串
class Solution(object):
    def __init__(self):
        self.q =""
        self.r = []
    def isip(self, st):
        qs = self.q.split('.')
        #ip只有四段 这里可以写在sbt里进行优化效率
        if len(qs)!= 4:
            return False
        for i in qs:
	#超限了不合法
            if int(i)>255 or i<0:
                return False
	#首字为0不合法
            if int(i[0])==0 and len(i)>1:
                return False
        return True
    def sbt(self, s, start):
        
        if start>=len(s):
            
            if self.isip(self.q):
                self.r.append(self.q)
            return
        for i in range(start,len(s)):
           #串的上界要到i+1，因为包前不包后，否则0：0不出数，并且方括号里注意写：而非，
            cs = s[start:i+1]
     	
            if int(cs)>255:
                break
	  #拼接要注意是否第一次接，是否加点
            if not self.q:
                self.q+=cs
            else:
                self.q=self.q+'.'+cs
            self.sbt(s,i+1)
	#回溯同样要看
            if '.' in self.q:

                self.q = self.q[:len(self.q)-1-len(cs)]
            else:
                self.q = ''
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        if not s:
            return []
        self.sbt(s, int(0))
        return self.r



#78.子集 https://leetcode.cn/problems/subsets/submissions/564056061/
#子集，要去重，且分长短
#因此有startindex和length两个维度需要控制
class Solution(object):
    def __init__(self):
        self.q=[]
        self.r = []
    def sbt(self,nums,s,l):
	#当当前q长度达到长度，则加入
        c = self.q[:]
        if len(c)>=l:
            self.r.append(c)
            return
	#这里逻辑照常即可，因为上面判断了q的长度作为终止条件
        for i in range(s,len(nums)):
            
            self.q.append(nums[i])
            self.sbt(nums,i+1,l)
            self.q.pop()


    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if not nums:
            return [[]]
	#这里进行维度控制，遍历所有维度即可
        for i in range(0,len(nums)+1):
            self.sbt(nums,0,i)
    
        return self.r
#也可以这样写，就是每变动一次q
#就往r里添加一次
#因为求全集就是求树的每一个节点
#由于for循环中隐含了终止条件，所以可以省略终止条件
class Solution(object):
    def __init__(self):
        self.q=[]
        self.r = []
    def sbt(self,nums,s):
        c = self.q[:]
        self.r.append(c)
        
        for i in range(s,len(nums)):
            
            self.q.append(nums[i])
            self.sbt(nums,i+1)
            self.q.pop()


    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        
        self.sbt(nums,0)
    
        return self.r



#90.子集II https://leetcode.cn/problems/subsets-ii/submissions/564405952/
#这次nums中有重复的数字了，因此和前面的一样，要先排序，再判断是否和前一个一致再进入逻辑
class Solution(object):
    def __init__(self):
        self.q = []
        self.r = []
    def sbt(self,nums,s):
        self.r.append(self.q[:])

        for i in range(s,len(nums)):
	#这里会先判断i>s 因此不会报错
	#and 语句会先判断第一个是否是T,是T才会看后半句
	#而or语句是看第一个是否是F,是F才会看后半句
	#当然这里换了条件顺序也不影响，因为nums[-1]也是合法的
            if i>s and nums[i]==nums[i-1]:
                continue
            self.q.append(nums[i])
            self.sbt(nums,i+1)
            self.q.pop()

    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
	#先排序
        nums = sorted(nums)
        self.sbt(nums,0)
        return self.r



#491.递增子序列 https://leetcode.cn/problems/non-decreasing-subsequences/submissions/564698536/
#这个问题在于，不能给序列排序，因此需要一个set进行去重，而不能简单的看是否相邻的元素相同
#这个set应该是放在每一层中间，是同一层不能有重复的
class Solution(object):
    def __init__(self):
        self.q=[]
        self.r =[]
    def sbt(self,nums,s):
        c = self.q[:]
	#序列长度至少为2
        if len(c)>=2:
            self.r.append(c)
	#给定set，记录适用过的元素
        uset = set()
        for i in range(s,len(nums)):
	#这里的判断条件：
	#q尾元素小于新元素或q为新队列时添加
	#且新元素是没有出现过的
	#否则跳过这个元素
            if (self.q and nums[i] < self.q[-1]) or nums[i] in uset:
                continue
	#set 添加用add
            uset.add(nums[i])
            self.q.append(nums[i])
            self.sbt(nums,i+1)
            self.q.pop()

    def findSubsequences(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
    
        self.sbt(nums,0)
        return self.r



#46.全排列 https://leetcode.cn/problems/permutations/submissions/564954881/
#求排列，因此不需要startindex去重
#但是需要每次检查是否已使用元素
#由于元素无重复，因此直接判断是否再当前队列中即可
#当元素有重复时，则维护一个used index数组标记即可
class Solution(object):
    def __init__(self):
        self.q=[]
        self.r = []
    def sbt(self,nums):
        c = self.q[:]
	#到达限定长度则添加
        if len(c)==len(nums):
            self.r.append(c)
            return	
        for i in range(len(nums)):
	#for 直接从头遍历，若元素已在q中，则跳过
            if nums[i] in self.q:
                continue
            
            self.q.append(nums[i])
            self.sbt(nums)
            self.q.pop()
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        self.sbt(nums)
        return self.r
#47.全排列 II  https://leetcode.cn/problems/permutations-ii/submissions/565154847/
#排列去重元素
#逻辑基本相同
class Solution(object):
    def __init__(self):
        self.q=[]
        self.r = []
    def sbt(self,nums,iq):
        c = self.q[:]
        if len(c)==len(nums):
            self.r.append(c)
            return
        
        for i in range(len(nums)):
           
            if not iq[i]:
                continue
            #这里要注意，要有一个 iq[i-1]==False 的条件，若没有这个，则会少取一个重复的元素，进而收集不到有用的排列
	    #另外， 条件为iq[i-1]==True也是可以的。 为True时是数层上对前一位去重，False则是树枝上去重。树层上去重更高效，但不好理解
            if i>0 and nums[i]==nums[i-1] and iq[i-1]==False:
                
                continue
            
            iq[i] = False
            self.q.append(nums[i])
            self.sbt(nums,iq)
            self.q.pop()
            iq[i] = True
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        iq = []
        for i in range(len(nums)):
            iq.append(True)
        self.sbt(nums,iq)
        return self.r

#51. N皇后 https://leetcode.cn/problems/n-queens/
#二维数组的棋盘上，横竖斜不能有两个Q
#这个问题复杂了很多：首先是如何构建回溯问题
#把棋盘上行作为深度，列作为层，每次检查是否可以放置棋子，就变成了回溯问题
#其次是终止条件，和之前不同，是标记一个处理行的索引，维度达到n则意味着棋盘已经遍历完毕，n+1时终止
#然后是很多判断是否合法的细节：由于行内是在循环中且都会回溯，因此行内不需要判断
#列内的判断相对简单，只需要判断之前的列上有没有出现Q
#斜方向的判断则需要一些想象力，由于是从上往下排，仅判断左上和右上有没有Q即可，但是二维数组的遍历方法咋一看是不容易想出来的
class Solution(object):
    #定义方法，制定规则来判断是否可以在棋盘q的某位置放置Q
    def isvalid(self,row,col,q):
	#检查列上有无Q
        for i in range(row):
            if q[i][col] == 'Q':

                return False
	#检查左上有无Q
	#这里用while条件更合适，左上意味着 -1 -1
        i,j = row-1,col-1
        while i >=0 and j>=0:
            if q[i][j] == 'Q':
                return False
            i-=1
            j-=1
	#检查右上有无Q
	#右上意味着 -1 +1
        i,j = row-1,col+1
        while i>=0 and j<len(q):
            if q[i][j] == 'Q':
                return False
            j+=1
            i-=1
	#记得符合规则返回True
        return True
    def sbt(self,row,n,q,r):
        #深度达到棋盘边界则记录棋盘（当前树轨迹）
        if row==n:
            c = []
		#由于棋盘定义为二维数组，所以需要遍历拼接字符串
            for l in q:
                row = ''
                for i in l:
                    row+=i
                c.append(row) 
            r.append(c)
            return
	#层内遍历
        for col in range(n):
           
            if self.isvalid(row,col,q):
                
                q[row][col] = 'Q'
		#深入 row+1
                self.sbt(row+1,n,q,r)
		#回溯
                q[row][col] = '.'
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
	#因为字符串不能直接操作更改索引位置
	#为了递归单层逻辑的方便直接把棋盘构筑成字符的二维数组
        q = [['.']*n for _ in range(n)]
        r= []
        self.sbt(0,n,q,r)
        return r
