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
