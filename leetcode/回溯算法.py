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
