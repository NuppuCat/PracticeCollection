#如果某一问题有很多重叠子问题，使用动态规划是最有效的。
#所以动态规划中每一个状态一定是由上一个状态推导出来的

#509. 斐波那契数 https://leetcode.cn/problems/fibonacci-number/submissions/571855137/
#简单的规划原则
class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
	#注意边界
        if n<2:
            return n
        F = [0] * (n+1)
        F[1] = 1
        for i in range(2,n+1):
            F[i] = F[i-1]+F[i-2]
        return F[n]
