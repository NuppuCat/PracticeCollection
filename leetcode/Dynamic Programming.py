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


#70. 爬楼梯  https://leetcode.cn/problems/climbing-stairs/submissions/572335795/
#这问题本质是斐波那契数列，但是为什么需要思考
#那就是到第i层，因为步长有两种，所以有两种情况，一种是从i-1上1层
#另一种是从i-2上两层
#而上到 i-1 和 i-2 层的方法和被记到了数组中
#因此到第i 层共有 r[i]  = r[i-1]+r[i-2] 种方法
#明白了这一点就很简单了，而因为n=2 r=2,因此初始化数组为1。虽然n = 0无意义
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        r = [1]*(n+1)
        
        for i in range(2,n+1):
            r[i]  = r[i-1]+r[i-2]
        return r[n]
