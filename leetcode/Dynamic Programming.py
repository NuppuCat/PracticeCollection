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

#746. 使用最小花费爬楼梯 https://leetcode.cn/problems/min-cost-climbing-stairs/submissions/572592459/
#总体思路是，用动态数组记录到达第i阶的最小花费
#那么每一阶的最小花费就是到达 前一阶 和 前两阶 的花费，加上它们相对应的花费

class Solution(object):
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
	#0，1可以直接跨过，所以从2开始遍历.即第一步无消耗
	#相当于到哪一步最小花多少
        r = [0] * (len(cost)+1)
        
        for i in range(2,len(cost)+1):
            r[i]  = min(r[i-2]+cost[i-2],r[i-1]+cost[i-1])
        return r[-1]

#这个解是最开始写出来的
#第一步有消耗，最后一步无消耗
#相当于到了先消耗所立的数字，计算到下一步所需的总消耗最小
class Solution(object):
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        r = [0] * (len(cost)+1)
        r[1] = cost[0]
        for i in range(2,len(cost)+1):
            r[i]  = min(r[i-2]+cost[i-1],r[i-1]+cost[i-1])
	#到达倒数第一二阶，比较大小
        return min(r[-1],r[-2])
