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


#62.不同路径 https://leetcode.cn/problems/unique-paths/submissions/572971006/
#思路上还是一致于前：即，因为机器人只能往右或者下走，那么动态数组到达m,n位置的方法总数= 到达其上边一格的方法数 + 到达其左边一格的方法数
#本题难度在于二维数组的动态规划

class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
	#用for 循环构建 二维数组
	#[[0]* n for _ in range(m)]
        r = [[0]*n for _ in range(m)]
	#到达第一列任意位置的方法只有一直往下一种，所以初始化为1
        for i in range(m):
            r[i][0] = 1
	#同理到第一行只有一直往左
        for i in range(n):
            r[0][i] = 1
	#从第二行/列开始遍历计算
        for i in range(1,m):
            for j in range(1,n):
                r[i][j] = r[i-1][j] + r[i][j-1]
        return r[m-1][n-1]

#另一种方法是求组合，仅作为参考
#不论怎么走都会走 m+n-2 步
#一定有 m - 1 步是要向下走的，不用管什么时候向下走。
#那么有几种走法呢？ 可以转化为，给你m + n - 2个不同的数，随便取m - 1个数，有几种取法。
#但是直接求分子因为阶乘很容易就会溢出
#所以要边除以分母，边往上乘
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        numerator = 1  # 分子
        denominator = m - 1  # 分母
        count = m - 1  # 计数器，表示剩余需要计算的乘积项个数
        t = m + n - 2  # 初始乘积项
        while count > 0:
            numerator *= t  # 计算乘积项的分子部分
            t -= 1  # 递减乘积项
            while denominator != 0 and numerator % denominator == 0:
                numerator //= denominator  # 约简分子
                denominator -= 1  # 递减分母
            count -= 1  # 计数器减1，继续下一项的计算
        return numerator  # 返回最终的唯一路径数

