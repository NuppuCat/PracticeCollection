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


#63. 不同路径 II https://leetcode.cn/problems/unique-paths-ii/submissions/573275252/
#这次加了障碍物，思路不变，加了很多细节
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
	#若起点或终点是石头，则动不了
        if obstacleGrid[0][0] or  obstacleGrid[len(obstacleGrid)-1][len(obstacleGrid[0])-1]:
            return 0
	#构建动态数组
        r = [[0]*len(obstacleGrid[0]) for _ in range(len(obstacleGrid))]
        for i in range(len(obstacleGrid[0])):
		#这里注意，假如第一行中间有石头，则后边的都到不了，因为只能往右下走
            if obstacleGrid[0][i]:
                   
                    break
            else:r[0][i] = 1
        for i in range(len(obstacleGrid)):
		#同理第一列也是
            if obstacleGrid[i][0]:
                    break
                    
            else:r[i][0] = 1
        for i in range(1,len(obstacleGrid)):
            for j in range(1,len(obstacleGrid[0])):
		#假如中间遇见石头则记为0即可
                if obstacleGrid[i][j]:
                    r[i][j] = 0 #continue也可
                else:
                    r[i][j] =  r[i-1][j] + r[i][j-1]
        return r[len(obstacleGrid)-1][len(obstacleGrid[0])-1]



#343. 整数拆分 https://leetcode.cn/problems/integer-break/submissions/573578197/
#本题核心在于思路
#建立dp数组，第i个就是数字几的最大乘积
#然后，每个数字从1到该数字-1遍历， 取两种情况的最大值：
#第一种是 j*(i-j) 即将该数字i，拆分成两个数字求乘积
#第二种是 j * r[i-j] 即该数字i 拆分成两个数字之后， 再将 i-j 拆分 的最大乘积
#然后 再将当前 j 和记录的最大 r[i]比较，以更新最大值
class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
	# [1] * (n+1) 不用框起来，框起来成了2维数组
        r = [1] * (n+1)
	#因为要计算n的，所以上界是n+1
	#而因为从2开始拆分有意义，所以2被初始化，动态规划从3开始
        for  i in range(3,n+1):
            for j in range(1,i):
                r[i] = max(r[i],max(j*(i-j),j*r[i-j]))
        return r[n]


#不同的二叉搜索树 https://leetcode.cn/problems/unique-binary-search-trees/submissions/573783286/
#本题在于思路和边界
#思路是：一个树以不同的根节点分配左右树，每一种情况都是所有可能的左树的情况*右树可能的情况
#而左右树的情况必然是现在树的子集，所以又可以用动态规划数组做
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
	#到n 所以数组长度n+1
	#初始化0，因为要把每种情况的积相加
        r = [0] * (n+1)
	#空集也算一种
        r[0] = 1
	#初始化到0，就从1 开始遍历
        for i in range(1,n+1):
		#这里j，代表第几个元素作为头
            for j in range(1,i+1):
		# j-1表示左支分配的元素数，i-j表示右枝分配的元素数。
		# r 代表最多的情况，所以相乘得到此种情况下的所有情况
                r[i] += r[j-1]*r[i-j]
        return r[n]






#01背包 https://kamacoder.com/problempage.php?pid=1046
#本题构建动态数组的思路很重要：
#构建行是物品，列是背包容量的二维数组

# input()接受输入字符串
#split() 根据空格拆分字符串
#map(int,..) 将列表中的每个元素转化为int
n, bagweight = map(int, input().split())

weight = list(map(int, input().split()))
value = list(map(int, input().split()))
#初始化
dp = [[0] * (bagweight + 1) for _ in range(n)]
#初始化第一个物品，即第一行的数值
#当第weight[0]列之后，由于只有一个物品，所以价值均为物品价值
for i in range(weight[0],bagweight + 1):
    dp[0][i] = value[0]

for i in range(1,n):
    for j in range(bagweight+1):
	#若背包容量小于当前物品，则最大价值等同于上面一格
        if j<weight[i]:
            dp[i][j] = dp[i-1][j]
	#若可以容纳当前物品，则对比容纳前一物品最大价值和 空出当前物品空间后的 最大价值+当前物品价值
        else:
            dp[i][j] = max(dp[i-1][j],dp[i-1][j-weight[i]]+value[i])

print(dp[n-1][bagweight])


#分割等和子集 https://leetcode.cn/problems/partition-equal-subset-sum/submissions/574231794/
#本题是01背包降维后成滚动数组的问题
#相当于是背包容量为 sum/2, 物品重量即物品价值的问题
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
	#不能整除则说明肯定不能分成两个等和子集
        if sum(nums)%2 ==1: return False
        t = int(sum(nums)/2)
	#dp数组为滚动数组，动态计算每加入一个物品，背包容量下的最大价值
        dp = [0]* (t+1)
	#挨个取出物品
        for i in nums:
		#倒序遍历更新数组
		#倒序是为了不重复放入，如果正序， dp[i-j]+i 中，前项会先被赋值，后边序列会一直重复增加i 增大
		#而每个i只能用一次。
		#所以使用倒序遍历，在该物品层，物品只被放入一次
            for j in range(t,i-1,-1):
		#对比未加入物品 i 之前的最大价值 和 放入物品 i 的最大价值
                dp[j] = max(dp[j],dp[j-i]+i)
	#因为重量和价值等价，所以dp[t]即容量为t时，最大价值就是t
	#小于t说明分割不了
        if dp[t] == t: return True
        return False


#最后一块石头的重量II https://leetcode.cn/problems/last-stone-weight-ii/submissions/574507441/
#感觉还是没抓住动态规划的要领
#本题的问题应该看作尽可能将石头分成质量接近的两堆
#这样就和上面的问题一致了
#用dp记录各个容量下背包最大的价值
#求出容量为数组和一半时的最大价值即可

class Solution(object):
    def lastStoneWeightII(self, stones):
        """
        :type stones: List[int]
        :rtype: int
        """
        t = sum(stones)//2
        dp = [0] * (t+1)

        for i in stones:
            for j in range(t,i-1,-1):
                dp[j] = max(dp[j], dp[j-i]+i)
	#求差获得分成最接近的两份的重量
        a = sum(stones)-dp[t]
        return abs(dp[t]-a)

#不能简单的通过排序求差，因为问题的本质就是将数组分成和最接近的两组
#而不是排序相减或者是找绝对值
#[31,26,33,21,40] 会输出9，而正确的答案是33和40组合获得最接近151//2=75的73，去减其它数得到5
class Solution(object):
    def lastStoneWeightII(self, stones):
        """
        :type stones: List[int]
        :rtype: int
        """
        print(stones)
        if len(stones)<2:
            a = stones[0]
            return a
        s = stones[:]
        s.sort(reverse=True)
        a = s[0]-s[1]
        s.append(a)
        
        return self.lastStoneWeightII(s[2:])


#0474.一和零 https://leetcode.cn/problems/ones-and-zeroes/submissions/575157244/
#本题本质还是01背包，构建动态数组，记录横纵向容量价值（即子集长度）最大值
#初始的想法
class Solution(object):
    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
	#做一个数组记录每个字符串的0和1
        s = [[0] * 2 for _ in range(len(strs))]
        for i in range(len(strs)):
            for j in strs[i]:
                if j=='0': s[i][0]+=1
                elif j=='1': s[i][1]+=1
        #构建dq
	#这里注意方向，如果要按01方向遍历，则for中是m
        dq = [[0] * (n+1) for _ in range(m+1)]
        for k in range(len(s)):
		#从大到小遍历容量，因为动态更新和更小容量的状态有关
            for i in range(m,s[k][0]-1,-1):
                for j in range(n,s[k][1]-1,-1):
			#对比放入本物品 和不放入本物品 的价值
                    dq[i][j] = max(dq[i-s[k][0]][j-s[k][1]] +1, dq[i][j])
        return dq[m][n]
 
#优雅版，性能提升40%
class Solution(object):
    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
	#构建dq数组
        dq = [[0]*(n+1) for _ in range(m+1)]
	#直接遍历物品，边遍历边更新
        for s in strs:
		#count函数统计字符
            zc = s.count('0')
            oc = s.count('1')
		#动态更新
            for i in range(m,zc-1,-1):
                for j in range(n,oc-1,-1):
                    dq[i][j] = max(dq[i][j], dq[i-zc][j-oc]+1)
        return dq[m][n]
        
        




