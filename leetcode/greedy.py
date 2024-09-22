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
