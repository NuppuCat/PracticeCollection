#617.合并二叉树
#递归遍历二叉树，合并两树位置值
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def mergeTrees(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: TreeNode
        """
        if not root1:
            return root2
        if not root2:
            return root1
        root = TreeNode(root1.val+root2.val)
        root.left = self.mergeTrees(root1.left,root2.left)
        root.right = self.mergeTrees(root1.right,root2.right)
        return root

#700.二叉搜索树中的搜索
#二叉搜索树，根的左支比根小，根的右枝比根大
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def searchBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        def sbt(root,val):
		#退出条件是为空或者搜到
            if not root or root.val == val:
                return root
		#定义返回值
            r = None
		#若搜索值大于根则搜右枝，小于根则搜左支
            if root.val>val: r= sbt(root.left,val)
            if root.val<val: r =sbt(root.right,val)
            return r

        return sbt(root,val)

#98.验证二叉搜索树
#二叉搜索树是，从左到右依次变大
#因此中序遍历， 左中右，看是否递增
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def __init__(self):
        self.m = float('-inf')
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        
        
        if not root:
            return True
	#中序遍历
        left = self.isValidBST(root.left)
	#预定义一直存储最左的值
	
        if self.m < root.val:
            self.m = root.val
	#若出现左值比节点值大，则说明不递增返回False
        else:
            return False

        right = self.isValidBST(root.right)
        
        return left and right

#530.二叉搜索树的最小绝对差
#因为直接递归搜索会有返回值的问题
#因此考虑先把二叉搜索树转化成递增数组
#然后找出数组相邻最小差
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
	#构建数组
    def __init__(self):
        self.q = []
	#中序遍历
    def sbt(self,root):
        if not root:
            return
        self.sbt(root.left)
        self.q.append(root.val)
        self.sbt(root.right)
    def getMinimumDifference(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.sbt(root)
        if len(self.q)<2:
            return
        r = float('inf')
        q = self.q[:]
        for i in range(len(q)-1):
           # if q[i+1]-q[i]<r:
            #    r = q[i+1]-q[i]
	   #可以用min函数简化判断过程
		 r = min(r,q[i+1]-q[i])
		
        return r


#501.二叉搜索树中的众数
#就是出现频率最高的元素
#这里搜索做一个方法
#统计做一个方法
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sbt(self,root,d):
        if not root:
            return
        self.sbt(root.left,d)
	#复习
        d[root.val] = d.get(root.val,0)+1
        self.sbt(root.right,d)
    def findMode(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        #defaultdict(int) 这种申明更简单，可以直接对新key+1
        d = defaultdict(int)
        self.sbt(root,d)
        mf = max(d.values())
        r = []
        for k in d.keys():
            if d[k] == mf:
                r.append(k)
        return r


#236. 二叉树的最近公共祖先 （M）
#
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if  root is None or root==p or root==q:

            return root
        left = self.lowestCommonAncestor(root.left,p,q)
        right = self.lowestCommonAncestor(root.right,p,q)
        if left is not None and right is not None:
            return root
        elif left is not None and  right is None:
            return left
        elif right is not None and  left is None:
            return right
        else:
            return None

#235. 二叉搜索树的最近公共祖先
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root.val < p.val and root.val<q.val:
            return self.lowestCommonAncestor(root.right,p,q)
        elif root.val >p.val and root.val >q.val:
            return self.lowestCommonAncestor(root.left,p,q)
        else:
            return root
#701.二叉搜索树中的插入操作
#https://leetcode.cn/problems/insert-into-a-binary-search-tree/
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def insertIntoBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
	#若位置为空则构建节点插入
        if not root:
            root = TreeNode(val)
            return root
	#递归寻找位置，利用sbt的特性
        if root.val > val:
            root.left = self.insertIntoBST(root.left,val)
        if root.val < val:
            root.right  = self.insertIntoBST(root.right,val)
	#插入完成后返回根节点
        return root


#0450.删除二叉搜索树中的节点.https://leetcode.cn/problems/delete-node-in-a-bst/
#删除节点，要考虑各种情况
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def deleteNode(self, root, key):
        """
        :type root: TreeNode
        :type key: int
        :rtype: TreeNode
        """
        if not root:
            return root
            
        if root.val == key: 	#找到了节点
            if root.left and not root.right:      #只有左支或右枝，直接返回子节点
                return root.left
            elif root.right and not root.left:
                return root.right
            elif not root.left and not root.right: #为叶子节点直接删除
                return None
            else: 	#都不为空时，找到右子树的最左节点
                c = root.right  
                while c.left:
                    c = c.left
                c.left= root.left	#然后将根的左支接到最左节点
                return root.right		#返回右枝替代原根节点
        if key>root.val:
            root.right = self.deleteNode(root.right,key)	#注意这里，因为是带返回值的递归，需要返回值把树连接，替代节点原有的位置
        if key<root.val:
            root.left = self.deleteNode(root.left,key)

        return root
        
#669. 修剪二叉搜索树  https://leetcode.cn/problems/trim-a-binary-search-tree/
#给定值的上下界修建树，保持树结构
#主要思考两层：1，根节点替换 2，子节点替换
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def trimBST(self, root, low, high):
        """
        :type root: TreeNode
        :type low: int
        :type high: int
        :rtype: TreeNode
        """
        if not root:
            return None
	#若根越界，返回左右枝作为新的根节点
        if root.val<low:
            root.right = self.trimBST( root.right, low, high)
            return root.right
        if root.val>high:
            root.left = self.trimBST(root.left,low,high)
            return root.left
	#再对子树过滤
        root.left = self.trimBST(root.left,low,high)
        root.right = self.trimBST( root.right, low, high)
        return root    


#108.将有序数组转换为二叉搜索树 https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/
#有序，且平衡，即中间是根
#有了逻辑再确定左右枝递归即可
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return
        n = len(nums)
        root = TreeNode(nums[n/2])
        l = nums[:n/2]
        r = nums[n/2+1:]
        root.left = self.sortedArrayToBST(l)
        root.right = self.sortedArrayToBST(r)
        return root



#538.把二叉搜索树转换为累加树 https://leetcode.cn/problems/convert-bst-to-greater-tree/
#这里逻辑主要在于记录一个当前累加值
#确定了这个再确定遍历顺序为右中左，基本上就已经解决了
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
	#也可以把变量申明放到函数中
    def __init__(self):
        self.c =0
	#注意函数申明要传入self
    def sbt(self,cur):
        if not cur:
           return
        self.sbt(cur.right)
        cur.val = self.c+cur.val
        self.c  = cur.val
        self.sbt(cur.left)
        

    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        self.sbt(root)
        return root
