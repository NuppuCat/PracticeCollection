#递归的要素
#1确定递归函数的参数和返回值： 确定哪些参数是递归的过程中需要处理的，
#那么就在递归函数里加上这个参数， 并且还要明确每次递归的返回值是什么进而确定递归函数的返回类型。

#2确定终止条件： 写完了递归算法, 运行的时候，经常会遇到栈溢出的错误，
#就是没写终止条件或者终止条件写的不对，操作系统也是用一个栈的结构来保存每一层递归的信息
#，如果递归没有终止，操作系统的内存栈必然就会溢出。

#3确定单层递归的逻辑： 确定每一层递归需要处理的信息。在这里也就会重复调用自己来实现递归的过程。
#三序遍历唯一不同的地方在于添加元素值的顺序，先遍历什么再添加元素
#前序遍历：就是二叉树顺序 上左右
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        r = []
        def sfb(node):
	#终止条件
            if node is None:
                return
            r.append(node.val)
            sfb(node.left)
            sfb(node.right)
	#这里记得引用方法开始递归
        sfb(root)
        return r
#后序遍历：就是 右左上
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        r = []
        def dfs(Node):
            if Node is None:
                return
            dfs(Node.left)
            dfs(Node.right)
            r.append(Node.val)
        dfs(root)
        return r
#中序遍历：从左到上，从上到右
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        r = []
        def sbn(Node):
            if Node is None:
                return
            sbn(Node.left)
            r.append(Node.val)
            sbn(Node.right)
        sbn(root)
        return r

#用栈来遍历
#前序中左右，入栈顺序是中右左
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        r = []
        s = [root]
        while s:
            
            c = s.pop()
            r.append(c.val)
            if c.right: s.append(c.right)
            if c.left: s.append(c.left)
        return r
#中序遍历是先到左，如果无左节点再压右
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        r = []
        s = []
        c = root
	#条件是c和s均为空
        while s or c:
            if c:
	#先将左支压栈压到底
                s.append(c)
                c = c.left
            else:
	#然后再遍历右支
	#压完左支的途中会把中点也压到
                c = s.pop()
                r.append(c.val)
                c=c.right
        return r
#后序是左右中
#因此入栈中左右
#出栈中右左
#倒序后就是左右中
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
	#先判root空
        if not root:
            return[]
        r = []
        s = [root]
        while s:
            c=  s.pop()
            r.append(c.val)
	#注意先左后右
            if c.left: s.append(c.left)
            if c.right: s.append(c.right)
           	#倒序
        return r[::-1]


#102.二叉树的层序遍历
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
	这个地方假如使用 collections.deque().append(root) 是存储不到的
	这是因为 collections.deque().append(root) 试图对一个临时创建的空双端队列调用 append 方法，并没有将结果存储下来。
	但是使用分开的语句是可以的
        e  = collections.deque()
        e.append(root)
	假如这样，会把root的内部结构存入deque :  e = collections.deque(root)
	所以需要将其数组化[root]
        print(q)
	q是正常的
        print(len(q))
	1
        print(collections.deque().append(root))
	none
        print(e)
	和q一样
        """
        if not root:
            return []
        r = []
	#注意构建初始化的方式，临时的空双队列需要直接放入元素
        q = collections.deque([root])
        
        while q:
	#层级元素存储
            l  =[]
	#遍历一层的元素，存入l
            for _ in range(len(q)):
                c = q.popleft()
                l.append(c.val)
	#下一层级的元素存入队列
                if c.left: q.append(c.left)
                if c.right: q.append(c.right)
            r.append(l)
        return r

#226.翻转二叉树
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None

        def rev(Node):
            Node.left,Node.right = Node.right,Node.left
            if Node.left: rev(Node.left)
            if Node.right: rev(Node.right)
        rev(root)
        return root

#101. 对称二叉树
#递归先判断左右为空的情况，然后再判断值，若值相同递归判断子节点
#值得注意的是要判断的是左树的左枝和有树的右枝，以及左树的右枝与右树的左支是否相同
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def cop(l,r):
            if l ==None and r!=None: return False
            elif l!=None and r==None: return False
            elif l==None and r==None: return True
            elif l.val!= r.val: return False

            return cop(l.left,r.right) and cop(l.right,r.left)
        if not root:
            return True
        return cop(root.left,root.right) 
        
#104.二叉树的最大深度    用递归：写递归就是用最简单的情况下，先确定参数和返回值以及终止条件，再在最简单的情况下写一个逻辑
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        d = 0
        def sbt(Node):
	#结束条件是节点为空
            if not Node:
                return 0
	#遍历左右树的深度后选出最深的一个加上父节点的深度返回
            dl = sbt(Node.left)
            dr = sbt(Node.right)
            d = 1+max(dl,dr)
            return d
        return sbt(root)

        
#559.n叉树的最大深度
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: Node
        :rtype: int
        """
        
        def md(Node):
            if not Node:
                return 0
            d= 0
	#遍历对比每个子枝的深度，用d存储最深的那个
            for c in Node.children:
                d=max(md(c),d)
	#加上1
            return 1+ d
        return md(root)      

#111.二叉树的最小深度
#这里注意一个逻辑，最小深度是根节点到最近叶子节点的距离
#因此要判断左右树是否为空，左为空则叶子节点为1+右树深度，右为空则叶子节点为1+左树深度
#都不空则1+min(l,r) 
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        d = 0
        
        def sbt(Node):
            if not Node:
                return 0
            elif not Node.left and Node.right:
                return 1+sbt(Node.right)
            elif not Node.right and Node.left:
                return 1+sbt(Node.left)
            else:
                l = sbt(Node.left)
                r  =sbt(Node.right)

                return 1+min(l,r) 
        return sbt(root)

#222.完全二叉树的节点个数
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        elif not root.left and not root.right:
            return 1
        
        def sbt(Node):
            if not Node:
                return 0
            #这里不需要条件判断左右枝是否存在，因为为0就是0
            a = sbt(Node.left)
            b = sbt(Node.right)
            return a+b+1
        return sbt(root)
#简化代码，将递归放在原方法中
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
	#注意这里引用自身要self
        return self.countNodes(root.left)+self.countNodes(root.right)+1


#110.平衡二叉树
#逻辑很重要，入参与终止条件不变
#逻辑在于设定的是计算树的高度，然后对比树的高度
#关键是高度似乎和要递归的逻辑目标有差异，所以设置一个不可能出现的值-1，作为条件外的标记
#如果子树高度不符合条件，返回-1
#如果两子树高度差大于1，返回-1
#如果一切正常返回树的高度 1+max（l,r）
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        
        def sbt(Node):
            if not Node:
                return 0
            l = sbt(Node.left)
            
            r = sbt(Node.right)
            if l==-1 or r==-1 or abs(l-r)>1:
                return -1
            return 1+max(l,r)
        a = sbt(root)
        return a!=-1        



##0257.二叉树的所有路径.md
#构建递归：前序遍历，遇到叶子节点则储存路径
#值得注意的是递归时路径需要回溯一步
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):

    def sbt(self,Node,p,r):
	#前序遍历，先入数组
            p.append(Node.val)
            if not Node.left and not Node.right:
                #这里不能直接append p,因为递归时内存地址不变，最终会导致错误的结果
	#python 2和3都不能使用copy()函数，因此使用slic p[:]来进行复制数组
                #q = p.copy()
                q= p[:]
                r.append(q)
                
                return
            if Node.left: 
                self.sbt(Node.left,p,r)
	#递归一次，溯回一次，就把树想成最简单的树即可
	#想成复杂的就是，一次递归会直接到叶，然后一层一层往回退
	#self.traversal(cur.left, path[:], result) 这种是隐回溯法，就是不改变p本身，只使用p的值，因为p没变所以可以不用回溯
	#比较优雅
	#但是不好理解，需要死记，还是加上回溯比较好。
                p.pop()
            if Node.right:
                self.sbt(Node.right,p,r)
                p.pop()
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        r  =[]
        p=[]
        if not root:
            return r
        self.sbt(root,p,r)
        print(r)
        r2=[]
        for  i in r:
	#复习join 的用法
	#这里用了map（），构建str格式i元素的映射
            r2.append('->'.join(map(str, i)))
        return r2
    

#404.左叶子之和
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        r = []
        def sbt(Node):
            if not Node:
                return 0
	#主要是这个判断条件，如果是左支且叶子，添加值即可
            if Node.left and not Node.left.left and not Node.left.right:
		#注意是添加左支的值
                r.append(Node.left.val) 
            sbt(Node.left)
            sbt(Node.right)
        
        sbt(root)
        print(r)
        return sum(r)

#513.找树左下角的值
#最深的最左边
#层序遍历即可
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def findBottomLeftValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        r = 0
        q = collections.deque([root])
        while q:
            l = []
            for _ in range(len(q)):
                n = q.popleft()
                
                l.append(n.val)
                if n.left: q.append(n.left)
                if n.right: q.append(n.right)
            r = l[0]
        return r

#112. 路径总和：给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。
#逻辑和找到所有路径一样
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def hasPathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: bool
        """
        r = []
        p = []
	#核心在于构建一个栈 p，来存储路径
        def sbt(Node,p,r):
            
            if not Node:
                return 
            p.append(Node.val)
            if not Node.left and not Node.right:
                #遇到叶子节点则存储路径
                q = p[:]
                r.append(sum(q))
                return
            
            if Node.left:
                #调用一次递归找一个叶子节点
                sbt(Node.left,p,r)
		#回溯一次弹一个栈
                p.pop()
            if Node.right:

                sbt(Node.right,p,r)
                p.pop()
        sbt(root,p,r)    
        return targetSum in r
#精简版代码
def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
	#叶子节点且值正确
        if not root.left and not root.right and sum == root.val:
            return True
	#这里的逻辑是：左树或者右树
	#传递sum-根值 差相等的节点值
        return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)


#105.从前序与中序遍历序列构造二叉树
#从前序和中序中拆出左右树来进行递归
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if not preorder:
            return None
	#根节点值
        r = preorder[0]
	#在中序中找到根节点
        ii = inorder.index(r)
        root = TreeNode(r)
	#中序是左中右
	
	#左右子树分割
        il = inorder[:ii]
	#注意这里要+1，跳过根节点
        ir = inorder[ii+1:]
	#分割前序子树
	#跳过根节点，长度要和中序左树一致，因此上界也要+1
        pl = preorder[1:1+len(il)]
	#从上界开始
        pr = preorder[1+len(il):]
	#递归
        root.left = self.buildTree(pl,il)
        root.right  = self.buildTree(pr,ir)
        return root
#106.从中序与后序遍历序列构造二叉树
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if not inorder:
            return None
        r  = postorder[-1]
        root = TreeNode(r)
        
        #中序拆分子树逻辑一致
        mi  = inorder.index(r)
        il = inorder[:mi]
        # +1 skip the root
        ir = inorder[mi+1:]
	#后序是，左右中
	#因此如下划分
        #左树长度和中序一致
        pl  = postorder[:len(il)]
	#到末尾不加入根节点
        pr  = postorder[len(il):len(postorder)-1]

        root.left = self.buildTree(il,pl)
        root.right  =self.buildTree(ir,pr)

        return root


#654.最大二叉树
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
	#若无数组，返回空
        if not nums:
            return None
	#若有，找到最大值和序列，构建根节点
        m = max(nums)
        i = nums.index(m)
        root = TreeNode(m)
	#递归传递子数组
        root.left = self.constructMaximumBinaryTree(nums[:i])
        root.right = self.constructMaximumBinaryTree(nums[i+1:])
        return root
                
