"""
Copy from stackoverflow...
https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python
"""
from SFXGBoost.data_structure.treestructure import TreeNode, FLTreeNode
# from config import logger

class FLVisNode():
    def __init__(self, logger, FLnode: FLTreeNode):
        self.key = FLnode.owner
        self.key = str(FLnode.splittingInfo.featureName) + " " + f"{FLnode.splittingInfo.splitValue:.{2}f}"
        self.weight = FLnode.weight
        self.right = FLVisNode(logger, FLnode.rightBranch) if(FLnode.rightBranch is not None) else None
        self.left = FLVisNode(logger, FLnode.leftBranch) if(FLnode.leftBranch is not None) else None
        self.splittingInfo = FLnode.splittingInfo
        # self.nUsers = FLnode.nUsers
        self.fid = FLnode.FID
        self.score = FLnode.score
        self.logger = logger

    def display(self, treeID):
        lines, *_ = self._display_aux()
        self.logger.warning("TreeStructure, TreeID: %d", treeID)
        for line in lines:
            self.logger.warning("%s", line)
        

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            #line = 'FID: %d: ' % self.fid + '[N = %d: W = %s S: %s]' % (self.nUsers, str(self.weight), str(self.score))
            line = 'w = %.2f' % self.weight
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.key
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.key
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        #s = '%s' % self.key
        #print(self.splittingInfo.selectedFeatureID)
        #s = 'FID: %d: ' % self.fid + self.splittingInfo.get_str_split_info() 
        s = '%s' % self.key
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2


import random

# b = FLVisNode(50)
# for _ in range(50):
#     b.insert(random.randint(0, 100))
# b.display()