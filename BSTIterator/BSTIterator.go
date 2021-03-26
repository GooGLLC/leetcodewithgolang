package BSTIterator

import . "leetcodewithgolang/structure"

type BSTIterator struct {
	stack TreeStack
}

func BSTConstructor(root *TreeNode) BSTIterator {
	iterator := BSTIterator{
		stack: TreeStack{},
	}

	for root != nil {
		iterator.stack.Push(root)
		root = root.Left
	}

	return iterator
}

func (this *BSTIterator) Next() int {
	top := this.stack.Pop()
	val := top.Val
	top = top.Right
	for top != nil {
		this.stack.Push(top)
		top = top.Left
	}

	return val
}

func (this *BSTIterator) HasNext() bool {
	return !this.stack.IsEmpty()
}
