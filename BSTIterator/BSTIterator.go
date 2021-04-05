package BSTIterator

import (
	. "leetcodewithgolang/structure"
	"strconv"
	"strings"
)

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

/**
 * Your Codec object will be instantiated and called as such:
 * ser := Constructor();
 * deser := Constructor();
 * data := ser.serialize(root);
 * ans := deser.deserialize(data);
 */

type Codec struct {
}

func TreeSD_Constructor() Codec {
	return Codec{}
}

// Serializes a tree to a single string.
func (this *Codec) serialize(root *TreeNode) string {
	if root == nil {
		return ""
	}
	queue := make([]*TreeNode, 0)
	res := ""
	res += strconv.Itoa(root.Val) + ","
	queue = append(queue, root)
	for len(queue) > 0 {
		qsize := len(queue)
		for i := 0; i < qsize; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left == nil {
				res += "#,"
			} else {
				queue = append(queue, node.Left)
				res += strconv.Itoa(node.Left.Val) + ","
			}

			if node.Right == nil {
				res += "#,"
			} else {
				queue = append(queue, node.Right)
				res += strconv.Itoa(node.Right.Val) + ","
			}
		}
	}

	return res[:len(res)-1]
}

// Deserializes your encoded data to tree.
func (this *Codec) deserialize(data string) *TreeNode {

	if len(data) == 0 {
		return nil
	}

	v := strings.Split(data, ",")

	parseInt := func(s string) int {
		v, _ := strconv.Atoi(s)
		return v
	}

	node := &TreeNode{Val: parseInt(v[0])}
	root := node
	queue := make([]*TreeNode, 0)
	queue = append(queue, root)
	i := 1
	for len(queue) > 0 {
		root = queue[0]

		queue = queue[1:]

		leftVal := v[i]
		i++
		if leftVal != "#" {
			root.Left = &TreeNode{Val: parseInt(leftVal)}
			queue = append(queue, root.Left)
		}

		rightVal := v[i]
		i++
		if rightVal != "#" {
			root.Right = &TreeNode{Val: parseInt(rightVal)}
			queue = append(queue, root.Right)
		}
	}

	return node
}
