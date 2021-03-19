package structure

type Stack []int

func (s *Stack) Push(i int) { *s = append(*s, i) }
func (s *Stack) Pop() int {
	v := (*s)[len(*s)-1]
	*s = (*s)[:len(*s)-1]
	return v
}

func (s *Stack) Peek() int     { return (*s)[len(*s)-1] }
func (s *Stack) IsEmpty() bool { return len(*s) == 0 }

type TreeStack []*TreeNode

func (s *TreeStack) Push(i *TreeNode) { *s = append(*s, i) }
func (s *TreeStack) Pop() *TreeNode {
	v := (*s)[len(*s)-1]
	*s = (*s)[:len(*s)-1]
	return v
}
func (s *TreeStack) Peek() *TreeNode { return (*s)[len(*s)-1] }
func (s *TreeStack) IsEmpty() bool   { return len(*s) == 0 }

type ListNode struct {
	Val  int
	Next *ListNode
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}
