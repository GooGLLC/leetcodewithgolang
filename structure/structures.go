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

type Queue []int

func (q *Queue) Enqueue(i int) {
	*q = append(*q, i)
}
func (q *Queue) Dequeue() int {
	v := (*q)[0]
	*q = (*q)[1:]
	return v
}

type IntHeap []int // 定义一个类型

func (h IntHeap) Len() int { return len(h) } // 绑定len方法,返回长度

//func (h IntHeap) Less(i, j int) bool { // 绑定less方法
//	return h[i] < h[j] // 如果h[i]<h[j]生成的就是小根堆，如果h[i]>h[j]生成的就是大根堆
//}
func (h IntHeap) Less(i, j int) bool { // 绑定less方法
	return h[i] > h[j] // 如果h[i]<h[j]生成的就是小根堆，如果h[i]>h[j]生成的就是大根堆
}
func (h IntHeap) Swap(i, j int) { // 绑定swap方法，交换两个元素位置
	h[i], h[j] = h[j], h[i]
}

func (h *IntHeap) Pop() interface{} { // 绑定pop方法，从最后拿出一个元素并返回
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func (h *IntHeap) Push(x interface{}) { // 绑定push方法，插入新元素
	*h = append(*h, x.(int))
}

func (h *IntHeap) IndexOf(v int) int {
	for i := 0; i < len(*h); i++ {
		if (*h)[i] == v {
			return i
		}
	}

	return -1
}
