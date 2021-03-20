package main

import (
	"fmt"
	"leecodewithgolang/structure"
	"leecodewithgolang/util"
	"sort"
	"strconv"
	"strings"
)

type MinStack struct {
	stack    []int
	minStack []int
}

/** initialize your data structure here. */
func MinStackConstructor() MinStack {
	s := MinStack{
		stack:    make([]int, 0),
		minStack: make([]int, 0),
	}
	return s
}

func (this *MinStack) Push(val int) {
	this.stack = append([]int{val}, this.stack...)
	if len(this.minStack) == 0 || val <= this.GetMin() {
		this.minStack = append([]int{val}, this.minStack...)
	}
}

func (this *MinStack) Pop() {
	top := this.stack[0]
	this.stack = this.stack[1:]
	if top == this.GetMin() {
		this.minStack = this.minStack[1:]
	}
}

func (this *MinStack) Top() int {
	return this.stack[0]
}

func (this *MinStack) GetMin() int {
	return this.minStack[0]
}

func getIntersectionNode(headA, headB *structure.ListNode) *structure.ListNode {
	alen := listLength(headA)
	blen := listLength(headB)
	if alen >= blen {
		diff := alen - blen
		for i := 0; i < diff; i++ {
			headA = headA.Next
		}

		for headA != headB {
			headA = headA.Next
			headB = headB.Next
		}

		return headA
	} else {
		return getIntersectionNode(headB, headA)
	}
}

func listLength(head *structure.ListNode) int {
	if head == nil {
		return 0
	} else if head.Next == nil {
		return 1
	} else {
		return 1 + listLength(head.Next)
	}
}

func findMin(nums []int) int {
	nlen := len(nums)
	if nlen == 1 {
		return nums[0]
	}

	lo := 0
	hi := nlen - 1
	for lo < hi {
		if nums[lo] < nums[hi] {
			return nums[lo]
		}

		if hi-lo == 1 {
			return util.min(nums[lo], nums[hi])
		}

		mid := lo + (hi-lo)/2
		if nums[mid] > nums[lo] {
			lo = mid + 1
		} else {
			hi = mid
		}
	}

	return nums[lo]
}

func maxProduct(nums []int) int {
	nlen := len(nums)
	minP := make([]int, nlen)
	maxP := make([]int, nlen)
	best := nums[0]
	for i := 0; i < nlen; i++ {
		minP[i] = nums[0]
		maxP[i] = nums[0]
		if i > 0 {
			minP[i] = util.min(minP[i], util.min(minP[i-1]*nums[i], maxP[i-1]*nums[i]))
			maxP[i] = util.max(maxP[i], util.max(minP[i-1]*nums[i], maxP[i-1]*nums[i]))
			best = util.max(maxP[i], best)
		}
	}

	return best
}

func sortList(head *structure.ListNode) *structure.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	middle := getMiddle(head)
	return mergeSortedList(sortList(head), sortList(middle))
}

func mergeSortedList(p1 *structure.ListNode, p2 *structure.ListNode) *structure.ListNode {
	res := &structure.ListNode{}
	cur := res
	for p1 != nil || p2 != nil {
		if p1 != nil && p2 != nil {
			if p1.Val < p2.Val {
				cur.Next = p1
				cur = cur.Next
				p1 = p1.Next
			} else {
				cur.Next = p2
				cur = cur.Next
				p2 = p2.Next
			}
		} else if p1 == nil {
			cur.Next = p2
			break
		} else {
			cur.Next = p1
			break
		}
	}

	return res.Next
}

func getMiddle(head *structure.ListNode) *structure.ListNode {
	if head == nil || head.Next == nil {
		return nil
	} else {
		p1 := &structure.ListNode{}
		p1.Next = head
		p2 := p1
		for p2 != nil && p2.Next != nil && p2.Next.Next != nil {
			p1 = p1.Next
			p2 = p2.Next.Next
		}

		p1.Next = nil
		return p2
	}
}

func wordBreak(s string, wordDict []string) []string {
	res := make([]string, 0)
	cur := make([]string, 0)
	dict := make(map[string]bool)
	for _, v := range wordDict {
		dict[v] = true
	}

	wordBreakDFS(&res, &cur, 0, dict, s)
	return res
}

func wordBreakDFS(res *[]string, cur *[]string, start int, dict map[string]bool, s string) {
	if start == len(s) {
		*res = append(*res, strings.Join(*cur, " "))
	} else {
		for i := 0; i < 10; i++ {
			endIndex := start + i
			if endIndex < len(s) {
				nextWord := s[start : endIndex+1]
				if dict[nextWord] {
					*cur = append(*cur, nextWord)
					wordBreakDFS(res, cur, endIndex+1, dict, s)
					*cur = (*cur)[:len(*cur)-1]
				}
			} else {
				break
			}
		}
	}
}

func detectCycle(head *structure.ListNode) *structure.ListNode {
	if head == nil {
		return nil
	}

	p := head
	q := head
	for true {
		if p.Next != nil {
			p = p.Next
		} else {
			break
		}

		if q.Next != nil && q.Next.Next != nil {
			q = q.Next.Next
		} else {
			break
		}

		if p == q {
			break
		}
	}

	if p != q {
		return nil
	}

	p = head
	for p != q {
		p = p.Next
		q = q.Next
	}

	return p
}

func hasCycle(head *structure.ListNode) bool {
	if head == nil {
		return false
	}

	p := head
	q := head
	for true {
		if p.Next != nil {
			p = p.Next
		} else {
			break
		}

		if q.Next != nil && q.Next.Next != nil {
			q = q.Next.Next
		} else {
			break
		}

		if p == q {
			return true
		}
	}

	return false
}

func candy(ratings []int) int {
	len := len(ratings)
	res := make([]int, len)
	for i := 0; i < len; i++ {
		res[i] = 1
	}
	for i := 1; i < len; i++ {
		if ratings[i] > ratings[i-1] {
			res[i] = res[i-1] + 1
		}
	}

	for i := len - 2; i >= 0; i-- {
		if ratings[i] > ratings[i+1] && res[i] <= res[i+1] {
			res[i] = res[i+1] + 1
		}
	}

	sum := 0
	for i := 0; i < len; i++ {
		sum += res[i]
	}
	return sum
}

func canCompleteCircuit(gas []int, cost []int) int {
	len := len(gas)
	minIndex := 0
	dp := make([]int, len)

	for i := 0; i < len; i++ {
		gas[i] -= cost[i]
		if i == 0 {
			dp[i] = gas[i]
		} else {
			dp[i] = dp[i-1] + gas[i]
		}
		if dp[minIndex] > dp[i] {
			minIndex = i
		}
	}

	cur := 0
	step := 0
	i := minIndex + 1
	for step < len {
		i = i % len
		cur += gas[i]
		if cur < 0 {
			return -1
		}
		step++
		i++
		i = i % len
	}

	return (minIndex + 1) % len
}

func longestConsecutive(nums []int) int {
	leftBoundaryToIntervalMap := make(map[int][]int)
	rightBoundaryToIntervalMap := make(map[int][]int)
	visit := make(map[int]bool)
	for _, v := range nums {
		if _, ok := visit[v]; ok {
			continue
		} else {
			visit[v] = true
		}

		useAsLeftBoundary := v + 1
		lb, okleft := leftBoundaryToIntervalMap[useAsLeftBoundary]
		if okleft {
			leftBoundaryToIntervalMap[v] = []int{v, lb[1]}
		} else {
			leftBoundaryToIntervalMap[v] = []int{v, v}
		}

		useAsRightBoundary := v - 1
		rb, okright := rightBoundaryToIntervalMap[useAsRightBoundary]
		if okright {
			rightBoundaryToIntervalMap[v] = []int{rb[0], v}
		} else {
			rightBoundaryToIntervalMap[v] = []int{v, v}
		}

		// also consider to merge two intervals
		linterval, _ := rightBoundaryToIntervalMap[v]
		rinterval, _ := leftBoundaryToIntervalMap[v]
		newInterval := []int{linterval[0], rinterval[1]}
		if v, _ := leftBoundaryToIntervalMap[newInterval[0]]; v[1] < newInterval[1] {
			leftBoundaryToIntervalMap[newInterval[0]] = newInterval
		}

		if v, _ := rightBoundaryToIntervalMap[newInterval[1]]; v[0] > newInterval[0] {
			rightBoundaryToIntervalMap[newInterval[1]] = newInterval
		}
	}

	best := 0
	for _, v := range leftBoundaryToIntervalMap {
		cur := v[1] - v[0] + 1
		if cur > best {
			best = cur
		}
	}
	return best
}

func minCut(s string) int {
	slen := len(s)
	dp := make([][]bool, slen)
	for i := 0; i < slen; i++ {
		dp[i] = make([]bool, slen)
	}

	for i := slen - 1; i >= 0; i-- {
		for j := i; j < slen; j++ {
			if i == j {
				dp[i][j] = true
			} else if i+1 == j {
				dp[i][j] = (s[i] == s[j])
			} else {
				dp[i][j] = (s[i] == s[j]) && dp[i+1][j-1]
			}
		}
	}
	visit := make(map[int]bool)
	step := 0
	queue := make([]int, 0)
	queue = append(queue, 0)
	for len(queue) > 0 {
		qsize := len(queue)
		for j := 0; j < qsize; j++ {
			cur := queue[0]
			queue = queue[1:]
			if cur == len(s) {
				return step - 1
			}
			for next := cur; next < len(s); next++ {
				if dp[cur][next] && !visit[next+1] {
					visit[next+1] = true
					queue = append(queue, next+1)
				}
			}
		}
		step++
	}

	return step - 1
}

func partition(s string) [][]string {
	res := make([][]string, 0)
	len := len(s)
	dp := make([][]bool, len)
	for i := 0; i < len; i++ {
		dp[i] = make([]bool, len)
	}

	for i := len - 1; i >= 0; i-- {
		for j := i; j < len; j++ {
			if i == j {
				dp[i][j] = true
			} else if i+1 == j {
				dp[i][j] = (s[i] == s[j])
			} else {
				dp[i][j] = (s[i] == s[j]) && dp[i+1][j-1]
			}
		}
	}

	cur := make([]string, 0)
	partitionDFS(&res, &cur, s, dp, 0)
	return res
}

func partitionDFS(res *[][]string, cur *[]string, s string, dp [][]bool, start int) {
	if start == len(s) {
		tmp := make([]string, len(*cur))
		copy(tmp, *cur)
		*res = append(*res, tmp)
	} else {
		for i := start; i < len(s); i++ {
			if dp[start][i] {
				*cur = append(*cur, s[start:i+1])
				partitionDFS(res, cur, s, dp, i+1)
				*cur = (*cur)[:len(*cur)-1]
			}
		}
	}
}
func solve(board [][]byte) {
	h := len(board)
	w := len(board[0])
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			if i == 0 || j == 0 {
				solveSurDFS(board, i, j)
			}
		}
	}

	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			if board[i][j] == 'O' {
				board[i][j] = 'X'
			}
		}
	}

	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			if board[i][j] == 'V' {
				board[i][j] = 'O'
			}
		}
	}
}

func solveSurDFS(board [][]byte, i int, j int) {
	dxy := [][]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
	h := len(board)
	w := len(board[0])
	if i < 0 || i >= h || j < 0 || j >= w || board[i][j] != 'O' {
		return
	} else {
		board[i][j] = 'V'
		for _, v := range dxy {
			newI := i + v[0]
			newJ := j + v[1]
			solveSurDFS(board, newI, newJ)
		}
	}
}

func sumNumbers(root *structure.TreeNode) int {
	if root == nil {
		return 0
	}

	sum := 0
	sumNumbersHelper(root, &sum, 0)
	return sum
}

func sumNumbersHelper(root *structure.TreeNode, sum *int, pathSum int) {
	pathSum = pathSum*10 + root.Val
	if root.Left == nil && root.Right == nil {
		*sum = *sum + pathSum
	} else {
		if root.Left != nil {
			sumNumbersHelper(root.Left, sum, pathSum)
		}

		if root.Right != nil {
			sumNumbersHelper(root.Right, sum, pathSum)
		}
	}
}

func maxProfit4(k int, prices []int) int {

	len := len(prices) + 1
	if k > len/2 {
		return maxProfit2(prices)
	}
	buyState := make([][]int, len)
	sellState := make([][]int, len)
	for i := 0; i < len; i++ {
		buyState[i] = make([]int, k+1)
		sellState[i] = make([]int, k+1)
	}

	for j := 1; j <= k; j++ {
		buyState[0][j] = -10 ^ 6
	}

	buyState[0][0] = -prices[0]
	best := 0
	for i := 1; i < len; i++ {
		for j := 1; j <= k; j++ {
			buyState[i][j] = util.max(buyState[i-1][j], sellState[i-1][j-1]-prices[i-1])
			sellState[i][j] = util.max(sellState[i-1][j], buyState[i-1][j]+prices[i-1])
			best = util.max(best, buyState[i][j])
			best = util.max(best, sellState[i][j])
		}
	}
	return best
}

func maxProfit3(prices []int) int {
	len := len(prices)
	dp := make([]int, len)
	minValue := prices[0]
	max := 0
	for i := 1; i < len; i++ {
		if minValue > prices[i] {
			minValue = prices[i]
		}

		dp[i] = prices[i] - minValue
		if dp[i] < dp[i-1] {
			dp[i] = dp[i-1]
		}
		if max < dp[i] {
			max = dp[i]
		}
	}

	// the max value to the right of this index, including this index
	rightMaxValue := make([]int, len)
	for i := len - 1; i >= 0; i-- {
		if i == len-1 {
			rightMaxValue[i] = prices[i]
		} else {
			rightMaxValue[i] = rightMaxValue[i+1]
			if rightMaxValue[i] < prices[i] {
				rightMaxValue[i] = prices[i]
			}
		}
	}

	for i := 1; i < len; i++ {
		cur := dp[i-1] + (rightMaxValue[i] - prices[i])
		if cur > max {
			max = cur
		}
	}
	return max
}

func maxProfit1(prices []int) int {
	min := prices[0]
	best := 0
	for _, v := range prices {
		if min > v {
			min = v
		}

		diff := v - min
		if diff > best {
			best = diff
		}
	}

	return best
}

func maxProfit2(prices []int) int {
	best := 0
	for i := 1; i < len(prices); i++ {
		best += util.max(0, prices[i]-prices[i-1])
	}
	return best
}

// pathSum can be any path rather than root to leaf
func maxPathSum(root *structure.TreeNode) int {
	leftEndSum := make(map[*structure.TreeNode]int)
	rightEndSum := make(map[*structure.TreeNode]int)
	maxEndSum(root, leftEndSum, rightEndSum)
	best := root.Val
	for k := range leftEndSum {
		leftVal := 0
		if k.Left != nil && util.max(leftEndSum[k.Left], rightEndSum[k.Left]) > leftVal {
			leftVal = util.max(leftEndSum[k.Left], rightEndSum[k.Left])
		}
		rightVal := 0
		if k.Right != nil && util.max(rightEndSum[k.Right], leftEndSum[k.Right]) > rightVal {
			rightVal = util.max(rightEndSum[k.Right], leftEndSum[k.Right])
		}
		if rightVal+leftVal+k.Val > best {
			best = rightVal + leftVal + k.Val
		}

	}
	return best
}

func maxEndSum(root *structure.TreeNode, leftEndSum map[*structure.TreeNode]int, rightEndSum map[*structure.TreeNode]int) int {
	if root == nil {
		return 0
	}

	leftEndSum[root] = root.Val
	rightEndSum[root] = root.Val

	if root.Left != nil {
		if leftSum := maxEndSum(root.Left, leftEndSum, rightEndSum); leftSum > 0 {
			leftEndSum[root] = leftSum + root.Val
		}
	}

	if root.Right != nil {
		if rightSum := maxEndSum(root.Right, leftEndSum, rightEndSum); rightSum > 0 {
			rightEndSum[root] = rightSum + root.Val
		}
	}

	fmt.Printf("%d-%d-%d,", root.Val, leftEndSum[root], rightEndSum[root])
	return util.max(leftEndSum[root], rightEndSum[root])
}

func ladderLength(beginWord string, endWord string, wordList []string) int {
	dict := make(map[string]bool)
	for _, v := range wordList {
		dict[v] = true
	}

	if beginWord == endWord {
		return 1
	}

	if !dict[endWord] {
		return 0
	}

	queue := make([]string, 0)
	queue = append(queue, beginWord)
	size := 1
	used := make(map[string]bool)
	used[beginWord] = true
	for len(queue) > 0 {
		qsize := len(queue)
		for i := 0; i < qsize; i++ {
			cur := queue[0]
			queue = queue[1:]
			neighbor := findNb(cur, dict)
			for _, v := range neighbor {
				if used[v] {
					continue
				}

				if v == endWord {
					return size + 1
				}
				used[v] = true
				queue = append(queue, v)
			}
		}

		size++
	}

	return 0
}

func findLadders(beginWord string, endWord string, wordList []string) [][]string {
	dict := make(map[string]bool)
	for _, v := range wordList {
		dict[v] = true
	}

	if beginWord == endWord {
		return [][]string{[]string{beginWord}}
	}

	if !dict[endWord] {
		return [][]string{}
	}

	prev := make(map[string][]string)
	prevLen := make(map[string]int) // len from current node to beginWord
	for _, v := range wordList {
		prevLen[v] = len(wordList) + 10
	}
	prevLen[beginWord] = 0
	queue := make([]string, 0)
	queue = append(queue, beginWord)
	size := 1
	for len(queue) > 0 {
		if prev[endWord] != nil {
			break
		}
		qsize := len(queue)
		for i := 0; i < qsize; i++ {
			cur := queue[0]
			queue = queue[1:]
			neighbor := findNb(cur, dict)
			for _, v := range neighbor {
				if prevLen[v] == size {
					prev[v] = append(prev[v], cur)
				} else if size < prevLen[v] {
					prevLen[v] = size
					prev[v] = []string{cur}
					queue = append(queue, v)
				}
			}
		}

		size++
	}

	if prev[endWord] == nil {
		return make([][]string, 0)
	}

	return findPath(beginWord, endWord, prev)
}

func findPath(bw string, ew string, prev map[string][]string) [][]string {
	res := make([][]string, 0)
	cur := []string{ew}
	findPathdfs(bw, ew, prev, &res, &cur)
	return res
}

func findPathdfs(bw string, ew string, prev map[string][]string, res *[][]string, cur *[]string) {
	if bw == ew {
		tmp := make([]string, len(*cur))
		copy(tmp, *cur)
		*res = append(*res, tmp)
	} else {
		prevList := prev[ew]
		for _, v := range prevList {
			*cur = append([]string{v}, (*cur)...)
			findPathdfs(bw, v, prev, res, cur)
			*cur = (*cur)[1:]
		}
	}
}

func findNb(cur string, dict map[string]bool) []string {
	res := make([]string, 0)
	for k, _ := range dict {
		if wordDiff(cur, k) {
			res = append(res, k)
		}
	}
	return res
}

func wordDiff(cur string, next string) bool {
	diff := false
	for i := 0; i < len(cur); i++ {
		if cur[i] != next[i] {
			if diff {
				return false
			} else {
				diff = true
			}
		}
	}

	return diff == true
}

func numDistinct(s string, t string) int {
	slen := len(s)
	tlen := len(t)
	dp := make([][]int, slen+1)
	for i := 0; i <= slen; i++ {
		dp[i] = make([]int, tlen+1)
	}

	for i := 0; i <= slen; i++ {
		for j := 0; j <= tlen; j++ {
			if j == 0 {
				dp[i][j] = 1
			} else if i == 0 {
				dp[i][j] = 0
			} else {
				dp[i][j] = dp[i-1][j]
				if s[i-1] == t[j-1] {
					dp[i][j] += dp[i-1][j-1]
				}
			}
		}
	}

	return dp[slen][tlen]
}

func minimumTotal(triangle [][]int) int {
	level := len(triangle)
	dp := make([][]int, level)
	for i := level - 1; i >= 0; i-- {
		dp[i] = make([]int, i+1)
		for j := 0; j <= i; j++ {
			if i == level-1 {
				dp[i][j] = triangle[i][j]
			} else {
				dp[i][j] = triangle[i][j] + dp[i+1][j]
				if j+1 <= i+1 {
					dp[i][j] = util.min(dp[i][j], triangle[i][j]+dp[i+1][j+1])
				}
			}
		}
	}
	return dp[0][0]
}

func generate(numRows int) [][]int {
	res := make([][]int, 0)
	if numRows >= 1 {
		res = append(res, []int{1})
	}
	if numRows >= 2 {
		res = append(res, []int{1, 1})
	}

	for i := 3; i <= numRows; i++ {
		lastRow := res[i-2]
		curRow := make([]int, len(lastRow)+1)
		for j := 0; j <= len(lastRow); j++ {
			if j == 0 || j == len(lastRow) {
				curRow[j] = 1
			} else {
				curRow[j] = lastRow[j-1] + lastRow[j]
			}
		}

		res = append(res, curRow)
	}

	return res
}

type Node struct {
	Val       int
	Neighbors []*Node
}

var cloneGraphMap map[*Node]*Node

func cloneGraph(node *Node) *Node {
	if cloneGraphMap == nil {
		cloneGraphMap = make(map[*Node]*Node)
	}
	if v, ok := cloneGraphMap[node]; ok {
		return v
	} else {
		cloneNode := &Node{Val: node.Val}
		cloneGraphMap[node] = cloneNode

		for _, v := range node.Neighbors {
			cloneNode.Neighbors = append(cloneNode.Neighbors, cloneGraph(v))
		}

		return cloneNode
	}
}

//func connect(root *Node) *Node {
//	queue := make([]*Node, 0)
//	if root == nil {
//		return root
//	}
//
//	queue = append(queue, root)
//	for len(queue) != 0 {
//		qsize := len(queue)
//		last := &Node{}
//		for i := 0; i < qsize; i++ {
//			cur := queue[0]
//			queue = queue[1:]
//			if cur.Left != nil {
//				queue = append(queue, cur.Left)
//			}
//			if cur.Right != nil {
//				queue = append(queue, cur.Right)
//			}
//
//			last.Next = cur
//			last = cur
//		}
//	}
//
//	return root
//}

func flatten(root *structure.TreeNode) {
	flattenHelper(root)
}

func flattenHelper(root *structure.TreeNode) (*structure.TreeNode, *structure.TreeNode) {
	if root == nil {
		return nil, nil
	} else {
		leftHead, leftTail := flattenHelper(root.Left)
		rightHead, rightTail := flattenHelper(root.Right)
		root.Left = nil
		root.Right = nil

		if leftHead != nil {
			root.Right = leftHead
		} else {
			root.Right = rightHead
		}

		if leftTail != nil {
			leftTail.Right = rightHead
		}

		r2 := root
		if rightTail != nil {
			r2 = rightTail
		} else if leftTail != nil {
			r2 = leftTail
		}

		return root, r2
	}
}

func pathSum(root *structure.TreeNode, targetSum int) [][]int {
	res := make([][]int, 0)
	cur := make([]int, 0)
	pathSumDfs(root, targetSum, &res, &cur)
	return res
}

func pathSumDfs(root *structure.TreeNode, targetSum int, res *[][]int, cur *[]int) {
	if root == nil {
		return
	}

	*cur = append(*cur, root.Val)
	if root.Left == nil && root.Right == nil {
		if targetSum == root.Val {
			tmp := make([]int, len(*cur))
			copy(tmp, *cur)
			*res = append(*res, tmp)
		}
	} else {
		pathSumDfs(root.Left, targetSum-root.Val, res, cur)
		pathSumDfs(root.Right, targetSum-root.Val, res, cur)
	}

	*cur = (*cur)[:len(*cur)-1]
}

func hasPathSum(root *structure.TreeNode, targetSum int) bool {
	if root == nil {
		return false
	} else if root.Left == nil && root.Right == nil {
		return targetSum == root.Val
	} else {
		return hasPathSum(root.Left, targetSum-root.Val) || hasPathSum(root.Right, targetSum-root.Val)
	}
}

func minDepth(root *structure.TreeNode) int {
	if root == nil {
		return 0
	} else if root.Left == nil && root.Right == nil {
		return 1
	} else {
		leftDepth := minDepth(root.Left)
		rightDepth := minDepth(root.Right)

		if leftDepth == 0 {
			return rightDepth + 1
		} else if rightDepth == 0 {
			return leftDepth + 1
		} else {
			if leftDepth < rightDepth {
				return leftDepth + 1
			} else {
				return rightDepth + 1
			}
		}
	}
}

func isBalanced(root *structure.TreeNode) bool {
	ok, _ := isBalancedHelper(root)
	return ok
}

func isBalancedHelper(root *structure.TreeNode) (bool, int) {
	if root == nil {
		return true, 0
	} else {
		leftok, leftdepth := isBalancedHelper(root.Left)
		rightok, rightdepth := isBalancedHelper(root.Right)
		if !leftok || !rightok {
			return false, 0
		} else {
			if leftdepth > rightdepth {
				return leftdepth <= rightdepth+1, 1 + leftdepth
			} else {
				return rightdepth <= leftdepth+1, 1 + rightdepth
			}
		}
	}
}

func sortedListToBST(head *structure.ListNode) *structure.TreeNode {
	len := findListLengthHelper(head)
	res, _ := sortedListToBSTHelper(head, len)
	return res
}

func sortedListToBSTHelper(head *structure.ListNode, len int) (*structure.TreeNode, *structure.ListNode) {
	if len == 0 {
		return nil, nil
	} else if len == 1 {
		return &structure.TreeNode{Val: head.Val}, head
	} else {
		left, tail := sortedListToBSTHelper(head, len/2)
		root := &structure.TreeNode{Val: tail.Next.Val}
		root.Left = left
		var trytailnext *structure.ListNode
		if tail.Next != nil {
			trytailnext = tail.Next.Next
		}
		right, rtail := sortedListToBSTHelper(trytailnext, len-len/2-1)
		root.Right = right
		if rtail != nil {
			return root, rtail
		} else {
			return root, tail.Next
		}
	}
}

func findListLengthHelper(head *structure.ListNode) int {
	if head == nil {
		return 0
	}
	return 1 + findListLengthHelper(head.Next)
}

func sortedArrayToBST(nums []int) *structure.TreeNode {
	return sortedArrayToBSTHelper(nums, 0, len(nums)-1)
}

func sortedArrayToBSTHelper(nums []int, start int, end int) *structure.TreeNode {
	if start > end {
		return nil
	} else {
		mid := start + (end-start)/2
		root := &structure.TreeNode{Val: nums[mid]}
		root.Left = sortedArrayToBSTHelper(nums, start, mid-1)
		root.Right = sortedArrayToBSTHelper(nums, mid+1, end)
		return root
	}
}

func buildTree(inorder []int, postorder []int) *structure.TreeNode {
	inorderValueToIndex := make(map[int]int)
	for k, v := range inorder {
		inorderValueToIndex[v] = k
	}

	return buildTreeWithPostInOrder(postorder, len(postorder)-1, len(postorder), inorder, 0, len(inorder), inorderValueToIndex)
}

func buildTreeWithPostInOrder(postorder []int, pend int, plen int, inorder []int, istart int, ilen int, inorderValueToIndex map[int]int) *structure.TreeNode {
	if plen <= 0 {
		return nil
	} else {
		rootValue := postorder[pend]
		root := &structure.TreeNode{Val: rootValue}
		rootIndexInOrder := inorderValueToIndex[rootValue]
		leftLen := rootIndexInOrder - istart
		rightLen := ilen - 1 - leftLen
		root.Right = buildTreeWithPostInOrder(postorder, pend-1, rightLen, inorder, rootIndexInOrder+1, rightLen, inorderValueToIndex)
		root.Left = buildTreeWithPostInOrder(postorder, pend-1-rightLen, leftLen, inorder, istart, leftLen, inorderValueToIndex)
		return root
	}
}

func buildTreePreInd(preorder []int, inorder []int) *structure.TreeNode {
	inorderValueToIndex := make(map[int]int)
	for k, v := range inorder {
		inorderValueToIndex[v] = k
	}

	return buildTreeWithPreInOrder(preorder, 0, len(preorder), inorder, 0, len(inorder), inorderValueToIndex)
}

func buildTreeWithPreInOrder(preorder []int, pstart int, plen int, inorder []int, istart int, ilen int, inorderValueToIndex map[int]int) *structure.TreeNode {
	if plen <= 0 {
		return nil
	} else if plen == 1 {
		return &structure.TreeNode{Val: preorder[pstart]}
	} else {
		rootValue := preorder[pstart]
		rootIndexInOrder := inorderValueToIndex[rootValue]
		leftLen := rootIndexInOrder - istart
		rightLen := ilen - 1 - leftLen
		root := &structure.TreeNode{Val: rootValue}
		root.Left = buildTreeWithPreInOrder(preorder, pstart+1, leftLen, inorder, istart, leftLen, inorderValueToIndex)
		root.Right = buildTreeWithPreInOrder(preorder, pstart+1+leftLen, rightLen, inorder, rootIndexInOrder+1, rightLen, inorderValueToIndex)
		return root
	}
}

func levelOrderBottom(root *structure.TreeNode) [][]int {
	res := make([][]int, 0)
	queue := make([]*structure.TreeNode, 0)
	if root == nil {
		return res
	} else {
		queue = append(queue, root)
	}

	for len(queue) > 0 {
		len := len(queue)
		cur := make([]int, 0)
		for i := 0; i < len; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}

			cur = append(cur, node.Val)
		}

		res = append([][]int{cur}, res...)
	}

	return res
}

func maxDepth(root *structure.TreeNode) int {
	if root == nil {
		return 0
	} else if root.Left == nil && root.Right == nil {
		return 1
	} else {
		maxD := 1
		if root.Left != nil {
			maxD = 1 + maxDepth(root.Left)
		}

		if root.Right != nil {
			d := 1 + maxDepth(root.Right)
			if d > maxD {
				maxD = d
			}
		}

		return maxD
	}
}

func zigzagLevelOrder(root *structure.TreeNode) [][]int {
	res := make([][]int, 0)
	queue := make([]*structure.TreeNode, 0)
	if root == nil {
		return res
	} else {
		queue = append(queue, root)
	}

	flip := false
	for len(queue) > 0 {
		len := len(queue)
		cur := make([]int, 0)
		for i := 0; i < len; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}

			if !flip {
				cur = append(cur, node.Val)
			} else {
				cur = append([]int{node.Val}, cur...)
			}
		}

		flip = !flip
		res = append(res, cur)
	}

	return res
}

func removeDuplicates(nums []int) int {
	next := 0
	i := 0
	for i < len(nums) {
		cur := nums[i]
		j := i + 1
		for j < len(nums) && nums[j] == cur {
			j++
		}
		len := j - i
		if len > 2 {
			len = 2
		}

		for k := 0; k < len; k++ {
			nums[next] = nums[i]
			next++
			i++
		}

		i = j
	}
	return next
}

var climbCache map[int]int

func climbStairs(n int) int {
	if climbCache == nil {
		climbCache = make(map[int]int)
	}
	if v, ok := climbCache[n]; ok {
		return v
	}
	if n <= 2 {
		return n
	} else {
		res := climbStairs(n-1) + climbStairs(n-2)
		climbCache[n] = res
		return res
	}
}

func restoreIpAddresses(s string) []string {
	res := make([]string, 0)
	if len(s) > 16 {
		return res
	}

	restoreIpAddressesdfs(s, 0, &res, "")
	return res
}

func restoreIpAddressesdfs(s string, parts int, res *[]string, cur string) {
	if parts > 4 {
		return
	}
	if len(s) == 0 {
		if parts == 4 {
			*res = append(*res, cur)
		}

		return
	} else {
		for i := 0; i < 3 && i < len(s); i++ {
			firstBitString := s[0 : i+1]
			if firstBitString[0] == '0' && len(firstBitString) > 1 {
				break
			}

			v, err := strconv.Atoi(firstBitString)
			if err != nil || v < 0 || v > 255 {
				continue
			}
			nextCur := cur
			nextCur += firstBitString
			if parts != 3 {
				nextCur += "."
			}
			restoreIpAddressesdfs(s[i+1:], parts+1, res, nextCur)

		}
	}
}

func subsetsWithDup(nums []int) [][]int {
	res := make([][]int, 0)
	cur := make([]int, 0)
	visit := make([]bool, len(nums))
	sort.Ints(nums)
	subsetsWithDupDfs(nums, &res, &cur, 0, &visit)
	return res
}

func subsetsWithDupDfs(nums []int, res *[][]int, cur *[]int, start int, visit *[]bool) {
	if start == len(nums) {
		tmp := make([]int, len(*cur))
		copy(tmp, *cur)
		*res = append(*res, tmp)
	} else {
		subsetsWithDupDfs(nums, res, cur, start+1, visit)
		if start == 0 || nums[start] != nums[start-1] || (*visit)[start-1] {
			(*visit)[start] = true
			(*cur) = append(*cur, nums[start])
			subsetsWithDupDfs(nums, res, cur, start+1, visit)
			*cur = (*cur)[:len(*cur)-1]
			(*visit)[start] = false
		}
	}
}

func maximalRectangle(matrix [][]byte) int {
	h := len(matrix)
	w := len(matrix[0])
	height := make([][]int, h)
	for i := 0; i < h; i++ {
		height[i] = make([]int, w)
	}

	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			if matrix[i][j] != '0' {
				if i == 0 {
					height[i][j] = 1
				} else {
					height[i][j] = 1 + height[i-1][j]
				}
			}
		}
	}

	best := 0
	for i := 0; i < h; i++ {
		res := histgram(height, i)
		if res > best {
			best = res
		}

	}

	return best
}

func histgram(height [][]int, base int) int {
	stack := structure.Stack{}

	h := height[base]
	h = append([]int{0}, h...)
	h = append(h, 0)
	best := 0
	for i := 0; i < len(h); i++ {
		if stack.IsEmpty() || h[stack.Peek()] <= h[i] {
			stack.Push(i)
		} else {
			//stack.pop > current
			l := stack.Pop()
			w := i - stack.Peek() - 1
			cur := h[l] * w
			if cur > best {
				best = cur
			}
			i--
		}
	}
	return best
}

func recoverTree(root *structure.TreeNode) {
	stack := structure.TreeStack{}
	var prev *structure.TreeNode
	var next *structure.TreeNode
	var last *structure.TreeNode
	for root != nil || !stack.IsEmpty() {
		for root != nil {
			stack.Push(root)
			root = root.Left
		}

		root = stack.Pop()
		if last != nil {
			if last.Val > root.Val {
				if prev == nil {
					prev = last
					next = root
				} else {
					next = root
				}
			}
		}

		last = root
		root = root.Right
	}

	t := next.Val
	next.Val = prev.Val
	prev.Val = t
}

func inorderTraversal(root *structure.TreeNode) []int {
	res := make([]int, 0)
	dfs_inorderTraversal(root, &res)
	return res

}

func dfs_inorderTraversal(root *structure.TreeNode, res *[]int) {
	if root == nil {
		return
	}
	dfs_inorderTraversal(root.Left, res)
	*res = append(*res, root.Val)
	dfs_inorderTraversal(root.Right, res)
}

func preorderTraversal(root *structure.TreeNode) []int {
	//res := make([]int, 0)
	//dfs_preorderTraversal(root, &res)
	//return res
	stack := structure.TreeStack{}
	res := make([]int, 0)
	if root == nil {
		return res
	}

	stack.Push(root)
	for !stack.IsEmpty() {
		root = stack.Pop()
		res = append(res, root.Val)
		if root.Right != nil {
			stack.Push(root.Right)
		}

		if root.Left != nil {
			stack.Push(root.Left)
		}
	}

	return res
}

func dfs_preorderTraversal(root *structure.TreeNode, i *[]int) {
	if root == nil {
		return
	}
	*i = append(*i, root.Val)
	dfs_preorderTraversal(root.Left, i)
	dfs_preorderTraversal(root.Right, i)
}

func postorderTraversal(root *structure.TreeNode) []int {
	res := make([]int, 0)
	dfs_postorderTraversal(root, &res)
	return res
}

func dfs_postorderTraversal(root *structure.TreeNode, i *[]int) {
	if root == nil {
		return
	}
	dfs_postorderTraversal(root.Left, i)
	dfs_postorderTraversal(root.Right, i)
	*i = append(*i, root.Val)
}

func reverseList(head *structure.ListNode) *structure.ListNode {
	//res := &ListNode{}
	//p := head
	//for p != nil {
	//	t := p.Next
	//	p.Next = res.Next
	//	res.Next = p
	//	p = t
	//}
	//return (*res).Next
	if head == nil {
		return head
	}
	p := head.Next
	head.Next = nil
	res := reverseList(p)
	p.Next = head
	return res
}

func partition2(head *structure.ListNode, x int) *structure.ListNode {
	sHead := structure.ListNode{}
	lHead := structure.ListNode{}
	s := &sHead
	p := &lHead
	for head != nil {
		v := head.Val
		if v < x {
			s.Next = head
			s = s.Next
		} else {
			p.Next = head
			p = p.Next
		}
		head = head.Next
	}

	p.Next = nil
	s.Next = lHead.Next
	return sHead.Next
}

func deleteDuplicates(head *structure.ListNode) *structure.ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	if head.Next.Val != head.Val {
		head.Next = deleteDuplicates(head.Next)
		return head
	} else {
		v := head.Val
		for head != nil && head.Val == v {
			head = head.Next
		}

		return deleteDuplicates(head)
	}
}

func deleteDuplicates2(head *structure.ListNode) *structure.ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	if head.Val == head.Next.Val {
		head.Next = head.Next.Next
		return deleteDuplicates(head)
	} else {
		head.Next = deleteDuplicates(head.Next)
		return head
	}
}

func largestRectangleArea(heights []int) int {
	stack := structure.Stack{}
	i := 0
	best := 0
	heights = append(heights, 0)
	heights = append([]int{0}, heights...)
	for i < len(heights) {
		if stack.IsEmpty() || heights[stack.Peek()] <= heights[i] {
			stack.Push(i)
			i++
		} else {
			h := heights[stack.Pop()]
			w := i - 1 - stack.Peek()
			area := h * w
			if area > best {
				best = area
			}
		}
	}

	return best
}

func exist(board [][]byte, word string) bool {
	h := len(board)
	w := len(board[0])

	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			used := make([][]bool, h)
			for i := 0; i < h; i++ {
				used[i] = make([]bool, w)
			}

			if existhelper(board, word, &used, i, j) {
				return true
			}
		}
	}

	return false
}

func existhelper(board [][]byte, word string, used *[][]bool, i int, j int) bool {
	dxy := [][]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	if len(word) == 0 {
		return true
	}
	if i < 0 || i >= len(board) || j < 0 || j >= len(board[0]) {
		return false
	}

	if (*used)[i][j] {
		return false
	}

	if word[0] != board[i][j] {
		return false
	}

	(*used)[i][j] = true
	for _, v := range dxy {
		newi := v[0] + i
		newj := v[1] + j
		if existhelper(board, word[1:], used, newi, newj) {
			return true
		}
	}
	(*used)[i][j] = false
	return false
}

func subsets(nums []int) [][]int {
	res := make([][]int, 0)
	subsetsdfs(nums, &res, make([]int, 0), 0)
	return res
}

func subsetsdfs(nums []int, res *[][]int, cur []int, start int) {
	if start == len(nums) {
		tmp := make([]int, len(cur))
		copy(tmp, cur)
		*res = append(*res, tmp)
	} else {
		subsetsdfs(nums, res, cur, start+1)

		cur = append(cur, nums[start])
		subsetsdfs(nums, res, cur, start+1)
		cur = cur[:len(cur)-1]

	}
}

func combine(n int, k int) [][]int {
	used := make([]bool, n)
	res := make([][]int, 0)
	combinedfs(&used, make([]int, 0), &res, k, 0)
	return res
}

func combinedfs(used *[]bool, cur []int, res *[][]int, k int, start int) {
	if k == 0 {
		tmp := make([]int, len(cur))
		copy(tmp, cur)
		*res = append(*res, tmp)
	} else {
		for i := start; i < len(*used); i++ {
			if !(*used)[i] {
				(*used)[i] = true
				cur = append(cur, i+1)
				combinedfs(used, cur, res, k-1, i+1)
				cur = cur[:len(cur)-1]
				(*used)[i] = false
			}
		}
	}
}

func minWindow(s string, t string) string {
	dict := make(map[rune]int)
	for _, v := range t {
		if _, ok := dict[v]; !ok {
			dict[v] = 1
		} else {
			dict[v]++
		}
	}

	has := make(map[rune]int)
	i := 0
	j := 0
	resLen := 0
	besti := 0
	bestj := len(s)
	sarray := []rune(s)
	for j < len(sarray) {
		charC := sarray[j]
		if v, ok := dict[charC]; ok {
			has[charC]++
			if has[charC] <= v {
				resLen++
			}
		}

		for resLen == len(t) {
			// result is always valid here
			if bestj-besti > j-i {
				bestj = j
				besti = i
			}

			lastC := sarray[i]
			if _, ok := dict[lastC]; ok {
				has[lastC]--
				if has[lastC] < dict[lastC] {
					resLen--
				}
			}
			i++
		}
		j++
	}

	if bestj == len(s) {
		return ""
	}
	return s[besti : bestj+1]
}

func swap(nums []int, i int, j int) {
	t := nums[i]
	nums[i] = nums[j]
	nums[j] = t
}

func sortColors(nums []int) {
	p := 0
	q := len(nums) - 1
	for i := 0; i <= q; i++ {
		if nums[i] == 0 {
			swap(nums, i, p)
			p++
		} else if nums[i] == 2 {
			swap(nums, q, i)
			q--
			i--
		} else {
			nums[i] = 1
		}
	}
}

func searchMatrix(matrix [][]int, target int) bool {
	h := len(matrix)
	w := len(matrix[0])
	lo := 0
	hi := h*w - 1
	for lo < hi {
		mid := lo + (hi-lo)/2
		x := mid / w
		y := mid % w
		v := matrix[x][y]
		if v == target {
			return true
		} else if v < target {
			lo = mid + 1
		} else {
			hi = mid
		}
	}

	return matrix[lo/w][lo%w] == target
}

func setZeroes(matrix [][]int) {
	isTopZero := false
	isLeftZero := false
	h := len(matrix)
	w := len(matrix[0])
	for j := 0; j < w; j++ {
		if matrix[0][j] == 0 {
			isTopZero = true
			break
		}
	}

	for i := 0; i < h; i++ {
		if matrix[i][0] == 0 {
			isLeftZero = true
			break
		}
	}

	for i := 1; i < h; i++ {
		for j := 1; j < w; j++ {
			if matrix[i][j] == 0 {
				matrix[i][0] = 0
				matrix[0][j] = 0
			}
		}
	}

	for i := 1; i < h; i++ {
		for j := 1; j < w; j++ {
			if matrix[i][0] == 0 || matrix[0][j] == 0 {
				matrix[i][j] = 0
			}
		}
	}

	if isTopZero {
		for j := 0; j < w; j++ {
			matrix[0][j] = 0
		}
	}

	if isLeftZero {
		for i := 0; i < h; i++ {
			matrix[i][0] = 0
		}
	}
}

func minDistance(word1 string, word2 string) int {
	len1 := len(word1) + 1
	len2 := len(word2) + 1
	dp := make([][]int, len1)
	for i := 0; i < len1; i++ {
		dp[i] = make([]int, len2)
	}

	for i := 0; i < len1; i++ {
		for j := 0; j < len2; j++ {
			if i == 0 && j == 0 {
				dp[i][j] = 0
			} else if i == 0 {
				dp[i][j] = j
			} else if j == 0 {
				dp[i][j] = i
			} else {
				dp[i][j] = 1 + util.min(util.min(dp[i-1][j], dp[i][j-1]), dp[i-1][j-1])
				if word1[i-1] == word2[j-1] {
					dp[i][j] = util.min(dp[i][j], dp[i-1][j-1])
				}

			}
		}
	}

	return dp[len1-1][len2-1]
}

func simplifyPath(path string) string {
	p := strings.Split(path, "/")
	stack := make([]string, 0)
	for _, v := range p {
		if len(v) == 0 || v == "." {
			continue
		} else if v == ".." {
			if len(stack) >= 1 {
				stack = stack[:len(stack)-1]
			}
		} else {
			stack = append(stack, v)
		}
	}

	res := ""
	for _, v := range stack {
		res += "/"
		res += v
	}

	if len(res) == 0 {
		return "/"
	}

	return res
}

func isSameTree(p *structure.TreeNode, q *structure.TreeNode) bool {
	if p == nil || q == nil {
		return p == q
	} else if p.Val != q.Val {
		return false
	} else {
		return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
	}
}

func minPathSum(grid [][]int) int {
	h := len(grid)
	w := len(grid[0])
	dp := make([][]int, h)
	for i := range dp {
		dp[i] = make([]int, w)
	}

	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			if i == 0 && j == 0 {
				dp[i][j] = grid[i][j]
			} else if i == 0 {
				dp[i][j] = grid[i][j] + dp[i][j-1]
			} else if j == 0 {
				dp[i][j] = grid[i][j] + dp[i-1][j]
			} else {
				dp[i][j] = grid[i][j]
				if grid[i-1][j] > grid[i][j-1] {
					dp[i][j] += grid[i][j-1]
				} else {
					dp[i][j] += grid[i-1][j]
				}
			}
		}
	}

	return dp[h-1][w-1]
}

func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	h := len(obstacleGrid)
	w := len(obstacleGrid[0])
	dp := make([][]int, h)
	for i := range dp {
		dp[i] = make([]int, w)
	}

	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			if obstacleGrid[i][j] == 1 {
				dp[i][j] = 0
			} else if i == 0 && j == 0 {
				dp[i][j] = 1
			} else if i == 0 {
				dp[i][j] += dp[i][j-1]
			} else if j == 0 {
				dp[i][j] += dp[i-1][j]
			} else {
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
			}
		}
	}

	return dp[h-1][w-1]
}

func rotateRight(head *structure.ListNode, k int) *structure.ListNode {
	if k == 0 || head == nil {
		return head
	}

	len, tail := findListLength(head)
	k = k % len
	if k == 0 {
		return head
	}

	p := head
	for i := 0; i < len-k-1; i++ {
		p = p.Next
	}
	res := p.Next
	p.Next = nil
	tail.Next = head
	return res
}

func findListLength(head *structure.ListNode) (int, *structure.ListNode) {
	count := 0
	for head.Next != nil {
		count++
		head = head.Next
	}

	return count + 1, head
}

func lengthOfLastWord(s string) int {
	s = strings.Trim(s, " ")
	last := strings.LastIndex(s, " ")
	return len(s) - last - 1
}

func insert(intervals [][]int, newInterval []int) [][]int {
	if len(intervals) == 0 {
		return append(intervals, newInterval)
	}

	res := make([][]int, 0)
	hasInsert := false
	for _, v := range intervals {
		if hasInsert {
			res = append(res, v)
		} else if v[1] < newInterval[0] {
			res = append(res, v)
		} else if v[0] > newInterval[1] {
			res = append(res, newInterval)
			res = append(res, v)
			hasInsert = true
		} else {
			if v[0] < newInterval[0] {
				newInterval[0] = v[0]
			}

			if v[1] > newInterval[1] {
				newInterval[1] = v[1]
			}
		}
	}

	if !hasInsert {
		res = append(res, newInterval)
	}

	return res
}

func canJump(nums []int) bool {
	possibleBest := 0
	for i := 0; i < len(nums); i++ {
		possibleBest = util.max(possibleBest, i+nums[i])
		if possibleBest <= i {
			return false
		} else if possibleBest >= len(nums)-1 {
			return true
		}
	}

	return false
}

func merge(intervals [][]int) [][]int {
	if len(intervals) <= 1 {
		return intervals
	}

	res := make([][]int, 0)
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i][0] < intervals[j][0] {
			return true
		} else {
			return false
		}
	})

	start := intervals[0][0]
	end := intervals[0][1]
	for _, v := range intervals {
		newStart := v[0]
		newEnd := v[1]
		if newStart > end {
			res = append(res, []int{start, end})
			start = newStart
			end = newEnd
		} else {
			if newEnd > end {
				end = newEnd
			}
		}
	}

	res = append(res, []int{start, end})
	return res
}

func maxSubArray(nums []int) int {
	best := nums[0]
	dp := make([]int, len(nums)) // ending with this element, what is the max value
	// no need this dp i, just two element will be enough
	for i := 0; i < len(nums); i++ {
		if i == 0 {
			dp[i] = nums[i]
		} else {
			dp[i] = nums[i]
			if dp[i-1]+nums[i] > dp[i] {
				dp[i] = dp[i-1] + nums[i]
			}
		}

		if best < dp[i] {
			best = dp[i]
		}
	}

	return best
}

func myPow(x float64, n int) float64 {
	if n == 0 {
		return 1
	} else if n < 0 {
		return myPow(1/x, -n)
	} else if n%2 != 0 {
		return myPow(x, n-1) * x
	} else {
		return myPow(x*x, n/2)
	}
}

func groupAnagrams(strs []string) [][]string {
	var res [][]string
	kmap := make(map[string][]string)
	for _, v := range strs {
		code := countEcode(v)
		if _, ok := kmap[code]; !ok {
			kmap[code] = make([]string, 0)
		}

		kmap[code] = append(kmap[code], v)
	}

	for _, v := range kmap {
		res = append(res, v)
	}

	return res
}

func countEcode(s string) string {
	res := make([]int, 26)
	for i := 0; i < len(s); i++ {
		c := s[i]
		res[c-'a']++
	}

	r := ""
	for i := 0; i < 26; i++ {
		r += fmt.Sprintf("%s%d", string(rune('a'+i)), res[i])
	}
	return r
}

func rotate(matrix [][]int) {
	n := len(matrix)
	for i := 0; i < n/2; i++ {
		for j := 0; j < n; j++ {
			t := matrix[i][j]
			matrix[i][j] = matrix[n-i-1][j]
			matrix[n-i-1][j] = t
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			t := matrix[i][j]
			matrix[i][j] = matrix[j][i]
			matrix[j][i] = t
		}
	}
}

func permuteUnique(nums []int) [][]int {
	sort.Ints(nums)
	var res [][]int
	len := len(nums)
	used := make([]bool, len)
	permuteUniquedfs(&res, nums, &used, []int{})
	return res
}

func permuteUniquedfs(res *[][]int, nums []int, used *[]bool, cur []int) {
	for i := 0; i < len(nums); i++ {
		if i != 0 && nums[i] == nums[i-1] && !(*used)[i-1] {
			continue
		}

		if !(*used)[i] {
			(*used)[i] = true
			next := append(cur, nums[i])
			if len(next) == len(nums) {
				temp := make([]int, len(nums))
				copy(temp, next)
				*res = append((*res), temp)
			} else {
				permuteUniquedfs(res, nums, used, next)
			}

			(*used)[i] = false
		}
	}
}

func permute(nums []int) [][]int {
	var res [][]int
	len := len(nums)
	used := make([]bool, len)
	permutedfs(&res, nums, &used, []int{})
	return res
}

func permutedfs(res *[][]int, nums []int, used *[]bool, cur []int) {
	for i := 0; i < len(nums); i++ {
		if !(*used)[i] {
			(*used)[i] = true
			next := append(cur, nums[i])
			if len(next) == len(nums) {
				*res = append((*res), next)
			} else {
				permutedfs(res, nums, used, next)
			}

			(*used)[i] = false
		}
	}
}

func jump(nums []int) int {
	count := 0
	len := len(nums)
	if len <= 1 {
		return count
	}
	curBest := 0
	possibleBest := 0
	for i := 0; i < len; i++ {
		possibleBest = util.max(possibleBest, i+nums[i])
		// current stuck, has to do a jump
		if curBest <= i {
			count++
			curBest = possibleBest
			if curBest >= len-1 {
				return count
			}
		}
	}

	return count
}

func isMatch(s string, p string) bool {
	slen := len(s)
	plen := len(p)
	dp := make([][]bool, slen+1)
	for i := 0; i <= slen; i++ {
		dp[i] = make([]bool, plen+1)
	}

	for i := 0; i <= slen; i++ {
		for j := 0; j <= plen; j++ {
			if i == 0 && j == 0 {
				dp[i][j] = true
			} else if i == 0 {
				dp[i][j] = p[j-1] == '*' && dp[i][j-1]
			} else if j == 0 {
				dp[i][j] = false // impossible to match
			} else {
				if s[i-1] == p[j-1] || p[j-1] == '?' {
					dp[i][j] = dp[i-1][j-1]
					continue
				}

				// current not match.
				if p[j-1] == '*' {
					if dp[i-1][j] {
						// without current p char, it can make a match. let * also contains the last value
						dp[i][j] = true
					} else if dp[i][j-1] {
						dp[i][j] = true
					}
				}
			}
		}
	}

	return dp[slen][plen]
}

func trap(height []int) int {
	len := len(height)
	left := make([]int, len)
	right := make([]int, len)
	for i := 0; i < len; i++ {
		if i == 0 {
			left[i] = 0
			right[len-i-1] = 0
		} else {
			left[i] = height[i-1]
			if left[i] < left[i-1] {
				left[i] = left[i-1]
			}

			right[len-i-1] = height[len-i]
			if right[len-i-1] < right[len-i] {
				right[len-i-1] = right[len-i]
			}
		}
	}

	res := 0
	for i := 0; i < len; i++ {
		bound := left[i]
		if bound > right[i] {
			bound = right[i]
		}

		if v := bound - height[i]; v > 0 {
			res += v
		}

	}
	return res
}

func combinationSum2(candidates []int, target int) [][]int {
	sort.Ints(candidates)
	var res [][]int
	var cur []int
	combinationSum2DFS(candidates, target, 0, &res, cur, true)
	return res
}

func combinationSum2DFS(candidates []int, target int, index int, res *[][]int, cur []int, skip bool) {
	if target < 0 {
		return
	} else if index == len(candidates) {
		if target == 0 {
			*res = append(*res, append([]int{}, cur...))
		}
		return
	} else {
		combinationSum2DFS(candidates, target, index+1, res, cur, true)
		if index == 0 || candidates[index] != candidates[index-1] || !skip {
			combinationSum2DFS(candidates, target-candidates[index], index+1, res, append(cur, candidates[index]), false)
		}
	}
}

func combinationSum(candidates []int, target int) [][]int {
	sort.Ints(candidates)
	var res [][]int
	var cur []int
	combinationSumDFS(candidates, target, 0, &res, cur)
	return res
}

func combinationSumDFS(candidates []int, target int, index int, res *[][]int, cur []int) {
	if target < 0 {
		return
	} else if target == 0 {
		*res = append(*res, append([]int{}, cur...))
		return
	} else {
		for i := index; i < len(candidates); i++ {
			combinationSumDFS(candidates, target-candidates[i], i, res, append(cur, candidates[i]))
		}
	}
}

func countAndSay(n int) string {
	if n == 1 {
		return "1"
	} else {
		res := ""
		base := countAndSay(n - 1)
		lastBeginning := 0
		for pos := range base {
			if base[lastBeginning] == base[pos] {
				continue
			}
			len := pos - lastBeginning
			res = fmt.Sprintf("%s%d%s", res, len, string(base[lastBeginning]))
			lastBeginning = pos
		}
		len := len(base) - lastBeginning
		res = fmt.Sprintf("%s%d%s", res, len, string(base[lastBeginning]))
		return res
	}
}

func searchInsert(nums []int, target int) int {
	if target > nums[len(nums)-1] {
		return len(nums)
	}
	lo := 0
	hi := len(nums) - 1
	for lo < hi {
		mid := lo + (hi-lo)/2
		if nums[mid] < target {
			lo = mid + 1
		} else {
			hi = mid
		}
	}

	return lo
}

func longestValidParentheses(s string) int {
	var stack []int
	stack = append(stack, -1)
	for pos, char := range s {
		if char == '(' {
			stack = append(stack, pos)
		} else { // char == ')'
			n := len(stack) - 1
			if stack[n] == -1 || s[stack[n]] != '(' {
				stack = append(stack, pos)
			} else {
				stack = stack[:n]
			}
		}
	}

	best := 0
	lastIndex := len(s)
	for len(stack) > 0 {
		n := len(stack) - 1
		if lastIndex-stack[n]-1 > best {
			best = lastIndex - stack[n] - 1
		}

		lastIndex = stack[n]
		stack = stack[:n]
	}

	if best <= 1 {
		return 0
	}

	return best
}

func search(nums []int, target int) int {
	lo := 0
	hi := len(nums) - 1
	for lo < hi {
		mid := lo + (hi-lo)/2
		val := nums[mid]
		if val == target {
			return mid
		}
		if nums[lo] <= nums[mid] {
			if nums[lo] <= target && target < nums[mid] {
				hi = mid
			} else {
				lo = mid + 1
			}
		} else {
			if nums[mid] < target && target <= nums[hi] {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
	}

	if nums[lo] == target {
		return lo
	}

	return -1
}

func searchRange(nums []int, target int) []int {
	if len(nums) == 0 {
		return []int{-1, -1}
	}
	lo := 0
	hi := len(nums) - 1
	for lo < hi {
		mid := lo + (hi-lo)/2
		if nums[mid] >= target {
			hi = mid
		} else {
			// nums[mid] < target
			lo = mid + 1
		}
	}

	if nums[lo] != target {
		return []int{-1, -1}
	}

	res := []int{lo, lo}
	hi = len(nums) - 1
	for lo < hi {
		mid := lo + (hi-lo)/2
		if nums[mid] <= target {
			lo = mid + 1
		} else {
			hi = mid
		}
	}

	if nums[lo] == target {
		res[1] = lo
	} else {
		res[1] = lo - 1
	}
	return res
}
