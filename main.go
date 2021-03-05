package main

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

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

func main() {
	//input := []int{2,0,2,1,1,0}
	//root := &TreeNode{Val:1}
	//n1 := &TreeNode{Val:3}
	//n2 := &TreeNode{Val:2}
	//root.Left = n1
	//n1.Right = n2
	//recoverTree(root)
	//fmt.Print(res)
	//i, err := strconv.Atoi("-02")
	//res := maximalRectangle([][]byte{[]byte("10100"), []byte("10111"), []byte("11111"), []byte("10010")})
	res := climbStairs(45)
	fmt.Print(res)
	//fmt.Print(err)
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
	stack := Stack{}

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

func recoverTree(root *TreeNode) {
	stack := TreeStack{}
	var prev *TreeNode
	var next *TreeNode
	var last *TreeNode
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

func inorderTraversal(root *TreeNode) []int {
	res := make([]int, 0)
	dfs_inorderTraversal(root, &res)
	return res

}

func dfs_inorderTraversal(root *TreeNode, res *[]int) {
	if root == nil {
		return
	}
	dfs_inorderTraversal(root.Left, res)
	*res = append(*res, root.Val)
	dfs_inorderTraversal(root.Right, res)
}

func preorderTraversal(root *TreeNode) []int {
	//res := make([]int, 0)
	//dfs_preorderTraversal(root, &res)
	//return res
	stack := TreeStack{}
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

func dfs_preorderTraversal(root *TreeNode, i *[]int) {
	if root == nil {
		return
	}
	*i = append(*i, root.Val)
	dfs_preorderTraversal(root.Left, i)
	dfs_preorderTraversal(root.Right, i)
}

func postorderTraversal(root *TreeNode) []int {
	res := make([]int, 0)
	dfs_postorderTraversal(root, &res)
	return res
}

func dfs_postorderTraversal(root *TreeNode, i *[]int) {
	if root == nil {
		return
	}
	dfs_postorderTraversal(root.Left, i)
	dfs_postorderTraversal(root.Right, i)
	*i = append(*i, root.Val)
}

func reverseList(head *ListNode) *ListNode {
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

func partition(head *ListNode, x int) *ListNode {
	sHead := ListNode{}
	lHead := ListNode{}
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

func deleteDuplicates(head *ListNode) *ListNode {
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

func deleteDuplicates2(head *ListNode) *ListNode {
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
	stack := Stack{}
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

func max(i int, j int) int {
	if i > j {
		return i
	} else {
		return j
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
				dp[i][j] = 1 + min(min(dp[i-1][j], dp[i][j-1]), dp[i-1][j-1])
				if word1[i-1] == word2[j-1] {
					dp[i][j] = min(dp[i][j], dp[i-1][j-1])
				}

			}
		}
	}

	return dp[len1-1][len2-1]
}

func min(i int, j int) int {
	if i < j {
		return i
	} else {
		return j
	}
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

func isSameTree(p *TreeNode, q *TreeNode) bool {
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

func rotateRight(head *ListNode, k int) *ListNode {
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

func findListLength(head *ListNode) (int, *ListNode) {
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
		possibleBest = max(possibleBest, i+nums[i])
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
		r += fmt.Sprintf("%s%d", string('a'+i), res[i])
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
		possibleBest = max(possibleBest, i+nums[i])
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
		for pos, _ := range base {
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
