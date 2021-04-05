package main

import (
	"container/heap"
	"fmt"
	. "leetcodewithgolang/UnionFind"
	. "leetcodewithgolang/structure"
	. "leetcodewithgolang/util"
	"math"
	"sort"
	"strconv"
	"strings"
)

func main() {
	res := addOperators("1000000009",
		9)
	fmt.Print(res)
}

func findDuplicate(nums []int) int {
	swap := func(n []int, i, j int) {
		t := n[i]
		n[i] = n[j]
		n[j] = t
	}

	for i := 0; i < len(nums); i++ {
		wantIndex := nums[i] - 1
		if wantIndex == i {
			continue
		}

		if nums[wantIndex] == nums[i] {
			return nums[i]
		} else {
			swap(nums, wantIndex, i)
			i--
		}
	}

	return 0
}

func moveZeroes(nums []int) {
	nlen := len(nums)
	nextPos := 0
	for i := 0; i < nlen; i++ {
		if nums[i] != 0 {
			nums[nextPos] = nums[i]
			nextPos++
		}
	}

	for nextPos < nlen {
		nums[nextPos] = 0
		nextPos++
	}
}

func addOperators(num string, target int) []string {
	res := make([]string, 0)
	nlen := len(num)
	dlist := make([]string, nlen)
	addOperatorsHelper(num, target, 0, &dlist, 0, 0, 0, "+", &res)
	return res
}

func addOperatorsHelper(num string, target int, index int, op *[]string, curRes int, curNum int, lastNum int, lastOp string, res *[]string) {
	if index == len(num) {
		if (*op)[index-1] == "+" {
			if curRes == target {
				*res = append(*res, addOperatorsGetResult(num, op))
			}
		}
	} else {
		curDigit, _ := strconv.Atoi(string(num[index]))
		if curDigit != 0 || curNum != 0 {
			(*op)[index] = ""
			// handle first number cannot be 0 if curNum is more 1 digit
			addOperatorsHelper(num, target, index+1, op, curRes, curNum*10+curDigit, lastNum, lastOp, res)
		}

		(*op)[index] = "+"
		addOperatorsHelper(num, target, index+1, op, computeCurResHelper(curRes, lastOp, curNum*10+curDigit, lastNum, curDigit), 0, 0, "+", res)

		(*op)[index] = "-"
		addOperatorsHelper(num, target, index+1, op, computeCurResHelper(curRes, lastOp, curNum*10+curDigit, lastNum, curDigit), 0, 0, "-", res)

		(*op)[index] = "*"
		addOperatorsHelper(num, target, index+1, op, curRes, 0, getLastNumberMuliplier(curNum*10+curDigit, lastNum, lastOp), "*", res)
	}
}

func getLastNumberMuliplier(curN int, lastN int, op string) int {
	if op == "*" {
		return curN * lastN
	} else if op == "+" {
		return curN
	} else {
		return -curN
	}
}

func computeCurResHelper(res int, op string, curNum int, lastNum int, curDigit int) int {
	if op == "+" {
		return res + curNum
	} else if op == "-" {
		return res - curNum
	} else if op == "*" {
		return res + curNum*lastNum
	}

	return 0
}

func addOperatorsGetResult(num string, op *[]string) string {
	res := ""
	for i := 0; i < len(num); i++ {
		res += string(num[i])
		if i != len(num)-1 {
			res += (*op)[i]
		}
	}
	return res
}

func hIndex2(citations []int) int {
	lo := 0
	hi := len(citations)
	n := len(citations)

	for lo < hi {
		mid := (lo + hi) / 2
		if mid == 0 || citations[n-mid] >= mid {
			lo = mid + 1
		} else {
			hi = mid
		}
	}

	if lo == 0 || citations[n-lo] >= lo {
		return lo
	} else {
		return lo - 1
	}
}

func hIndex(citations []int) int {
	h := make([]int, 10001)
	for _, v := range citations {
		h[v]++
	}

	for i := 999; i >= 0; i-- {
		h[i] += h[i+1]

		if h[i] >= i {
			return i
		}
	}

	return 0
}

var numSquareMap map[int]int

func numSquares(n int) int {
	if n <= 1 {
		return n
	}

	if numSquareMap == nil {
		numSquareMap = make(map[int]int)
	} else if v, ok := numSquareMap[n]; ok {
		return v
	}

	best := n
	for i := 1; i < n; i++ {
		if i*i <= n {
			cur := 1 + numSquares(n-i*i)
			if cur < best {
				best = cur
			}
		} else {
			break
		}
	}

	numSquareMap[n] = best
	return best
}

func searchMatrix2(matrix [][]int, target int) bool {
	h := len(matrix)
	w := len(matrix[0])
	i := 0
	j := w - 1
	for {
		if matrix[i][j] == target {
			return true
		} else if matrix[i][j] < target {
			i++
		} else {
			j--
		}

		if i >= h || j < 0 {
			return false
		}
	}

	return false
}

func maxSlidingWindow(nums []int, k int) []int {
	deque := make([]int, 0)
	nlen := len(nums)
	res := make([]int, nlen-k+1)
	for i := 0; i < nlen; i++ {
		for len(deque) > 0 && nums[i] > deque[0] {
			deque = deque[1:]
		}

		// enque at head
		deque = append([]int{nums[i]}, deque...)
		curMax := deque[len(deque)-1]
		if 1+i-k >= 0 {
			res[1+i-k] = curMax
			previousValue := nums[1+i-k]
			if previousValue == deque[len(deque)-1] {
				deque = deque[:len(deque)-1]
			}
		}
	}

	return res
}

func productExceptSelf(nums []int) []int {
	nlen := len(nums)
	res := make([]int, nlen)
	res[0] = 1
	for i := 1; i < nlen; i++ {
		res[i] = res[i-1] * nums[i-1]
	}
	rightProduct := 1
	for i := nlen - 2; i >= 0; i-- {
		rightProduct *= nums[i+1]
		res[i] = res[i] * rightProduct
	}
	return res
}

func majorityElement2(nums []int) []int {
	var c1, c2, v1, v2 int
	v1 = -1
	v2 = -2
	for i := 0; i < len(nums); i++ {
		if nums[i] == v1 {
			c1++
		} else if nums[i] == v2 {
			c2++
		} else if c1 == 0 {
			c1 = 1
			v1 = nums[i]
		} else if c2 == 0 {
			c2 = 1
			v2 = nums[i]
		} else {
			c1--
			c2--
		}
	}

	res := make([]int, 0)
	limit := len(nums) / 3

	count1 := 0
	count2 := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] == v1 {
			count1++
		} else if nums[i] == v2 {
			count2++
		}
	}

	if count1 > limit {
		res = append(res, v1)
	}

	if count2 > limit {
		res = append(res, v2)
	}

	return res
}

func kthSmallest(root *TreeNode, k int) int {
	val := 0
	kthSmallestHelper(root, &k, &val)
	return val
}

func kthSmallestHelper(root *TreeNode, k *int, val *int) {
	if root.Left != nil {
		kthSmallestHelper(root.Left, k, val)
	}

	if *k == 1 {
		*val = root.Val
	}

	*k--

	if *k > 0 && root.Right != nil {
		kthSmallestHelper(root.Right, k, val)
	}
}

func getSkyline(buildings [][]int) [][]int {
	blen := len(buildings)
	points := make([]SkylinePoint, 2*blen)
	for i := 0; i < blen; i++ {
		points[i*2] = SkylinePoint{
			index:   buildings[i][0],
			isStart: true,
			height:  buildings[i][2],
		}
		points[i*2+1] = SkylinePoint{
			index:   buildings[i][1],
			isStart: false,
			height:  buildings[i][2],
		}
	}

	sort.Slice(points, func(i, j int) bool {
		p1 := points[i]
		p2 := points[j]
		if p1.index != p2.index {
			return p1.index < p2.index
		} else if p1.isStart != p2.isStart {
			return p1.isStart
		} else if p1.isStart {
			return p1.height >= p2.height
		} else {
			return p1.height <= p2.height
		}
	})

	maxHeap := &MaxIntHeap{}
	heap.Init(maxHeap)
	res := make([][]int, 0)
	curMaxHeight := -1
	for i := 0; i < len(points); i++ {
		curPoint := points[i]
		if curPoint.isStart {
			heap.Push(maxHeap, curPoint.height)
		} else {
			removeIndex := maxHeap.IndexOf(curPoint.height)
			heap.Remove(maxHeap, removeIndex)
		}

		topValue := 0
		if len(*maxHeap) > 0 {
			topValue = (*maxHeap)[0]
		}

		if topValue != curMaxHeight {
			res = append(res, []int{curPoint.index, topValue})
			curMaxHeight = topValue
		}
	}
	return res
}

type SkylinePoint struct {
	index   int
	isStart bool
	height  int
}

func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}

	if root == p || root == q {
		return root
	}

	leftRoot := lowestCommonAncestor(root.Left, p, q)
	rightRoot := lowestCommonAncestor(root.Right, p, q)
	if leftRoot != nil && rightRoot != nil {
		return root
	} else if leftRoot != nil {
		return leftRoot
	} else {
		return rightRoot
	}
}

func maximalSquare(matrix [][]byte) int {
	h := len(matrix)
	w := len(matrix[0])
	dp := make([][]int, h*w)
	for i := 0; i < h; i++ {
		dp[i] = make([]int, w)
	}
	best := 0
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			if matrix[i][j] == '1' {
				if i == 0 || j == 0 {
					dp[i][j] = 1
				} else {
					dp[i][j] = 1 + Min(dp[i-1][j-1], Min(dp[i-1][j], dp[i][j-1]))
				}
			}
			best = Max(best, dp[i][j])
		}
	}

	return best * best
}

func containsNearbyAlmostDuplicate(nums []int, k int, t int) bool {
	w := int64(t)
	windows := w + 1
	bucketToValue := make(map[int64]int64)
	for i := 0; i < len(nums); i++ {
		dequeue := i - k - 1
		if dequeue >= 0 {
			bucketIndex := int64(nums[dequeue]) / windows
			if nums[dequeue] < 0 {
				bucketIndex--
			}
			delete(bucketToValue, bucketIndex)
		}

		bucketIndex := int64(nums[i]) / windows
		if nums[i] < 0 {
			bucketIndex--
		}
		if _, ok := bucketToValue[bucketIndex]; ok {
			return true
		}
		if lowerValue, ok := bucketToValue[bucketIndex-1]; ok {
			if math.Abs(float64(int64(nums[i])-lowerValue)) <= float64(t) {
				return true
			}
		}
		if higerValue, ok := bucketToValue[bucketIndex+1]; ok {
			if math.Abs(float64(int64(nums[i])-higerValue)) <= float64(t) {
				return true
			}
		}

		bucketToValue[bucketIndex] = int64(nums[i])
	}

	return false
}

func countNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}

	leftHeight := getLeftHeight(root)
	rightHeight := getRightHeight(root)
	if leftHeight == rightHeight {
		return int(math.Pow(2.0, float64(leftHeight))) - 1
	}

	leftCount := countNodes(root.Left)
	rightCount := countNodes(root.Right)
	return 1 + leftCount + rightCount
}

func getRightHeight(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return 1 + getRightHeight(root.Right)
}

func getLeftHeight(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return 1 + getLeftHeight(root.Left)
}

func findOrder(numCourses int, prerequisites [][]int) []int {
	set := make(map[int]bool)
	for i := 0; i < numCourses; i++ {
		set[i] = true
	}

	link := make(map[int][]int)
	prevCount := make(map[int]int)
	for i := 0; i < len(prerequisites); i++ {
		cur := prerequisites[i]
		if link[cur[1]] == nil {
			link[cur[1]] = make([]int, 0)
		}
		link[cur[1]] = append(link[cur[1]], cur[0])
		set[cur[0]] = false
		prevCount[cur[0]]++
	}

	queue := Queue{}
	for k, v := range set {
		if v {
			queue.Enqueue(k)
		}
	}

	res := make([]int, 0)
	for len(queue) > 0 {
		qsize := len(queue)
		for i := 0; i < qsize; i++ {
			cur := queue.Dequeue()
			res = append(res, cur)
			for _, v := range link[cur] {
				prevCount[v]--
				if prevCount[v] == 0 {
					queue.Enqueue(v)
				}
			}
		}
	}

	if len(res) == numCourses {
		return res
	} else {
		return make([]int, 0)
	}
}

func rob(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	} else if len(nums) == 2 {
		return Max(nums[0], nums[1])
	} else {
		l := len(nums)
		return Max(rob2(nums[1:]), rob2(nums[:l-1]))
	}
}

func rob2(nums []int) int {
	nlen := len(nums)
	dp := make([]int, nlen)
	for i := 0; i < nlen; i++ {
		if i == 0 {
			dp[i] = nums[i]
		} else if i == 1 {
			dp[i] = Max(nums[1], nums[0])
		} else {
			dp[i] = Max(dp[i-1], nums[i]+dp[i-2])
		}
	}
	return dp[nlen-1]
}

func rightSideView(root *TreeNode) []int {
	levelToValue := make(map[int]int)
	rightSideViewDfs(root, levelToValue, 0)
	res := make([]int, 0)
	i := 0
	for {
		if v, ok := levelToValue[i]; ok {
			res = append(res, v)
		} else {
			break
		}
		i++
	}
	return res
}

func rightSideViewDfs(root *TreeNode, value map[int]int, level int) {
	if root == nil {
		return
	}
	rightSideViewDfs(root.Left, value, level+1)
	value[level] = root.Val
	rightSideViewDfs(root.Right, value, level+1)
}

func numIslands(grid [][]byte) int {
	h := len(grid)
	w := len(grid[0])
	uf := UFConstructor(h * w)

	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			if grid[i][j] == '1' {
				if i+1 < h {
					if grid[i+1][j] == '1' {
						uf.Union(i*w+j, (i+1)*w+j)
					}
				}

				if j+1 < w {
					if grid[i][j+1] == '1' {
						uf.Union(i*w+j, i*w+1+j)
					}
				}
			}
		}
	}

	set := make(map[int]bool)
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			if grid[i][j] == '1' {
				root := uf.FindRoot(i*w + j)
				set[root] = true
			}
		}
	}

	return len(set)
}

func findPeakElement(nums []int) int {
	lo := 0
	hi := len(nums) - 1
	for lo < hi {
		mid := (lo + hi) / 2
		if (mid-1 < 0 || nums[mid] > nums[mid-1]) && (mid+1 > len(nums) || nums[mid] > nums[mid+1]) {
			return mid
		} else if mid-1 < 0 {
			lo = mid + 1
		} else if mid+1 > len(nums) {
			hi = mid
		} else if nums[mid] < nums[mid-1] {
			hi = mid
		} else {
			lo = mid + 1
		}
	}

	return lo
}

func findRepeatedDnaSequences(s string) []string {
	dictToCount := make(map[string]int)
	for i := 0; i < len(s); i++ {
		endingIndex := i + 10
		if endingIndex > len(s) {
			break
		}

		subStr := s[i:endingIndex]
		if v, ok := dictToCount[subStr]; ok {
			dictToCount[subStr] = v + 1
		} else {
			dictToCount[subStr] = 1
		}
	}

	res := make([]string, 0)
	for k, v := range dictToCount {
		if v > 1 {
			res = append(res, k)
		}
	}

	return res
}

func calculateMinimumHP(dun [][]int) int {
	h := len(dun)
	w := len(dun[0])
	dp := make([][]int, h) // min health to enter this room
	for i := 0; i < h; i++ {
		dp[i] = make([]int, w)
	}

	for i := h - 1; i >= 0; i-- {
		for j := w - 1; j >= 0; j-- {
			if i == h-1 && j == w-1 {
				if dun[i][j] >= 0 {
					dp[i][j] = 1
				} else {
					dp[i][j] = -dun[i][j] + 1
				}
			} else if i == h-1 {
				dp[i][j] = Max(1, dp[i][j+1]-dun[i][j])
			} else if j == w-1 {
				dp[i][j] = Max(1, dp[i+1][j]-dun[i][j])
			} else {
				dp[i][j] = Max(1, Min(dp[i][j+1], dp[i+1][j])-dun[i][j])
			}
		}
	}

	return dp[0][0]
}

func largestNumber(nums []int) string {
	allzero := true
	for _, v := range nums {
		if v > 0 {
			allzero = false
			break
		}
	}
	if allzero {
		return "0"
	}

	sort.Slice(nums, func(i, j int) bool {
		leftValue := fmt.Sprintf("%d%d", nums[i], nums[j])
		rightValue := fmt.Sprintf("%d%d", nums[j], nums[i])
		slen := len(leftValue)
		for i := 0; i < slen; i++ {
			lc := leftValue[i]
			rc := rightValue[i]
			if lc == rc {
				continue
			} else {
				return lc > rc
			}
		}

		return true
	})

	sb := strings.Builder{}
	for _, v := range nums {
		sb.WriteString(strconv.Itoa(v))
	}
	return sb.String()
}

func majorityElement(nums []int) int {
	value := nums[0]
	count := 0
	for _, v := range nums {
		if v == value {
			count++
		} else {
			count--
		}
		if count < 0 {
			count = 1
			value = v
		}
	}

	return value
}

func evalRPN(tokens []string) int {
	stack := Stack{}
	for _, v := range tokens {
		switch v {
		case "+":
			{
				right := stack.Pop()
				left := stack.Pop()
				stack.Push(right + left)
			}
		case "-":
			{
				right := stack.Pop()
				left := stack.Pop()
				stack.Push(left - right)
			}
		case "*":
			{
				right := stack.Pop()
				left := stack.Pop()
				stack.Push(left * right)
			}
		case "/":
			{
				right := stack.Pop()
				left := stack.Pop()
				stack.Push(left / right)
			}
		default:
			{
				v, _ := strconv.Atoi(v)
				stack.Push(v)
			}
		}
	}

	return stack.Pop()
}
