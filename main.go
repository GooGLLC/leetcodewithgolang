package main

import (
	"fmt"
	. "leecodewithgolang/UnionFind"
	. "leecodewithgolang/structure"
	. "leecodewithgolang/util"
	"sort"
	"strconv"
	"strings"
)

func main() {
	n := []int{0, 1, 2, 3, 4}
	res := rob(n[:1])
	fmt.Print(res)
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

//func calculateMinimumHP(dun [][]int) int {
//	h := len(dun)
//	w := len(dun[0])
//	dp := make([][]int, h) // min health to enter this room
//	for i := 0; i < h; i++{
//		dp[i] = make([]int, w)
//	}
//
//	for i := h-1; i >=0 ; i--{
//		for j:= w-1; j >= 0; j--{
//			if i == h - 1 && j == w- 1 {
//				if dun[i][j] >= 0 {
//					dp[i][j] = 1
//				} else {
//					dp[i][j] = -dun[i][j] + 1
//				}
//			} else if i == h - 1{
//				dp[i][j] = max(1, dp[i][j+1] - dun[i][j])
//			} else if j == w- 1 {
//				dp[i][j] = max(1, dp[i+1][j] - dun[i][j])
//			} else {
//				dp[i][j] = max(1, min(dp[i][j+1], dp[i+1][j])- dun[i][j])
//			}
//		}
//	}
//
//	return dp[0][0]
//}

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
