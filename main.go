package main

import (
	"fmt"
	"sort"
	"strings"
)

func max(i int, j int) int {
	if i > j {
		return i
	} else {
		return j
	}
}

type ListNode struct {
	Val  int
	Next *ListNode
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func main() {
	//res := merge([][]int{"eat","tea","tan","ate","nat","bat"})
	//fmt.Print(res)
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
