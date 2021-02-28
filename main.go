package main

import (
	"fmt"
	"sort"
)

func main() {
	res := isMatch("acdcb", "a*c?b")
	fmt.Print(res)
}

func jump(nums []int) int {

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
