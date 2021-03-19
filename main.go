package main

import (
	"fmt"
	"leecodewithgolang/structure"
	"sort"
	"strconv"
	"strings"
)

func main() {
	res := largestNumber([]int{3, 30, 34, 5, 9})
	fmt.Print(res)
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
	stack := structure.Stack{}
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
