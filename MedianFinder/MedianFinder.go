package MedianFinder

import (
	"container/heap"
	. "leetcodewithgolang/structure"
)

type MedianFinder struct {
	minHeap *MinIntHeap
	maxHeap *MaxIntHeap
}

/** initialize your data structure here. */
func MedianFinder_Constructor() MedianFinder {
	mf := MedianFinder{
		minHeap: &MinIntHeap{},
		maxHeap: &MaxIntHeap{},
	}

	heap.Init(mf.maxHeap)
	heap.Init(mf.minHeap)
	return mf
}

func (this *MedianFinder) AddNum(num int) {
	if this.minHeap.Len() > 0 && num >= (*this.minHeap)[0] {
		heap.Push(this.minHeap, num)
	} else {
		heap.Push(this.maxHeap, num)
	}

	if this.maxHeap.Len() == this.minHeap.Len() || this.maxHeap.Len() == this.minHeap.Len()+1 {
		return
	}

	if this.maxHeap.Len() > this.minHeap.Len() {
		removeVal := heap.Pop(this.maxHeap)
		heap.Push(this.minHeap, removeVal)
	} else {
		removeVal := heap.Pop(this.minHeap)
		heap.Push(this.maxHeap, removeVal)
	}
}

func (this *MedianFinder) FindMedian() float64 {
	if this.maxHeap.Len() == this.minHeap.Len() {
		return float64((*this.maxHeap)[0]+(*this.minHeap)[0]) / 2.0
	} else {
		return float64((*this.maxHeap)[0] * 1.0)
	}
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * obj := Constructor();
 * obj.AddNum(num);
 * param_2 := obj.FindMedian();
 */
