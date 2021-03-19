package LRU

type LRUCache struct {
	valueMap map[int]int
	keyMap   map[int](*DoubleLinkedListNode)
	cap      int
	data     *DoubleLinkedList
}

func Constructor(capacity int) LRUCache {
	return LRUCache{
		cap:      capacity,
		valueMap: make(map[int]int),
		keyMap:   make(map[int]*DoubleLinkedListNode),
		data:     &DoubleLinkedList{},
	}
}

func (this *LRUCache) Get(key int) int {
	if v, ok := this.valueMap[key]; ok {
		node := this.keyMap[key]
		this.data.moveToHead(node)
		return v
	} else {
		return -1
	}
}

func (this *LRUCache) Put(key int, value int) {
	this.valueMap[key] = value
	if node, ok := this.keyMap[key]; ok {
		this.data.moveToHead(node)
	} else {
		head := this.data.addToHead(key)
		this.keyMap[key] = head
		if len(this.valueMap) > this.cap {
			tail := this.data.deleteTail()
			tailKey := tail.key
			delete(this.valueMap, tailKey)
			delete(this.keyMap, tailKey)
		}
	}
}

type DoubleLinkedListNode struct {
	prev *DoubleLinkedListNode
	next *DoubleLinkedListNode
	key  int
}

type DoubleLinkedList struct {
	head *DoubleLinkedListNode
	tail *DoubleLinkedListNode
}

func (list *DoubleLinkedList) addToHead(key int) *DoubleLinkedListNode {
	return list.addToHeadWithNode(&DoubleLinkedListNode{key: key})
}

func (list *DoubleLinkedList) addToHeadWithNode(newNode *DoubleLinkedListNode) *DoubleLinkedListNode {
	if list.head == nil {
		list.head = newNode
		list.tail = newNode
	} else {
		oldHead := list.head
		newNode.next = oldHead
		oldHead.prev = newNode
		list.head = newNode
	}

	return newNode
}

func (list *DoubleLinkedList) deleteTail() *DoubleLinkedListNode {
	if list.head == list.tail {
		tail := list.tail
		list.head = nil
		list.tail = nil
		return tail
	} else {
		tail := list.tail
		newTail := list.tail.prev
		newTail.next = nil
		list.tail = newTail
		tail.prev = nil
		return tail
	}
}

func (list *DoubleLinkedList) moveToHead(node *DoubleLinkedListNode) {
	if node == list.tail {
		tail := list.deleteTail()
		list.addToHeadWithNode(tail)
	} else if node != list.head {
		prev := node.prev
		next := node.next
		prev.next = next
		next.prev = prev
		node.prev = nil
		node.next = list.head
		list.head.prev = node
		list.head = node
	}
}
