package structure

type TrieNode struct {
	next  []*TrieNode
	valid bool
}

func buildTrieNode() *TrieNode {
	node := &TrieNode{
		next: make([]*TrieNode, 26),
	}

	return node
}

type Trie struct {
	root *TrieNode
}

/** Initialize your data structure here. */
func Constructor() Trie {
	return Trie{root: buildTrieNode()}
}

/** Inserts a word into the trie. */
func (this *Trie) Insert(word string) {
	this.insert(word, this.root)
}

/** Returns if the word is in the trie. */
func (this *Trie) Search(word string) bool {
	return search(word, this.root)
}

func search(word string, root *TrieNode) bool {
	c := word[0]
	index := c - 'a'
	if root.next[index] == nil {
		return false
	} else {
		nextNode := root.next[index]
		if len(word) == 1 {
			return nextNode.valid
		} else {
			return search(word[1:], nextNode)
		}
	}
}

/** Returns if there is any word in the trie that starts with the given prefix. */
func (this *Trie) StartsWith(prefix string) bool {
	return startsWith(prefix, this.root)
}

func startsWith(prefix string, root *TrieNode) bool {
	if len(prefix) == 0 {
		return true
	} else {
		c := prefix[0]
		index := c - 'a'
		if root.next[index] == nil {
			return false
		} else {
			return startsWith(prefix[1:], root.next[index])
		}
	}
}

func (this *Trie) insert(word string, root *TrieNode) {
	c := word[0]
	index := c - 'a'
	if root.next[index] == nil {
		root.next[index] = buildTrieNode()
	}

	if len(word) == 1 {
		root.next[index].valid = true
	} else {
		this.insert(word[1:], root.next[index])
	}
}

/**
 * Your Trie object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Insert(word);
 * param_2 := obj.Search(word);
 * param_3 := obj.StartsWith(prefix);
 */
