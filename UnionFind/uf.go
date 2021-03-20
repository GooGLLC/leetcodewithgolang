package UnionFind

type UF struct {
	root []int
}

func UFConstructor(N int) UF {
	d := make([]int, N)
	for i := 0; i < len(d); i++ {
		d[i] = i
	}

	return UF{
		root: d,
	}
}

func (u *UF) Union(i, j int) {
	if !u.IsUnion(i, j) {
		iroot := u.FindRoot(i)
		jroot := u.FindRoot(j)
		u.root[iroot] = jroot
	}
}

func (u *UF) IsUnion(i, j int) bool {
	return u.FindRoot(i) == u.FindRoot(j)
}

func (u *UF) FindRoot(i int) int {
	path := make([]int, 0)
	for i != u.root[i] {
		path = append(path, i)
		i = u.root[i]
	}

	for j := 0; j < len(path); j++ {
		u.root[path[j]] = u.root[i]
	}

	return u.root[i]
}
