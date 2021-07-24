package neat

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"example.com/butils"
)

type NodeType int

const (
	Input NodeType = iota
	Output
	Hidden
)

func (nt NodeType) String() string {
	switch nt {
	case Input:
		return "INPUT"
	case Output:
		return "OUTPUT"
	case Hidden:
		return "HIDDEN"
	default:
		return "UNKNOWN"
	}
}

func SquashLinear(val float64) float64 {
	return val
}

func SquashSigmoid(val float64) float64 {

	return 1 / (1 + math.Exp(-val))
}

type Range struct{ min, max float64 }

func (r Range) Random() float64 {
	return r.min + rand.Float64()*(r.max-r.min)
}

type Gene struct {
	enabled bool
	in, out uint
	weight  float64
}

func (g Gene) String() string {
	return fmt.Sprintf("%t %v->%v (%.3g)\n", g.enabled, g.in, g.out, g.weight)
}

type NetworkConfig struct {
	inputs, outputs    uint
	squash             func(float64) float64
	evalIterations     uint
	weightRange        Range
	nudgeRange         Range
	changeWeightChance float64
}

type Network struct {
	genes                    map[uint]*Gene
	fitness, adjustedFitness float64
	config                   NetworkConfig
}

func (n Network) Copy() Network {
	newGenes := make(map[uint]*Gene)
	for i := range n.genes {
		newGene := *n.genes[i]
		newGenes[i] = &newGene
	}

	n.genes = newGenes
	return n
}

func (n Network) String() string {
	str := "NETWORK--------\n"
	for k, v := range n.genes {
		str += fmt.Sprintf("%t [%v] %v->%v (%.3g)\n", v.enabled, k, v.in, v.out, v.weight)
	}
	str += "---------------\n"
	return str
}

func (n Network) MaxId() (max uint) {
	for id := range n.genes {
		if id > max {
			max = id
		}
	}
	return
}

func (n Network) MinId() (min uint) {
	min = n.MaxId()
	for id := range n.genes {
		if id < min {
			min = id
		}
	}
	return
}

func (n Network) MaxNode() (max uint) {
	for _, node := range n.Nodes() {
		if node > max {
			max = node
		}
	}
	return
}

func (n Network) HasGene(id uint) bool {
	_, ok := n.genes[id]
	return ok
}

func (n Network) Gene(id uint) (Gene, bool) {
	gene, ok := n.genes[id]
	if ok {
		return *gene, true
	} else {
		return Gene{}, false
	}
}

func (n Network) GeneInOut(in, out uint) (Gene, bool) {
	for _, gene := range n.genes {
		if gene.in == in && gene.out == out {
			return *gene, true
		}
	}
	return Gene{}, false
}

func (n *Network) AddGene(id, in, out uint, weight float64) error {
	if n.HasGene(id) {
		return fmt.Errorf("Already has gene of id %v", id)
	}

	enabled := true
	gene := Gene{enabled, in, out, weight}
	n.genes[id] = &gene
	return nil
}

func (n *Network) NudgeWeights() {
	for id := range n.genes {
		var newWeight float64
		if butils.RandomChance(n.config.changeWeightChance) {
			newWeight = n.config.weightRange.Random()
		} else {
			newWeight = n.genes[id].weight + n.config.nudgeRange.Random()
		}
		n.genes[id].weight = newWeight
	}
}

func (n *Network) RandomizeWeights() {
	for id := range n.genes {
		newWeight := n.config.weightRange.Random()
		n.genes[id].weight = newWeight
	}
}

func (n *Network) MutateAddGene(inno *uint) {
	nodes := n.Nodes()
	inNodes := make([]uint, 0)
	outNodes := make([]uint, 0)

	for _, node := range nodes {
		switch n.NodeType(node) {
		case Input:
			inNodes = append(inNodes, node)
		case Hidden:
			inNodes = append(inNodes, node)
			outNodes = append(outNodes, node)
		case Output:
			outNodes = append(outNodes, node)
		}
	}

	const tries = 5

	for i := 0; i < tries; i++ {
		in := butils.RandomFrom(inNodes).(uint)
		out := butils.RandomFrom(outNodes).(uint)
		_, ok1 := n.GeneInOut(in, out)
		_, ok2 := n.GeneInOut(out, in)

		if in != out && !ok1 && !ok2 {
			weight := n.config.weightRange.Random()
			n.AddGene(*inno, in, out, weight)
			*inno += 1
			return
		}
	}

	fmt.Printf("Failed to add gene after %v tries\n", tries)

}

func (n *Network) MutateAddNode(inno *uint) {
	possibleGeneIds := make([]uint, 0)
	for id, gene := range n.genes {
		if gene.enabled {
			possibleGeneIds = append(possibleGeneIds, id)
		}
	}

	if len(possibleGeneIds) == 0 {
		fmt.Println("MutateAddNode: No genes to pick from")
		return
	}

	id := butils.RandomFrom(possibleGeneIds).(uint)
	gene := n.genes[id]

	var newNode uint
	for _, node := range n.Nodes() {
		if node > newNode {
			newNode = node
		}
	}

	newNode++

	gene.enabled = false

	n.AddGene(*inno, gene.in, newNode, 1.0)
	*inno++

	n.AddGene(*inno, newNode, gene.out, gene.weight)
	*inno++
}

func (n Network) GenesOutOf(node uint) []Gene {
	genes := make([]Gene, 0)
	for _, gene := range n.genes {
		if gene.in == node {
			genes = append(genes, *gene)
		}
	}
	return genes
}

func (n Network) GenesInto(node uint) []Gene {
	genes := make([]Gene, 0)
	for _, gene := range n.genes {
		if gene.out == node {
			genes = append(genes, *gene)
		}
	}
	return genes
}

func (n Network) activate(inputs ...float64) []float64 {
	if uint(len(inputs)) != n.config.inputs {
		panic("Wrong number of inputs")
	}

	nodeMap := make(map[uint]float64)

	for _, node := range n.Nodes() {
		nodeMap[node] = 0
	}

	// Set inputs
	for i := 0; i < int(n.config.inputs); i++ {
		nodeMap[uint(i)] = inputs[i]
	}

	for i := 0; i < int(n.config.evalIterations); i++ {
		for j := n.MaxNode(); j > n.config.inputs-1; j-- {
			_, ok := nodeMap[uint(j)]

			if ok {
				node := uint(j)
				var val float64
				for _, gene := range n.GenesInto(node) {
					if gene.enabled {
						val += gene.weight * nodeMap[gene.in]
					}
				}
				val = n.config.squash(val)
				nodeMap[node] = val
			}
		}
		fmt.Println(nodeMap)
	}

	outputs := make([]float64, 0)
	for i := n.config.inputs; i < (n.config.inputs + n.config.outputs); i++ {
		outputs = append(outputs, nodeMap[i])
	}
	return outputs
}

func (n Network) Nodes() []uint {
	nodeMap := make(map[uint]bool)
	for i := 0; i < int(n.config.inputs+n.config.outputs); i++ {
		nodeMap[uint(i)] = true
	}

	for _, gene := range n.genes {
		nodeMap[gene.in] = true
		nodeMap[gene.out] = true
	}

	nodes := make([]uint, 0)
	for k := range nodeMap {
		nodes = append(nodes, k)
	}

	return nodes
}

func (n Network) NodeType(node uint) NodeType {
	switch {
	case node < n.config.inputs:
		return Input
	case node < n.config.inputs+n.config.outputs:
		return Output
	default:
		return Hidden
	}
}

func NewNetwork(config NetworkConfig) Network {
	rand.Seed(time.Now().UnixNano())
	butils.InitRand()

	return Network{
		config: config,
		genes:  make(map[uint]*Gene),
	}
}
