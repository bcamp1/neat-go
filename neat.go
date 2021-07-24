package neat

import (
	"fmt"
	_ "fmt"
	"math"
	"math/rand"
	"sort"

	"example.com/butils"
)

type NeatConfig struct {
	size                                             int
	networkConfig                                    NetworkConfig
	startingWeights                                  int
	c1, c2, c3, threshold                            float64
	addGeneChance, addNodeChance, nudgeWeightsChance float64
}

type Neat struct {
	inno   uint
	gen    int
	config NeatConfig
	pop    []*Network
}

type Species struct {
	orgs        []*Network
	offspring   int
	adjustedSum float64
}

func (s Species) Rep() (*Network, bool) {
	if len(s.orgs) > 0 {
		return s.orgs[0], true
	} else {
		return &Network{}, false
	}
}

func NewSpecies(rep *Network) *Species {
	offspring := 0
	adjustedSum := 0.0
	orgs := append(make([]*Network, 0), rep)

	return &Species{orgs, offspring, adjustedSum}
}

type SpeciesList []*Species

func (s SpeciesList) String() string {
	str := "Species: "
	for _, spec := range s {
		str += fmt.Sprint(len(spec.orgs), " ")
	}
	return str
}

func (s SpeciesList) PrintOffspring() {
	str := "Offspring: "
	for _, spec := range s {
		str += fmt.Sprint(spec.offspring, " ")
	}
	fmt.Println(str)
}

func NewNeat(config NeatConfig) Neat {
	var inno uint = 0
	gen := 0

	seedNet := NewNetwork(config.networkConfig)

	if config.startingWeights > int(config.networkConfig.inputs*config.networkConfig.outputs) {
		panic("Too many starting weights")
	}

	for len(seedNet.genes) < config.startingWeights {
		seedNet.MutateAddGene(&inno)
	}

	pop := make([]*Network, 0)
	for i := 0; i < config.size; i++ {
		newNet := seedNet.Copy()
		newNet.RandomizeWeights()
		pop = append(pop, &newNet)
	}

	neat := Neat{inno, gen, config, pop}

	return neat
}

func (neat *Neat) MutateNetwork(n *Network, inno *uint) {
	if butils.RandomChance(neat.config.addGeneChance) {

		n.MutateAddGene(inno)
	}

	if butils.RandomChance(neat.config.addNodeChance) {
		n.MutateAddNode(inno)
	}

	if butils.RandomChance(neat.config.nudgeWeightsChance) {
		n.NudgeWeights()
	}
}

func (neat *Neat) Distance(n1, n2 Network) float64 {
	min1, min2 := n1.MinId(), n2.MinId()
	max1, max2 := n1.MaxId(), n2.MaxId()

	var min, max uint
	var excessStart, excessEnd uint

	if min1 < min2 {
		min = min1
		excessStart = min2
	} else {
		min = min2
		excessStart = min2
	}

	if max1 > max2 {
		max = max1
		excessEnd = max2
	} else {
		max = max2
		excessEnd = max1
	}

	var excess, disjoint int

	weightDiffs := make([]float64, 0)

	for id := min; id <= max; id++ {
		n1Gene, n1Has := n1.Gene(id)
		n2Gene, n2Has := n2.Gene(id)

		if n1Has != n2Has {
			if id < excessStart || id > excessEnd {
				excess++
			} else {
				disjoint++
			}
		} else if n1Has == n2Has {
			w1 := n1Gene.weight
			w2 := n2Gene.weight
			weightDiffs = append(weightDiffs, math.Abs(w1-w2))
		}
	}

	// Get average of weightDiffs
	sum := 0.0
	for i := range weightDiffs {
		sum += weightDiffs[i]
	}

	wBar := 0.0

	if len(weightDiffs) > 0 {
		wBar = sum / float64(len(weightDiffs))
	}

	var N int
	if len(n1.genes) > len(n2.genes) {
		N = len(n1.genes)
	} else {
		N = len(n2.genes)
	}

	if N < 20 {
		N = 1
	}

	c1 := neat.config.c1
	c2 := neat.config.c2
	c3 := neat.config.c3

	return c1*(float64(excess)/float64(N)) + c2*(float64(disjoint)/float64(N)) + c3*wBar
}

func PrintFitness(group []*Network) {
	str := "Fitness: "
	for i := range group {
		str += fmt.Sprintf("%.3g ", group[i].fitness)
	}
	fmt.Println(str)
}

func PrintAdjustedFitness(group []*Network) {
	str := "Adjusted: "
	for i := range group {
		str += fmt.Sprintf("%.3g ", group[i].adjustedFitness)
	}
	fmt.Println(str)
}

func BestFitness(group []*Network) *Network {
	bestFitness := 0.0
	var bestNet *Network

	for i := range group {
		if group[i].fitness > bestFitness {
			bestFitness = group[i].fitness
			bestNet = group[i]
		}
	}

	return bestNet
}

func SortByFitness(group *[]*Network) {
	sort.Slice(*group, func(i, j int) bool {
		return (*group)[i].fitness > (*group)[j].fitness
	})
}

func SortByAdjustedFitness(group *[]*Network) {
	sort.Slice(*group, func(i, j int) bool {
		return (*group)[i].adjustedFitness > (*group)[j].adjustedFitness
	})
}

func SumAdjustedFitness(group []*Network) float64 {
	sum := 0.0
	for i := range group {
		sum += group[i].adjustedFitness
	}
	return sum
}

func (neat *Neat) Speciate() SpeciesList {
	speciesList := make(SpeciesList, 0)

	for _, network := range neat.pop {
		found := false
		for i := range speciesList {
			species := speciesList[i]
			rep, ok := species.Rep()
			if ok {
				distance := neat.Distance(*network, *rep)
				if distance <= neat.config.threshold {
					species.orgs = append(species.orgs, network)
					found = true
					break
				}
			}
		}
		if !found {
			speciesList = append(speciesList, NewSpecies(network))
		}
	}

	//Assign adjusted fitness
	for _, species := range speciesList {
		n := len(species.orgs)
		sum := 0.0
		for _, org := range species.orgs {
			org.adjustedFitness = org.fitness / float64(n)
			sum += org.adjustedFitness
		}

		species.adjustedSum = sum

		// Sort each species by fitness
		SortByFitness(&species.orgs)
	}

	// Sort species list by adjusted sum
	sort.Slice(speciesList, func(i, j int) bool {
		return speciesList[i].adjustedSum > speciesList[j].adjustedSum
	})

	return speciesList
}

func (neat *Neat) Reproduce() {
	speciesList := neat.Speciate()
	totalAdjustedSum := 0.0
	for i := range speciesList {
		totalAdjustedSum += speciesList[i].adjustedSum
	}

	// Set offspring
	fitnessPerOffspring := totalAdjustedSum / float64(neat.config.size)
	if fitnessPerOffspring == 0 {
		fitnessPerOffspring = 1
	}
	offspringSum := 0

	for i := range speciesList {
		speciesList[i].offspring = int(speciesList[i].adjustedSum / fitnessPerOffspring)
		offspringSum += speciesList[i].offspring
	}

	// Sprinkle in remaining
	offspringRemaining := neat.config.size - offspringSum
	for i := 0; i < offspringRemaining; i++ {
		speciesList[i%len(speciesList)].offspring += 1
	}

	speciesList.PrintOffspring()

	// Reproduce
	newPop := make([]*Network, 0)
	for _, spec := range speciesList {
		// Copy champion
		if len(spec.orgs) > 5 && spec.offspring > 0 {
			champion := BestFitness(spec.orgs).Copy()
			newPop = append(newPop, &champion)
			spec.offspring -= 1
		}

		for spec.offspring > 0 {
			newOrg := spec.orgs[rand.Intn(len(spec.orgs))].Copy()
			neat.MutateNetwork(&newOrg, &neat.inno)
			newPop = append(newPop, &newOrg)
			spec.offspring -= 1
		}
	}

	neat.pop = newPop
	neat.gen++
}
