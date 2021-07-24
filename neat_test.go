package neat

import (
	"fmt"
	"testing"
)

func TestNet(t *testing.T) {
	networkConfig := NetworkConfig{
		inputs:             4,
		outputs:            6,
		squash:             SquashSigmoid,
		evalIterations:     10,
		weightRange:        Range{-5, 5},
		nudgeRange:         Range{-0.2, 0.2},
		changeWeightChance: 0.1,
	}

	neatConfig := NeatConfig{
		networkConfig:   networkConfig,
		size:            20,
		startingWeights: 5,
		c1:              1, c2: 1, c3: 1., threshold: 3.0,
		addGeneChance:      0.05,
		addNodeChance:      0.03,
		nudgeWeightsChance: 0.8,
	}

	neat := NewNeat(neatConfig)

	for i := 0; i < 100; i++ {
		for _, net := range neat.pop {
			net.fitness = Range{0, 10}.Random()
		}
		fmt.Println("GEN", neat.gen)
		specList := neat.Speciate()
		fmt.Println(specList)
		neat.Reproduce()
	}
}
