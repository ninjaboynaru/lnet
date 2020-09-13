package main

import "fmt"

type neuron struct {
	weights           vector
	bias              float64
	derivativeInputs  matrix
	derivativeWeights vector
	derivativeBias    float64
}

func newNeuron(inputCount int) neuron {
	if inputCount <= 0 {
		panic(fmt.Sprintf("Can not create neuron with input count %d", inputCount))
	}

	var bias float64 = randRangeFloat64(0, 1)
	var weights []float64 = make(vector, inputCount)
	for index := range weights {
		weights[index] = randRangeFloat64(0.1, 1)
	}

	return neuron{weights: weights, bias: bias}
}
