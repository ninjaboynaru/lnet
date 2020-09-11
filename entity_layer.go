package main

import (
	"fmt"
)

type layer struct {
	layerSize  int
	inputCount int
	lastInput  matrix
	neurons    []neuron
}

func newLayer(layerSize, inputCount int) layer {
	if layerSize <= 0 {
		panic(fmt.Sprintf("Can not create layer with size %d", layerSize))
	}

	if inputCount <= 0 {
		panic(fmt.Sprintf("Can not create layer with input count %d", inputCount))
	}

	var neurons []neuron = make([]neuron, layerSize)
	for index := range neurons {
		neurons[index] = newNeuron(inputCount)
	}

	return layer{layerSize: layerSize, inputCount: inputCount, neurons: neurons}
}

func newLayerExplicit(weights matrix, biases vector) layer {
	var neuronCount = len(weights)
	var biasCount = len(biases)

	if neuronCount == 0 {
		panic("Can not create layer with 0 neurons")
	}

	if biasCount == 0 {
		panic("Can not create layer with 0 biases")
	}

	if neuronCount != biasCount {
		panic(fmt.Sprintf("Layer neuron count %d does not match layer bias count %d", neuronCount, biasCount))
	}

	var firstWeightSetLen int = len(weights[0])
	for index := range weights {
		var currentWeightSet vector = weights[index]
		if len(currentWeightSet) != firstWeightSetLen {
			panic("Found neurons in layer with differint input counts")
		}
	}

	var l layer = newLayer(neuronCount, firstWeightSetLen)
	l.layerSize = neuronCount
	l.inputCount = firstWeightSetLen

	for index := range l.neurons {
		var n *neuron = &l.neurons[index]
		n.weights = weights[index]
		n.bias = biases[index]
	}

	return l
}

func (l layer) singleInputForward(input vector) vector {
	if l.inputCount != len(input) {
		panic(fmt.Sprintf("Layer input count %d does not match len of provided input %d", l.inputCount, len(input)))
	}

	var output vector = make(vector, l.layerSize)

	for neuronIndex, n := range l.neurons {
		var neuronOutput float64

		for weightIndex, weight := range n.weights {
			neuronOutput += weight * input[weightIndex]
		}

		output[neuronIndex] = neuronOutput + n.bias
	}

	return output
}

func (l *layer) forward(input matrix) matrix {
	if len(input) == 0 {
		panic("Can not forward layer with empty input batch")
	}

	var output matrix = make(matrix, len(input))

	for rowIndex, inputSample := range input {
		output[rowIndex] = l.singleInputForward(inputSample)
	}

	l.lastInput = input
	return output
}

func (l *layer) backward(forwardDerivatives matrix) {
	for _, forwardDerivativeRow := range forwardDerivatives {
		var forwardLen int = len(forwardDerivativeRow)
		if forwardLen != l.layerSize {
			panic(fmt.Sprintf("Forward derivatives contain a row whose length %d does not match layer size %d", forwardLen, l.layerSize))
		}
	}

	for neuronIndex := range l.neurons {
		var n *neuron = &l.neurons[neuronIndex]
		var neuronForwardDerivative float64 = 0

		for _, forwardRow := range forwardDerivatives {
			neuronForwardDerivative += forwardRow[neuronIndex]
		}

		n.derivativeInputs = make(vector, len(n.weights))
		for derivativeInputIndex := range n.derivativeInputs {
			n.derivativeInputs[derivativeInputIndex] = n.weights[derivativeInputIndex] * neuronForwardDerivative
		}

		n.derivativeWeights = make(vector, len(n.weights))
		for _, inputSample := range l.lastInput {
			for weightIndex, weightValue := range n.derivativeWeights {
				n.derivativeWeights[weightIndex] = weightValue + (inputSample[weightIndex] * neuronForwardDerivative)
			}
		}

		n.derivativeBias = 1 * neuronForwardDerivative

	}

}
