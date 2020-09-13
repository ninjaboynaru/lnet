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

// getLayerInputDerivatives pprocesses all the layers neurons input derivatives into a single matrix.
func (l layer) getInputDerivatives() matrix {
	var inputDerivatives matrix = make(matrix, len(l.lastInput))

	for sampleIndex := range inputDerivatives {
		var inputDerivativeForSample vector = make(vector, l.inputCount)

		for inputDerivativeIndex := range inputDerivativeForSample {
			for _, n := range l.neurons {
				inputDerivativeForSample[inputDerivativeIndex] = inputDerivativeForSample[inputDerivativeIndex] + n.derivativeInputs[sampleIndex][inputDerivativeIndex]
			}
		}

		inputDerivatives[sampleIndex] = inputDerivativeForSample
	}

	return inputDerivatives
}

func (l *layer) backward(forwardInputDerivatives matrix) {
	var lastInputLen int = len(l.lastInput)
	var forwardInputDerivativesLen int = len(forwardInputDerivatives)

	if lastInputLen == 0 {
		panic("Layer has not previous input. Can not backpropigate")
	}

	if forwardInputDerivativesLen != lastInputLen {
		panic(fmt.Sprintf(
			"Forward derivatives length %d does not match previous inputs length %d. There should a row in the forward derivatives matrix for each input sample",
			forwardInputDerivativesLen, lastInputLen,
		))
	}

	for _, forwardDerivativeRow := range forwardInputDerivatives {
		var forwardDerivativeRowLen int = len(forwardDerivativeRow)
		if forwardDerivativeRowLen != l.layerSize {
			panic(fmt.Sprintf("The passed forward input derivative contains a row whose length %d does not match the layer size %d", forwardDerivativeRowLen, l.layerSize))
		}
	}

	for neuronIndex := range l.neurons {
		var n *neuron = &l.neurons[neuronIndex]

		n.derivativeWeights = make(vector, len(n.weights))
		for weightIndex := range n.derivativeWeights {
			for inputSampleIndex, inputSample := range l.lastInput {
				var forwardDerivativeForSample vector = forwardInputDerivatives[inputSampleIndex]
				var forwardDerivativeValueForSample float64 = forwardDerivativeForSample[neuronIndex]
				var inputValueForWeight float64 = inputSample[weightIndex]
				var derivativeWeightForSample float64 = inputValueForWeight * forwardDerivativeValueForSample

				derivativeWeightForSample /= float64(lastInputLen)
				n.derivativeWeights[weightIndex] = n.derivativeWeights[weightIndex] + derivativeWeightForSample
			}
		}

		n.derivativeInputs = make(matrix, lastInputLen)
		for inputSampleIndex := range l.lastInput {
			var sampleDerivativeInput vector = make(vector, len(n.weights))

			for derivativeIndex := range sampleDerivativeInput {
				var matchingWeightValue float64 = n.weights[derivativeIndex]
				var matchingForwardInputDerivative float64 = forwardInputDerivatives[inputSampleIndex][neuronIndex]
				sampleDerivativeInput[derivativeIndex] = matchingWeightValue * matchingForwardInputDerivative
			}

			n.derivativeInputs[inputSampleIndex] = sampleDerivativeInput
		}

		n.derivativeBias = 0
		for _, forwardDerivativeSample := range forwardInputDerivatives {
			n.derivativeBias += (1 * forwardDerivativeSample[neuronIndex])
		}
	}
}
