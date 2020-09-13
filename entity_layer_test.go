package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewLayerPanics(t *testing.T) {
	var assert *assert.Assertions = assert.New(t)

	assert.Panics(func() { newLayer(-1, 1) }, "Should panic with negative layer size")
	assert.Panics(func() { newLayer(0, 1) }, "Should panic with layer size 0")

	assert.Panics(func() { newLayer(1, -1) }, "Should panic with negative input count")
	assert.Panics(func() { newLayer(1, 0) }, "Should panic with input count 0")

	var weights matrix
	var biases vector
	var tryNewLayerExplicit func() = func() {
		newLayerExplicit(weights, biases)
	}

	weights = matrix{}
	biases = vector{1.0, 2.0}
	assert.Panics(tryNewLayerExplicit, "Should panic with 0 neurons")

	weights = matrix{{1.0}, {2.0}}
	biases = vector{}
	assert.Panics(tryNewLayerExplicit, "Should panic with 0 biases")

	weights = matrix{}
	biases = vector{}
	assert.Panics(tryNewLayerExplicit, "Should panic with both 0 neurons and biases")

	weights = matrix{{1.0}, {2.0}}
	biases = vector{1.0}
	assert.Panics(tryNewLayerExplicit, "Should panic with mismatch between neuron and bias counts")

	weights = matrix{{1.0}, {2.0}}
	biases = vector{1.0}
	assert.Panics(tryNewLayerExplicit, "Should panic with mismatch between neuron and bias counts")

	weights = matrix{
		{1.0, 2.0, 3.0},
		{1.1, 2.2, 5.4},
		{1.2},
	}
	biases = vector{1.0, 1.0, 1.0}
	assert.Panics(tryNewLayerExplicit, "Should panic with neurons that have different input counts")
}

func TestNewLayerLengths(t *testing.T) {
	const layerSize int = 2
	const inputCount int = 3

	var require *require.Assertions = require.New(t)
	var layer layer = newLayer(layerSize, inputCount)

	require.Len(layer.neurons, layerSize, "Layer has incorrect amount of neurons")

	for _, n := range layer.neurons {
		require.Len(n.weights, inputCount, "Neuron in layer has incorrect ammount of weights for given input size")
	}

	require.Equal(layerSize, layer.layerSize, "Incorrect layerSize value")
	require.Equal(inputCount, layer.inputCount, "Incorrect inputCount value")
}

func TestNewLayerExplicitLengths(t *testing.T) {
	const neuronCount = 2
	const inputCount = 3

	var weights matrix = matrix{
		{1, 2, 3},
		{4, 5, 6},
	}
	var biases vector = vector{
		1,
		1,
	}

	var require *require.Assertions = require.New(t)
	var layer layer = newLayerExplicit(weights, biases)

	require.Equal(neuronCount, layer.layerSize, "Incorrect layerSize value")
	require.Equal(inputCount, layer.inputCount, "Incorrect inputCount value")
	require.Len(layer.neurons, neuronCount, "Layer has incorrect amount of neurons")

	for index, n := range layer.neurons {
		require.Equal(weights[index], n.weights, "Neuron weights incorrect")
		require.Equal(biases[index], n.bias, "Neuron bias incorrect")
	}

}

func TestLayerForward(t *testing.T) {
	var inputs matrix
	var expectedOutput matrix
	var actualOutput matrix

	var assert *assert.Assertions = assert.New(t)
	var biases vector = vector{1, 2, 4}
	var weights matrix = matrix{
		{2, 2, 4},
		{6, 4, 8},
		{12, 1, 1},
	}
	var l layer = newLayerExplicit(weights, biases)

	inputs = matrix{{1, 3, 2}}
	expectedOutput = matrix{{17, 36, 21}}
	actualOutput = l.forward(inputs)
	assert.Equal(expectedOutput, actualOutput, "Layer forward returns wrong output for single input row")

	inputs = matrix{
		{2, 2, 2},
		{1, 3, 2},
	}

	expectedOutput = matrix{
		{17, 38, 32},
		{17, 36, 21},
	}

	actualOutput = l.forward(inputs)
	assert.Equal(expectedOutput, actualOutput, "Layer forward returns wrong output for multiple input rows")
}

// func TestLayerBackwardPanics(t *testing.T) {
// 	var l layer
// 	var doPanicFunc func()
// 	var mockForwardInputDerivative vector

// 	l = newLayer(3, 3)
// 	mockForwardInputDerivative = vector{1, 1, 1}
// 	doPanicFunc = func() {
// 		l.backward(mockForwardInputDerivative)
// 	}
// 	assert.Panics(t, doPanicFunc, "Should panic on back propigate with no previous input")

// 	l = newLayer(3, 3)
// 	var input matrix = matrix{{1, 1, 1}, {1, 1, 1}}
// 	mockForwardInputDerivative = vector{1, 1, 1, 1, 1, 1, 1}
// 	l.forward(input)
// 	doPanicFunc = func() {
// 		l.backward(mockForwardInputDerivative)
// 	}
// 	assert.Panics(t, doPanicFunc, "Should panic on back propigate with wrong size forward derivative argument")
// }

// func TestLayerBackward(t *testing.T) {
// 	var assert *assert.Assertions = assert.New(t)
// 	var biases vector = vector{1, 2, 4}
// 	var weights matrix = matrix{
// 		{2, 2, 4},
// 		{6, 4, 8},
// 		{12, 1, 1},
// 	}
// 	var l layer = newLayerExplicit(weights, biases)

// 	var inputs matrix = matrix{
// 		{2, 2, 2},
// 		{1, 3, 2},
// 	}

// 	var mockForwardInputDerivative = vector{1, 1, 1}

// 	l.forward(inputs)
// 	l.backward(mockForwardInputDerivative)

// 	var expectedInputDerivative vector = vector{20, 7, 13}
// 	var actualInputDerivative vector = l.getLayerInputDerivative()

// 	assert.Equal(expectedInputDerivative, actualInputDerivative, "Layer backwards produces wrong input derivatives for multiple input rows")

// }
