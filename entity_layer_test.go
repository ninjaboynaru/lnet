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

	var weightsMatrix matrix
	var biasesVector vector
	var tryNewLayerExplicit func() = func() {
		newLayerExplicit(weightsMatrix, biasesVector)
	}

	weightsMatrix = matrix{}
	biasesVector = vector{1.0, 2.0}
	assert.Panics(tryNewLayerExplicit, "Should panic with 0 neurons")

	weightsMatrix = matrix{{1.0}, {2.0}}
	biasesVector = vector{}
	assert.Panics(tryNewLayerExplicit, "Should panic with 0 biases")

	weightsMatrix = matrix{}
	biasesVector = vector{}
	assert.Panics(tryNewLayerExplicit, "Should panic with both 0 neurons and biases")

	weightsMatrix = matrix{{1.0}, {2.0}}
	biasesVector = vector{1.0}
	assert.Panics(tryNewLayerExplicit, "Should panic with mismatch between neuron and bias counts")

	weightsMatrix = matrix{{1.0}, {2.0}}
	biasesVector = vector{1.0}
	assert.Panics(tryNewLayerExplicit, "Should panic with mismatch between neuron and bias counts")

	weightsMatrix = matrix{
		{1.0, 2.0, 3.0},
		{1.1, 2.2, 5.4},
		{1.2},
	}
	biasesVector = vector{1.0, 1.0, 1.0}
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

	require.Equal(layer.layerSize, layerSize, "Incorrect layerSize value")
	require.Equal(layer.inputCount, inputCount, "Incorrect inputCount value")
}

func TestNewLayerExplicitLengths(t *testing.T) {
	const neuronCount = 2
	const inputCount = 3

	var weightsMatrix matrix = matrix{
		{1, 2, 3},
		{4, 5, 6},
	}
	var biasesVector vector = vector{
		1,
		1,
	}

	var require *require.Assertions = require.New(t)
	var layer layer = newLayerExplicit(weightsMatrix, biasesVector)

	require.Equal(layer.layerSize, neuronCount, "Incorrect layerSize value")
	require.Equal(layer.inputCount, inputCount, "Incorrect inputCount value")
	require.Len(layer.neurons, neuronCount, "Layer has incorrect amount of neurons")

	for index, n := range layer.neurons {
		require.Equal(n.weights, weightsMatrix[index], "Neuron weights incorrect")
		require.Equal(n.bias, biasesVector[index], "Neuron bias incorrect")
	}

}

func TestLayerForward(t *testing.T) {
	var expectedOutput matrix = matrix{
		{17, 38, 32},
		{17, 36, 21},
	}

	var weightsMatrix matrix = matrix{
		{2, 2, 4},
		{6, 4, 8},
		{12, 1, 1},
	}

	var biases vector = vector{
		1,
		2,
		4,
	}

	var inputs = matrix{
		{2, 2, 2},
		{1, 3, 2},
	}

	var l layer = newLayerExplicit(weightsMatrix, biases)
	var actualOutput matrix = l.forward(inputs)

	assert.Equal(t, actualOutput, expectedOutput, "Layer forward returns wrong value")

}
