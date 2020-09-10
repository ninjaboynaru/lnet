package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewNeuronPanics(t *testing.T) {
	var assert *assert.Assertions = assert.New(t)

	assert.Panics(func() { newNeuron(-1) }, "Should panic with negative input count")
	assert.Panics(func() { newNeuron(-1) }, "Should panic with input count 0")
}

func TestNewNeuronWeightLength(t *testing.T) {
	const inputCount int = 3
	var neuron neuron = newNeuron(inputCount)

	require.Len(t, neuron.weights, inputCount, "Neuron has incorrect amount of weights for given input size")
}
