package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestReluForwad(t *testing.T) {
	var input matrix = matrix{
		{12, -5, 5},
		{-2, -2, 0.012},
	}

	var expectedOutput matrix = matrix{
		{12, 0, 5},
		{0, 0, 0.012},
	}

	var r reluActivation = reluActivation{}
	var actualOutput matrix = r.forward(input)

	assert.Equal(t, actualOutput, expectedOutput, "Relu forward returns wrong value")
}

func TestReluBackwardPanics(t *testing.T) {
	var assert *assert.Assertions = assert.New(t)
	var r reluActivation
	var input matrix
	var mockForwardInputDerivatives matrix

	var doPanic func() = func() {
		r.backward(mockForwardInputDerivatives)
	}

	r = reluActivation{}
	assert.Panics(doPanic, "Should panic on back propigation with no previous input")

	input = matrix{{1, 1, 1}, {1, 1, 1}}
	mockForwardInputDerivatives = matrix{{1, 1, 1}}
	r = reluActivation{}
	r.forward(input)
	assert.Panics(doPanic, "Should panic on back propigation when forward input derivatives length does not match previous input length")

	input = matrix{{1, 1, 1}, {1, 1, 1}}
	mockForwardInputDerivatives = matrix{{1, 1, 1}, {1}}
	r = reluActivation{}
	r.forward(input)
	assert.Panics(doPanic, "Shoudl panic on back propigation when forward input derivatives contains rows whose length do not match previous input row lengths")
}

func TestReluBackwardDerivativeInput(t *testing.T) {
	var input matrix = matrix{
		{1, 1, 1},
		{-1, 1, -1},
	}

	var mockForwardInputDerivatives matrix = matrix{
		{2, 2, 2},
		{2, 2, 2},
	}

	var r reluActivation = reluActivation{}
	r.forward(input)
	r.backward(mockForwardInputDerivatives)

	var expectedInputDerivatives matrix = matrix{
		{2, 2, 2},
		{0, 2, 0},
	}
	var actualInputDerivatives = r.getInputDerivatives()

	assert.Equal(t, expectedInputDerivatives, actualInputDerivatives, "RELU Activation back propigate produces wrong input derivatives")
}
