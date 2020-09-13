package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSoftmaxForward(t *testing.T) {
	var input matrix = matrix{
		{2, 5, 6},
		{4, 4, 6},
	}

	var expectedOutput matrix = matrix{
		{0.013212886953789417, 0.265387928772242, 0.7213991842739688},
		{0.10650697891920076, 0.10650697891920076, 0.7869860421615986},
	}

	var s softmax
	var actualOutput matrix = s.forward(input)

	assert.Equal(t, actualOutput, expectedOutput, "Softmax forward returns wrong value")
}

func TestSoftmaxBackwardPanics(t *testing.T) {
	var assert *assert.Assertions = assert.New(t)
	var s softmax
	var mockForwardInputDerivatives matrix
	var inputs matrix
	var doPanic func() = func() { s.backward(mockForwardInputDerivatives) }

	s = softmax{}
	mockForwardInputDerivatives = matrix{{1, 1, 1}, {1, 1, 1}}
	assert.Panics(doPanic, "Should panic on back propigate when forward has not yet ben called")

	s = softmax{}
	inputs = matrix{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}
	mockForwardInputDerivatives = matrix{{1, 1, 1}, {1, 1, 1}}
	s.forward(inputs)
	assert.Panics(doPanic, "Should panic on back propigate when forward derivatives length does not match previous output/input length")

	s = softmax{}
	inputs = matrix{{1, 1, 1}, {1, 1, 1}}
	mockForwardInputDerivatives = matrix{{1, 1}, {1, 1, 1, 1, 1}}
	s.forward(inputs)
	assert.Panics(doPanic, "Should panic on back propigate when forward derivatives row length does not match previous output/input row length")
}

func TestSoftmaxBackwardDerivativeInput(t *testing.T) {
	var s softmax = softmax{}
	var inputs matrix = matrix{
		{6, 2, 2},
		{4, 3, 2},
	}
	var mockForwardInputDerivatives matrix = matrix{
		{1, 0, 0},
		{1, 2, 0},
	}

	s.forward(inputs)
	s.backward(mockForwardInputDerivatives)

	var expectedInputDerivatives matrix = matrix{
		{0.034088151482230225, -0.017044075741115054, -0.017044075741115054},
		{-0.10291137744498538, 0.20686949103015304, -0.1039581135851675},
	}

	var actualInputDerivatives matrix = s.getInputDerivatives()

	assert.Equal(t, expectedInputDerivatives, actualInputDerivatives, "Softmax backwards produces wrong input derivatives")

}
