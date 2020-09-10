package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCrossentropyForwardPanics(t *testing.T) {
	var input matrix
	var targets []int
	var assert *assert.Assertions = assert.New(t)

	var tryForward func() = func() {
		crossentropy{}.forward(input, targets)
	}

	input = matrix{{1, 2}, {1, 2}}
	targets = []int{1}
	assert.Panics(tryForward, "Should panic with mismatch between input and targets length")

	targets = []int{0, 2}
	assert.Panics(tryForward, "Should panic with targets value out of bounds of corresponding input row")

	targets = []int{0, -1}
	assert.Panics(tryForward, "Should panic with targets value out of bounds of corresponding input row")

}

func TestCrossentropyForward(t *testing.T) {
	var input matrix = matrix{
		{0.1, 0.5, 0.4},
		{0.2, 0.3, 0.6},
		{0.03, 0.4985, 0.4985},
	}

	var targets []int = []int{1, 2, 0}

	var expectedOutput vector = vector{
		0.6931471805599453,
		0.5108256237659907,
		3.506557897319982,
	}

	var actualOutput = crossentropy{}.forward(input, targets)

	assert.Equal(t, actualOutput, expectedOutput, "Crossentropy forward returns wrong value")
}
