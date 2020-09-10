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

	var actualOutput matrix = softmax{}.forward(input)

	assert.Equal(t, actualOutput, expectedOutput, "Softmax forward returns wrong value")
}
