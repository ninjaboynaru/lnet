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

	var actualOutput matrix = reluActivation{}.forward(input)

	assert.Equal(t, actualOutput, expectedOutput, "Relu forward returns wrong value")
}
