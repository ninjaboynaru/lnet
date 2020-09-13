package main

import (
	"fmt"
	"math"
)

type crossentropy struct {
}

func (c crossentropy) forward(input matrix, targets []int) vector {
	var inputLen int = len(input)
	var targetsLen int = len(targets)

	if inputLen != targetsLen {
		panic(fmt.Sprintf(
			"Crossentropy targets length %d does not match input batch size %d. There must be one target value per row in the inputs batch matrix",
			inputLen, targetsLen,
		))
	}

	const safetyMargin float64 = 1e-7
	var output vector = make(vector, inputLen)

	for index, inputRow := range input {
		var targetIndex int = targets[index]
		var rowLength int = len(inputRow)

		if targetIndex <= -1 || targetIndex >= rowLength {
			panic(fmt.Sprintf("A crossentropy target index %d is out of bounds of its corresponding input row length %d", targetIndex, rowLength))
		}

		var targetValue float64 = inputRow[targetIndex]
		var loss float64 = clip(safetyMargin, 1-safetyMargin, targetValue)
		loss = -1 * math.Log(loss)

		output[index] = loss
	}

	return output

}
