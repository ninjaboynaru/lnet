package main

import "math"

type reluActivation struct {
}

func (r reluActivation) forward(input matrix) matrix {
	var output matrix = make(matrix, len(input))

	for inputRowIndex, inputRow := range input {
		output[inputRowIndex] = make(vector, len(inputRow))

		for inputValueIndex, inputValue := range inputRow {
			output[inputRowIndex][inputValueIndex] = math.Max(0, inputValue)
		}
	}

	return output
}
