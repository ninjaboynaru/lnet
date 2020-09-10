package main

import "math"

type softmax struct {
}

func (s softmax) singleInputForward(input vector) vector {
	var output vector = make(vector, len(input))
	var exponentialSum float64 = 0

	for index, value := range input {
		var exponentialValue float64 = math.Exp(value)
		exponentialSum += exponentialValue
		output[index] = exponentialValue
	}

	for index, value := range output {
		output[index] = value / exponentialSum
	}

	return output
}

func (s softmax) forward(input matrix) matrix {
	var output matrix = make(matrix, len(input))

	for inputRowIndex, inputRow := range input {
		output[inputRowIndex] = s.singleInputForward(inputRow)
	}

	return output
}
