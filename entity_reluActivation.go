package main

import (
	"fmt"
	"math"
)

type reluActivation struct {
	lastInput        matrix
	inputDerivatives matrix
}

func (r *reluActivation) forward(input matrix) matrix {
	var output matrix = make(matrix, len(input))

	for inputRowIndex, inputRow := range input {
		output[inputRowIndex] = make(vector, len(inputRow))

		for inputValueIndex, inputValue := range inputRow {
			output[inputRowIndex][inputValueIndex] = math.Max(0, inputValue)
		}
	}

	r.lastInput = input
	return output
}

func (r reluActivation) getInputDerivatives() matrix {
	return r.inputDerivatives
}

func (r *reluActivation) backward(forwardInputDerivatives matrix) {
	var lastInputLen int = len(r.lastInput)
	var forwardDerivativesLen int = len(forwardInputDerivatives)

	if lastInputLen == 0 {
		panic("RELU Activation has not previous input. Can not back propigate")
	}

	if lastInputLen != forwardDerivativesLen {
		panic(fmt.Sprintf(
			"Forward derivatives length %d does not match previous input length %d. There must be a row in the forward derivatives matrix for each input sample in the previous input",
			forwardDerivativesLen, lastInputLen,
		))
	}

	for forwardDerivativeRowIndex := range forwardInputDerivatives {
		var derivativeRowLen int = len(forwardInputDerivatives[forwardDerivativeRowIndex])
		var inputRowLen int = len(r.lastInput[forwardDerivativeRowIndex])

		if derivativeRowLen != inputRowLen {
			panic(fmt.Sprintf(
				"The passed forward input derivative containes a row whose length %d does not match the length %d of its corresponding input row",
				derivativeRowLen, inputRowLen,
			))
		}
	}

	var inputDerivatives matrix = make(matrix, forwardDerivativesLen)

	for rowIndex := range inputDerivatives {
		var inputRow vector = r.lastInput[rowIndex]
		var forwardDerivativeRow vector = forwardInputDerivatives[rowIndex]
		var inputDerivativeRow vector = make(vector, len(inputRow))

		for valueIndex := range inputDerivativeRow {
			if inputRow[valueIndex] <= 0 {
				inputDerivativeRow[valueIndex] = 0
			} else {
				inputDerivativeRow[valueIndex] = forwardDerivativeRow[valueIndex]
			}
		}

		inputDerivatives[rowIndex] = inputDerivativeRow
	}

	r.inputDerivatives = inputDerivatives
}
