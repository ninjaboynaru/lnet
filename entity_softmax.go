package main

import (
	"fmt"
	"math"
)

type softmax struct {
	lastOutput       matrix
	inputDerivatives matrix
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

func (s *softmax) forward(input matrix) matrix {
	var output matrix = make(matrix, len(input))

	for inputRowIndex, inputRow := range input {
		output[inputRowIndex] = s.singleInputForward(inputRow)
	}

	s.lastOutput = output
	return output
}

func (s softmax) getInputDerivatives() matrix {
	return s.inputDerivatives
}

func (s softmax) singleSampleBackward(forwardInputDerivativeRow vector, outputRow vector) vector {
	var derivativeRowLen int = len(forwardInputDerivativeRow)
	var outputRowLen int = len(outputRow)

	if derivativeRowLen != outputRowLen {
		panic(fmt.Sprintf(
			"The passed forward input derivative containes a row whose length %d does not match the length %d of its corresponding output row",
			derivativeRowLen, outputRowLen,
		))
	}

	var sampleInputDerivative vector = make(vector, outputRowLen)

	for currentValueIndex := range sampleInputDerivative {
		var currentValueOutput = outputRow[currentValueIndex]
		var currentValueDerivative vector = make(vector, outputRowLen)

		for outputIndex, outputValue := range outputRow {
			var derivativeValue float64

			if outputIndex == currentValueIndex {
				derivativeValue = outputValue * (1 - outputValue)
			} else {
				derivativeValue = (-1 * outputValue) * currentValueOutput
			}

			var matchingForwardDerivativeValue float64 = forwardInputDerivativeRow[outputIndex]
			derivativeValue *= matchingForwardDerivativeValue
			currentValueDerivative[outputIndex] = derivativeValue
		}

		var finalDerivativeValue float64 = vectorSum(currentValueDerivative)
		sampleInputDerivative[currentValueIndex] = finalDerivativeValue

	}

	return sampleInputDerivative

}

func (s *softmax) backward(forwardInputDerivatives matrix) {
	var lastOutputLen int = len(s.lastOutput)
	var forwardDerivativesLen int = len(forwardInputDerivatives)

	if lastOutputLen == 0 {
		panic("Softmax has no previous output. Can not back propigate")
	}

	if forwardDerivativesLen != lastOutputLen {
		panic(fmt.Sprintf(
			"Forward derivatives length %d does not match softmax last output length %d. There must be a row in the forward derivatives matrix for each output sample",
			forwardDerivativesLen, lastOutputLen,
		))
	}

	var inputDerivatives matrix = make(matrix, lastOutputLen)

	for sampleIndex := range inputDerivatives {
		var sampleOutput vector = s.lastOutput[sampleIndex]
		var sampleForwardDerivative vector = forwardInputDerivatives[sampleIndex]
		var sampleInputDerivative vector = s.singleSampleBackward(sampleForwardDerivative, sampleOutput)

		inputDerivatives[sampleIndex] = sampleInputDerivative
	}

	s.inputDerivatives = inputDerivatives
}
