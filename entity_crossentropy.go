package main

import (
	"fmt"
	"math"
)

type crossentropy struct {
	lastInput        matrix
	lastTargets      []int
	lastOutput       vector
	inputDerivatives matrix
}

func (c *crossentropy) forward(input matrix, targets []int) vector {
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

	c.lastInput = input
	c.lastTargets = targets
	c.lastOutput = output
	return output
}

func (c crossentropy) getInputDerivatives() matrix {
	return c.inputDerivatives
}

func (c *crossentropy) backward() {
	var lastInputLen int = len(c.lastInput)
	var lastTargetsLen int = len(c.lastTargets)

	if lastInputLen == 0 {
		panic("Crossentropy has no previous input. Can not back propigate")
	}

	if lastTargetsLen == 0 {
		panic("Crossentropy has no previous targets. Can not back propigate")
	}

	var inputDerivatives matrix = make(matrix, lastInputLen)
	for inputDerivativeIndex := range inputDerivatives {
		var inputRow vector = c.lastInput[inputDerivativeIndex]
		var derivativeRow vector = make(vector, len(inputRow))
		var targetIndex int = c.lastTargets[inputDerivativeIndex]

		for derivativeRowIndex := range derivativeRow {
			if targetIndex == derivativeRowIndex {
				derivativeRow[derivativeRowIndex] = -1 / inputRow[derivativeRowIndex]
			} else {
				derivativeRow[derivativeRowIndex] = 0
			}
		}

		inputDerivatives[inputDerivativeIndex] = derivativeRow
	}

	c.inputDerivatives = inputDerivatives
}

func (c crossentropy) calculateAverageLoss() float64 {
	if len(c.lastOutput) == 0 {
		panic("Crossentropy has not previous output. Can not calculate average loss")
	}

	var averageLoss float64 = 0
	for _, sampleLossValue := range c.lastOutput {
		averageLoss += sampleLossValue
	}

	averageLoss = averageLoss / float64(len(c.lastOutput))

	return averageLoss
}
