package main

import (
	"fmt"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())

	var inputs, targets = extractIrisSmall()

	var l1 layer = newLayer(10, 4)
	var relu1 reluActivation = reluActivation{}

	var l2 layer = newLayer(3, 10)
	var relu2 reluActivation = reluActivation{}

	var softmaxActivation softmax = softmax{}
	var crossentropyLoss crossentropy = crossentropy{}

	const epochs int = 10000
	const learningRateStart float64 = 1
	const learningRateDecay float64 = 0.00000001
	const logRate int = 100

	var learningRate float64 = learningRateStart
	var currentEpoch int = 0

	var forward func() = func() {
		var l1Output matrix = l1.forward(inputs)
		var relu1Output matrix = relu1.forward(l1Output)

		var l2Output matrix = l2.forward(relu1Output)
		var relu2Output matrix = relu2.forward(l2Output)

		var softmaxOutput matrix = softmaxActivation.forward(relu2Output)
		crossentropyLoss.forward(softmaxOutput, targets)
	}

	var backward func() = func() {
		crossentropyLoss.backward()
		var crossentropyInputDerivatives matrix = crossentropyLoss.getInputDerivatives()

		softmaxActivation.backward(crossentropyInputDerivatives)
		var softmaxInputDerivatives matrix = softmaxActivation.getInputDerivatives()

		relu2.backward(softmaxInputDerivatives)
		var relu2InputDerivatives matrix = relu2.getInputDerivatives()

		l2.backward(relu2InputDerivatives)
		var l2InputDerivatives matrix = l2.getInputDerivatives()

		relu1.backward(l2InputDerivatives)
		var relu1InputDerivatives matrix = relu1.getInputDerivatives()

		l1.backward(relu1InputDerivatives)
	}

	var optimizeLayer func(*layer) = func(l *layer) {
		for index := range l.neurons {
			var n *neuron = &l.neurons[index]

			for weightIndex, derivativeValue := range n.derivativeWeights {
				n.weights[weightIndex] = n.weights[weightIndex] + (-1 * derivativeValue * learningRate)
			}

			n.bias = n.bias + (-1 * n.derivativeBias * learningRate)
		}
	}

	var optimize func() = func() {
		optimizeLayer(&l1)
		optimizeLayer(&l2)
	}

	var updateLearningRate func() = func() {
		learningRate = learningRateStart * (1 / (1 + learningRateDecay*float64(currentEpoch)))
	}

	for i := 0; i < epochs; i++ {
		currentEpoch = i + 1
		updateLearningRate()

		forward()
		backward()
		optimize()

		if i%logRate == 0 {
			var averageLoss float64 = crossentropyLoss.calculateAverageLoss()
			fmt.Printf("Learning Rate: %f\nEpoch %d Average Loss: %f\n\n", learningRate, currentEpoch, averageLoss)
		}
	}
}
