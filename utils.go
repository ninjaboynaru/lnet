package main

import "math/rand"

type vector []float64
type matrix []vector

func randRangeFloat64(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func clip(min, max, value float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}

	return value
}
