package sampling

import (
	"testing"
	"math/rand/v2"
	"math"

	"github.com/stretchr/testify/assert"
)

const delta = 1e-2

func TestUniformFromInterval(t *testing.T) {
	for i := 0; i < 1000; i++ {
		var a = rand.Float64()
		var b = rand.Float64()
		var sample float64
		if a > b {
			sample = UniformFromInterval(b,a)
			assert.GreaterOrEqual(t, sample, b)
			assert.LessOrEqual(t, sample, a)
		} else {
			sample = UniformFromInterval(a,b)
			assert.GreaterOrEqual(t, sample, a)
			assert.LessOrEqual(t, sample, b)
		}
	}
}

func TestNormalSample(t *testing.T) {
	N := 10000
	m, stdd := 0.0, 1.0
	samples := make([]float64, 0, N)
	for i := 0; i < N; i++ {
		sample := NormalSample(m, stdd)
		samples = append(samples, sample)
	}

	sampleVariance := sampleVariance(samples)
	sampleStdd := math.Sqrt(sampleVariance)
	assert.InDelta(t, stdd, sampleStdd, delta)
	assert.InDelta(t, stdd*stdd, sampleVariance, delta)
}

/***********************
	Helper functions
***********************/

func sampleVariance(samples []float64) float64 {
	mean := sampleMean(samples)
	var squaredDifferenceSum float64
	for _, sample := range samples {
		squaredDifferenceSum += ((sample - mean)*(sample - mean))
	}
	variance := squaredDifferenceSum/float64((len(samples)))
	return variance
}

func sampleMean(samples []float64) float64 {
	var totalSum float64
	for _, sample := range samples {
		totalSum += sample
	}
	mean := totalSum/float64(len(samples))
	return mean
}