package sampling

import (
	"math"
	"math/rand/v2"
	"monte-carlo/internal/linalg"
)

const Pi = math.Pi

// Returns uniformly sampled float in the half open interval [a,b)
func Uniform(a, b float64) float64 {
	// use bijection f(x) = a + x(b-a)
	u := rand.Float64()
	dist := math.Abs(b - a)
	y := a + u*(dist)
	return y
}

// Returns point sampled from Normal distribution with
// mean = m and standard deviation = stdd
func Gaussian(m, stdd float64) float64 {
	// Uses Box-Muller transformation
	u := rand.Float64()
	v := rand.Float64()
	sample := stdd*math.Sqrt(-2.0*math.Log(u))*math.Cos(v*2*Pi) + m
	return sample
}

// Returns uniformly sampled vector along the circumference of the N-Sphere
// where an N-Sphere is defined by the set
//
//	{v \in R^N | |v| = 1}
func UniformFromNSphere(N int) linalg.Vector {
	sample := make([]float64, 0, N)
	for i := 0; i < N; i++ {
		u := Gaussian(0, 1)
		sample = append(sample, u)
	}
	vector := linalg.Vector{Dim: N, Point: sample}
	vector, err := linalg.ScalarDiv(vector, vector.Len())
	if err != nil {
		// only here if vector was zero vector so return zero vector
		return linalg.ZeroVector(N)
	}
	return vector
}

// Returns uniformly sampled float from within the N-Ball
// where an N-Ball is defined by the set
//
//	{v \in R^N | |v| < 1}
func UniformFromNBall(N int) linalg.Vector {
	sample := UniformFromNSphere(N)
	d := rand.Float64()
	return linalg.ScalarMult(sample, math.Pow(d, 1.0/float64(N)))
}

// Returns point sampled from normal distribution
// with mean = m and standard deviation = stdd in R^N
func GaussianN(m, stdd float64, N int) linalg.Vector {
	sample := make([]float64, 0, N)
	for i := 0; i < N; i++ {
		sample = append(sample, Gaussian(m, stdd))
	}
	sampleVector := linalg.Vector{Dim: N, Point: sample}
	return sampleVector
}
