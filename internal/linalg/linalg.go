package linalg

import (
	"errors"
	"math"
)

const epsilon = 1e-6

// Uses margin of error = 1e-6
func equal(x, y float64) bool {
	dist := math.Abs(x - y)
	return dist <= epsilon
}

// Returns v*a
func ScalarMult(v Vector, a float64) Vector {
	newPoint := make([]float64, 0, v.Dim)
	for _, pi := range v.Point {
		newPoint = append(newPoint, pi*a)
	}
	return Vector{Dim: v.Dim, Point: newPoint}
}

// Returns v/a
func ScalarDiv(v Vector, a float64) (Vector, error) {
	if a == 0 {
		return Vector{}, errors.New("division error, cannot divide by 0")
	}
	return ScalarMult(v, 1/a), nil
}

// Returns zero vector for dim N
func ZeroVector(N int) Vector {
	point := make([]float64, N)
	return Vector{Dim: N, Point: point}
}

type Vector struct {
	Dim              int       // dimension of R Vector is in
	Point            []float64 // array representing vector in R^Dim space
	length           float64   // it is ||Point||
	calculatedLength bool
}

func (v Vector) IsZero() bool {
	for i := 0; i < v.Dim; i++ {
		if !equal(v.Point[i], 0) {
			return false
		}
	}
	return true
}

// Returns length of the vector following euclidean norm
func (v Vector) Len() float64 {
	if v.IsZero() {
		return 0
	} else if v.calculatedLength {
		return v.length
	}

	var squaredSum float64
	for _, vi := range v.Point {
		squaredSum += vi * vi
	}
	len := math.Sqrt(squaredSum)
	v.calculatedLength = true
	v.length = len
	return len
}
