package main

import (
	"os"

	"github.com/cdipaolo/goml/base"
	"github.com/gocarina/gocsv"
)

var (
	genders      = map[string]float64{"male": 1.0, "female": 0.0}
	econStatuses = map[int][]float64{
		1: []float64{1.0, 0.0},
		2: []float64{0.0, 1.0},
		3: []float64{0.0, 0.0},
	}
	// some do not have ports
	embarkPorts = map[string][]float64{
		"C": []float64{1.0, 0.0, 0.0},
		"Q": []float64{0.0, 1.0, 0.0},
		"S": []float64{0.0, 0.0, 1.0},
		"":  []float64{0.0, 0.0, 0.0},
	}
)

// Passenger container for someone on the titanic
type Passenger struct {
	ID       int     `csv:"PassengerId"`
	Survived float64 `csv:"Survived"`
	Sex      string  `csv:"Sex"`
	PClass   int     `csv:"Pclass"`
	Age      float64 `csv:"Age"`
	NumSibs  float64 `csv:"SibSp"`
	NumParch float64 `csv:"Parch"`
	Embarked string  `csv:"Embarked"`
}

// Row takes a passenger to a row of floats
func (p *Passenger) Row() (row []float64) {
	logger := log.WithField("ID", p.ID)
	row = append(row, p.Age, p.NumSibs, p.NumParch)
	gender, ok := genders[p.Sex]
	if !ok {
		logger.Fatalf("'%s' is not a valid gender", p.Sex)
	}
	row = append(row, gender)
	status, ok := econStatuses[p.PClass]
	if !ok {
		logger.Fatalf("'%d' is not a valid pclass", p.PClass)
	}
	row = append(row, status[0], status[1])
	embarked, ok := embarkPorts[p.Embarked]
	if !ok {
		logger.Fatalf("'%s' is not a valid port", p.Embarked)
	}
	row = append(row, embarked[0], embarked[1])
	return
}

// Datapoint returns the datapoint
func (p *Passenger) Datapoint() base.Datapoint {
	return base.Datapoint{
		X: p.Row(),
		Y: []float64{p.Survived},
	}
}

// LoadPassengers returns passenger records from a file
func LoadPassengers(filename string) (records []*Passenger) {
	file, err := os.Open(filename)
	if err != nil {
		log.WithError(err).Fatalf("error opening %s", filename)
	}
	err = gocsv.Unmarshal(file, &records)
	if err != nil {
		log.WithError(err).Fatal("error Unmarshalling data")
	}
	return
}

// LoadRows loads the titanic data
func LoadRows(filename string) (data [][]float64, expected []float64) {
	passengers := LoadPassengers(filename)
	for _, p := range passengers {
		row := p.Row()
		data = append(data, row)
		expected = append(expected, p.Survived)
	}
	return
}
