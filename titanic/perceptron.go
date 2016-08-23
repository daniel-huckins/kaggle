package main

import (
	"encoding/csv"
	"fmt"
	"os"

	"github.com/Sirupsen/logrus"
	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/perceptron"
)

var log = logrus.New()

func main() {
	passengers := LoadPassengers("./data/train.csv")
	stream := make(chan base.Datapoint, len(passengers))
	errors := make(chan error)
	model := perceptron.NewPerceptron(1e-4, 8)

	go model.OnlineLearn(errors, stream, func(theta [][]float64) {
		// log.WithField("theta", theta).Info("updated theta")
	})

	for _, p := range passengers {
		stream <- p.Datapoint()
	}

	close(stream)
	// catch errors
	for {
		err := <-errors
		if err != nil {
			log.WithError(err).Fatal("error from errors")
		} else {
			break
		}
	}

	file, err := os.Create("./data/go_perceptron.csv")
	if err != nil {
		log.WithError(err).Fatal("error opening file")
	}
	writer := csv.NewWriter(file)
	writer.Write([]string{"PassengerId", "Survived"})
	tests := LoadPassengers("./data/test.csv")
	for _, t := range tests {
		res, err := model.Predict(t.Row())
		if err != nil {
			log.WithError(err).Fatal("error predicting")
		}
		// perceptron gives 1 or -1
		prediction := res[0]
		if prediction < 0 {
			prediction = 0.0
		}
		writer.Write([]string{
			fmt.Sprintf("%d", t.ID),
			fmt.Sprintf("%.f", prediction),
		})
	}

	writer.Flush()
}
