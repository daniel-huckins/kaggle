package main

import (
	"encoding/csv"
	"fmt"
	"os"

	"github.com/Sirupsen/logrus"
	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/linear"
)

var log = logrus.New()

func main() {
	X, y := LoadRows("./data/train.csv")
	model := linear.NewLogistic(base.BatchGA, 1e-4, 0, 800, X, y)
	err := model.Learn()
	if err != nil {
		log.WithError(err).Fatal("error learning")
	}
	file, err := os.Create("./data/go_logistic_results.csv")
	if err != nil {
		log.WithError(err).Fatal("error creating file")
	}
	writer := csv.NewWriter(file)
	writer.Write([]string{"PassengerId", "Survived"})
	tests := LoadPassengers("./data/test.csv")
	for _, row := range tests {
		predictions, err := model.Predict(row.Row())
		if err != nil {
			log.WithError(err).Fatal("error making prediction")
		}
		writer.Write([]string{
			fmt.Sprintf("%d", row.ID),
			fmt.Sprintf("%.f", predictions[0]),
		})
	}
	writer.Flush()
}
