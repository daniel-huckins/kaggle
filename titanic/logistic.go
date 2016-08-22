package main

import (
	"encoding/csv"
	"os"
	"strconv"

	"github.com/Sirupsen/logrus"
	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/linear"
)

var log = logrus.New()

func main() {
	X, y := LoadRows("./data/train.csv")
	model := linear.NewLogistic(base.BatchGA, 1e-4, 6, 800, X, y)
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
		prediction := model.Predict(row.Row())[0]
		writer.Write([]string{
			strconv.Itoa(row.ID),
			strconv.FormatFloat(prediction, nil, -1, 64),
		})
	}
}
