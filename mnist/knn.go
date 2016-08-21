package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"

	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/cluster"
)

func trainingSet(filename string) (y []float64, x [][]float64) {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	reader := csv.NewReader(file)
	first := true
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		if first {
			first = false
			continue
		}
		isLabel := true
		var row []float64
		for _, td := range record {
			if isLabel {
				i, err := strconv.ParseFloat(td, 32)
				if err != nil {
					log.Fatal(err)
				}
				y = append(y, i)
				isLabel = false
			} else {
				f, err := strconv.ParseFloat(td, 32)
				if err != nil {
					log.Fatal(err)
				}
				row = append(row, f)
			}
		}
		x = append(x, row)
	}
	return
}

func testSet(filename string) (data [][]float64) {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	reader := csv.NewReader(file)
	first := true
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		if first {
			first = false
			continue
		}
		var row []float64
		for _, td := range record {
			f, err := strconv.ParseFloat(td, 32)
			if err != nil {
				log.Fatal(err)
			}
			row = append(row, f)
		}
		data = append(data, row)
	}
	return
}

func main() {
	y, X := trainingSet("./data/train.csv")
	log.Println("got training data, creating knn")
	knn := cluster.NewKNN(10, X, y, base.EuclideanDistance)
	test := testSet("./data/test.csv")
	file, err := os.Create("go_results.csv")
	if err != nil {
		log.Fatal(err)
	}
	writer := csv.NewWriter(file)
	writer.Write([]string{"ImageId", "Label"})
	count := len(test)
	for i, t := range test {
		res, err := knn.Predict(t, false)
		if err != nil {
			log.Fatal(err)
		}
		label := fmt.Sprintf("%.0f", res[0])
		id := strconv.Itoa(i + 1)
		writer.Write([]string{id, label})
		if i%10 == 0 {
			log.Printf("%d/%d\n", i, count)
		}
	}
	writer.Flush()
}
