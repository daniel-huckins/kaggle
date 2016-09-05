package main

import (
	"encoding/csv"
	"fmt"
	"image"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"strings"

	"github.com/anthonynsimon/bild/effect"
	"github.com/anthonynsimon/bild/transform"
	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/cluster"
)

const (
	trainingDir = "./data/train"
	testDir     = "./data/test"
	width       = 64
	height      = 64
)

func extractImage(imagefile string) *image.Gray {
	file, err := os.Open(imagefile)
	if err != nil {
		log.Fatal(err)
	}
	img, err := jpeg.Decode(file)
	if err != nil {
		log.Fatal(err)
	}
	resized := transform.Resize(img, width, height, transform.NearestNeighbor)
	return effect.Grayscale(resized)
}

func f(ints []uint8) (floats []float64) {
	for _, i := range ints {
		floats = append(floats, float64(i))
	}
	base.NormalizePoint(floats)
	return
}

func trainedKNN() *cluster.KNN {
	files, err := ioutil.ReadDir(trainingDir)
	if err != nil {
		log.Fatal(err)
	}
	dog, cat := float64(1.0), float64(0.0)

	data := [][]float64{}
	Y := []float64{}

	for _, file := range files {
		img := extractImage(fmt.Sprintf("%s/%s", trainingDir, file.Name()))
		name := strings.Split(file.Name(), ".")[0]
		x := f(img.Pix)
		data = append(data, x)
		var y float64
		if name == "dog" {
			y = dog
		} else {
			y = cat
		}
		Y = append(Y, y)
	}

	knn := cluster.NewKNN(2, data, Y, base.EuclideanDistance)
	return knn
}

func main() {
	files, err := ioutil.ReadDir(testDir)
	if err != nil {
		log.Fatal(err)
	}

	knn := trainedKNN()

	output, err := os.Create("go_results.csv")
	if err != nil {
		log.Fatal(err)
	}
	writer := csv.NewWriter(output)
	defer writer.Flush()
	writer.Write([]string{"id", "label"})

	for _, file := range files {
		img := extractImage(fmt.Sprintf("%s/%s", testDir, file.Name()))
		test := f(img.Pix)
		res, err := knn.Predict(test)
		if err != nil {
			log.Fatal(err)
		}
		id := strings.Split(file.Name(), ".")[0]
		label := fmt.Sprintf("%.0f", res[0])
		writer.Write([]string{id, label})
	}

}
