package main

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"
)

const irisSmallCsvPath string = `C:\Users\THPC\Main\Development\Go\lnet\data\iris_small.csv`

func extractIrisSmall() (matrix, []int) {
	var file *os.File
	var err error

	file, err = os.Open(irisSmallCsvPath)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	var reader *csv.Reader = csv.NewReader(file)
	var rawCsvData [][]string

	rawCsvData, err = reader.ReadAll()

	if err != nil {
		log.Fatal(err)
	}

	var rawCsvDataLen int = len(rawCsvData)
	if rawCsvDataLen <= 1 {
		log.Fatal("Small Iris CSV data file contains no rows or only the header row. Can not extract data")
	}

	var inputs matrix = make(matrix, rawCsvDataLen - 1)
	var targets []int = make([]int, rawCsvDataLen - 1)

	for recrodIndex, record := range rawCsvData {
		if recrodIndex == 0 {
			continue
		}
		var recrodLen int = len(record)

		if recrodLen != 7 {
			log.Fatal("Small Iris CSV data file contains rows with less or more than 7 values. All rows must have exactly 7 values")
		}

		var inputSample vector = make(vector, recrodLen - 3)
		var sampleTarget int = -1

		for recrodValueIndex, recrodValue := range record {
			if recrodValueIndex == 0 || recrodValueIndex == 1 || recrodValueIndex == 2 || recrodValueIndex == 3 {
				floatValue, err := strconv.ParseFloat(recrodValue, 64)
				if err != nil {
					log.Fatal(err)
				}

				inputSample[recrodValueIndex] = floatValue
				continue
			}

			floatValue, err := strconv.ParseFloat(recrodValue, 64)
			var intValue int = int(floatValue)

			if err != nil {
				log.Fatal(err)
			}

			if intValue != 1 && intValue != 0 {
				log.Fatal("All row values from indexes 4 through 6 must be integers 0 or 1")
			}

			if recrodValueIndex == 4 && intValue == 1 {
				sampleTarget = 0
			}
			if recrodValueIndex == 5 && intValue == 1 {
				sampleTarget = 1
			}
			if recrodValueIndex == 6 && intValue == 1 {
				sampleTarget = 2
			}
		}

		if sampleTarget == -1 {
			log.Fatal("A row does not contain a value of 1 in any of the label cloumns")
		}

		inputs[recrodIndex - 1] = inputSample
		targets[recrodIndex - 1] = sampleTarget
	}

	return inputs, targets

}
