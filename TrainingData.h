#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

class TrainingData
{
public:
	TrainingData(const std::string filename);
	bool isEof() { return trainingDataFile_.eof(); }
	void GetTopology(std::vector<unsigned int>& topology);

	// returns the number of input values read from the file
	unsigned int GetNextInputs(std::vector<double>& inputVals);
	unsigned int GetTargetOutputs(std::vector<double>& targetOutputVals);

private:
	std::ifstream trainingDataFile_;
};