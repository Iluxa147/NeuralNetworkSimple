#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

/*#include "rapidjson/reader.h"

struct TopologyJSON
{
	bool StartObject() { return true; }
	bool Key(const char* str, rapidjson::SizeType length, bool copy)
	{
		cout << "Key(" << str << ", " << length << ", " << std::boolalpha << copy << ")" << endl;
		return true;
	}
	bool EndObject(rapidjson::SizeType memberCount) { cout << "EndObject(" << memberCount << ")" << endl; return true; }
	bool StartArray() { cout << "StartArray()" << endl; return true; }
	bool EndArray(rapidjson::SizeType elementCount) { cout << "EndArray(" << elementCount << ")" << endl; return true; }
};*/


class TrainingData
{
public:
	TrainingData(const std::string filename);
	TrainingData();
	bool isEof() { return inTrainingDataFile_.eof(); }
	void GetTopology(std::vector<unsigned int>& topology);
	void RewindDatatFile();
	void CreateTrainingDataFile(const std::string filename);
	//void TrainingData::GetTopologyJSON(std::vector<unsigned int>& topology);

	// returns the number of input values read from the file
	unsigned int GetNextInputs(std::vector<double>& inputVals);
	unsigned int GetTargetOutputs(std::vector<double>& targetOutputVals);

private:
	std::ifstream inTrainingDataFile_;
	std::ofstream outTrainingDataFile_;
};