#include "TrainingData.h"

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

TrainingData::TrainingData(const std::string filename)
{
	trainingDataFile_.open(filename.c_str());
	
	fopen_s(&trainingDataJSON_, "TrainingData.json", "rb");
}

TrainingData::~TrainingData()
{
	fclose(trainingDataJSON_);
	delete trainingDataJSON_;
}

void TrainingData::GetTopology(std::vector<unsigned int>& topology)
{
	std::string line;
	std::string label;

	std::getline(trainingDataFile_, line);
	std::stringstream ss(line);
	ss >> label;

	if (this->isEof() || label.compare("topology:") != 0)
	{
		abort();
	}

	while (!ss.eof())
	{
		unsigned int n;
		ss >> n;
		topology.push_back(n);
	}

	return;
}

void TrainingData::RewindDatatFile()
{
	trainingDataFile_.clear();
	trainingDataFile_.seekg(0);
}

void TrainingData::GetTopologyJSON(std::vector<unsigned int>& topology)
{

	char readBuffer[5];
	rapidjson::FileReadStream is(trainingDataJSON_, readBuffer, sizeof(readBuffer));
	//is.Take();
	rapidjson::Document d;
	d.ParseStream(is);

	//rapidjson::Value doc;

	//topology.push_back(doc["topology"].GetUint);
}


unsigned int TrainingData::GetNextInputs(std::vector<double>& inputVals)
{
	inputVals.clear();

	std::string line;
	std::getline(trainingDataFile_, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;

	if (label == "topology:")
	{
		while (!ss.eof())
		{
			unsigned int n;
			ss >> n;
		}
		std::getline(trainingDataFile_, line);
		const char* c = line.c_str();
		ss.clear();
		ss.write(c, sizeof(line));
		ss.seekg(0);
		ss >> label;
	}

	if (label.compare("in:") == 0)
	{
		double oneValue;

		while (ss >> oneValue)
		{
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}

unsigned int TrainingData::GetTargetOutputs(std::vector<double>& targetOutputVals)
{
	targetOutputVals.clear();

	std::string line;
	std::getline(trainingDataFile_, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;


	if (label.compare("out:") == 0)
	{
		double oneValue;

		while (ss >> oneValue)
		{
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}
