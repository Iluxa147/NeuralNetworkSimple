#include "TrainingData.h"

TrainingData::TrainingData(const std::string filename)
{
	trainingDataFile_.open(filename.c_str());
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

unsigned int TrainingData::GetNextInputs(std::vector<double>& inputVals)
{
	inputVals.clear();

	std::string line;
	std::getline(trainingDataFile_, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;


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
