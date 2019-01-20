#include "TrainingData.h"

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

TrainingData::TrainingData(const std::string filename)
{
	inTrainingDataFile_.open(filename.c_str());
	//fopen_s(&trainingDataJSON_, "TrainingData.json", "rb");
}

TrainingData::TrainingData()
{
}

void TrainingData::GetTopology(std::vector<unsigned int>& topology)
{
	std::string line;
	std::string label;

	std::getline(inTrainingDataFile_, line);
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
	inTrainingDataFile_.clear();
	inTrainingDataFile_.seekg(0);
}

void TrainingData::CreateTrainingDataFile(const std::string filename)
{
	outTrainingDataFile_.open(filename.c_str());

	std::string topology = "topology: 2 4 1\n";

	outTrainingDataFile_.write(topology.c_str(), topology.size());

	//XOR - two inputs, one output
	for (int i = 2000; i > 0; --i)
	{
		int n1 = rand() % 2;
		int n2 = rand() % 2;
		int t = n1^n2; // 0 or 1
		std::string in = "in: " + std::to_string(static_cast<double>(n1)) + " " + std::to_string(static_cast<double>(n2)) + '\n';
		outTrainingDataFile_.write(in.c_str(), in.size());
		std::string out = "out: " + std::to_string(static_cast<double>(t)) + '\n';
		outTrainingDataFile_.write(out.c_str(), out.size());
	}
}

/*void TrainingData::GetTopologyJSON(std::vector<unsigned int>& topology)
{

	char readBuffer[5];
	rapidjson::FileReadStream is(trainingDataJSON_, readBuffer, sizeof(readBuffer));
	//is.Take();
	rapidjson::Document d;
	d.ParseStream(is);
}*/


unsigned int TrainingData::GetNextInputs(std::vector<double>& inputVals)
{
	inputVals.clear();

	std::string line;
	std::getline(inTrainingDataFile_, line);
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
		std::getline(inTrainingDataFile_, line);
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
	std::getline(inTrainingDataFile_, line);
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
