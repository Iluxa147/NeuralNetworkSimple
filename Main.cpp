#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include "Net.h"
#include "TrainingData.h"

#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"

//#define TmpDataCreate
#define TmpDataCreateJSON

void ShowVectorVals(std::string label, std::vector<double>& v)
{
	std::cout << label << "  ";
	for (size_t i = 0; i < v.size(); ++i)
	{
		std::cout << v[i] << "  ";
	}
	std::cout << std::endl;
}

#ifdef TmpDataCreate
void CreateTrainingDataFile()
{
	std::FILE *stream;
	freopen_s(&stream, "TrainingData.txt", "w", stdout);

	//XOR - two inputs/one output
	std::cout << "topology: 2 4 1" << std::endl;
	for (int i = 2000; i >= 0; --i)
	{
		//int n2 = (int)(2.0f*rand() / double(RAND_MAX));
		int n1 = rand() % 2;
		int n2 = rand() % 2;
		int t = n1^n2; // 0 or 1
		std::cout << "in: " << n1 << ".0 " << n2 << ".0 " << std::endl;
		std::cout << "out: " << t << ".0" << std::endl;
	}

	fclose(stdout);
}
#endif // TmpDataCreate

#ifdef TmpDataCreateJSON
rapidjson::Document CreateTrainingDataJSON()
{
	std::FILE *f;
	fopen_s(&f, "TrainingData.json", "wb");

	rapidjson::Value json_val;

	rapidjson::Document doc;
	auto& allocator = doc.GetAllocator();

	doc.SetObject();

	json_val.SetArray()
		.PushBack(2, allocator)
		.PushBack(4, allocator)
		.PushBack(1, allocator);

	doc.AddMember("topology", json_val, allocator);

	/*for (int i = 1; i >= 0; --i)
	{
		int n1 = rand() % 2;
		int n2 = rand() % 2;
		int t = n1^n2; // 0 or 1

		json_val.SetArray()
			.PushBack(static_cast<double>(n1), allocator)
			.PushBack(static_cast<double>(n2), allocator);
		doc.AddMember("in", json_val, allocator);

		json_val.SetDouble(static_cast<double>(t));
		doc.AddMember("out", json_val, allocator);
	}*/



	char writeBuffer[256];
	rapidjson::FileWriteStream os(f, writeBuffer, sizeof(writeBuffer));
	
	rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
	doc.Accept(writer);
	fclose(f);

	return doc;
}
#endif //TmpDataCreateJSON

int main()
{
#ifdef TmpDataCreate
	CreateTrainingDataFile();
#endif // TmpDataCreate

#ifdef TmpDataCreateJSON
	CreateTrainingDataJSON();
#endif // TmpDataCreateJSON

	TrainingData trainData("TrainingData.txt");

	//number of neurons on each layer from start to end (excluding bias neuron). Vector size is a layer count
	
	std::vector<unsigned int> topology;
	trainData.GetTopology(topology);
	
	Net<double> myNet(topology);
	
	std::vector<double> inputVals;
	std::vector<double> targetVals;
	std::vector<double> resultVals;

	int trainingPass = 0;

	std::FILE *stream;
	freopen_s(&stream, "Result.txt", "w", stdout);

	while (!trainData.isEof())
	{
		++trainingPass;
		std::cout << std::endl << "Pass " << trainingPass;

		//get new input data and feed forward
		if (trainData.GetNextInputs(inputVals) != topology[0])
		{
			break;
		}
		ShowVectorVals(": Inputs: ", inputVals);
		myNet.FeedForward(inputVals);

		//collect net's actual results
		myNet.GetResults(resultVals);
		ShowVectorVals("Outputs: ", resultVals);

		//train net what outputs should have been
		trainData.GetTargetOutputs(targetVals);
		ShowVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.BackProp(targetVals);

		//report how well the training is working, average over recent samples
		std::cout << "Net recent average error: " << myNet.GetRecentAverageError() << std::endl;
	}

	std::cout << std::endl << "That's it!" << std::endl;
	
	fclose(stdout);

	
	/*std::vector<unsigned int> topology; 

	std::vector<double> inputVals;
	std::vector<double> targetVals;
	std::vector<double> resultVals;
	
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);

	Net<double> myNet(topology);



	myNet.BackProp(targetVals);

	myNet.GetResults(resultVals);*/

	system("pause");
	return 0;
}