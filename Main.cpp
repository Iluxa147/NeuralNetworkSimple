#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>

#include <thread>
#include <mutex>

#include "Net.h"
#include "TrainingData.h"

#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
//#include "rapidjson/writer.h"
//#include "rapidjson/reader.h"

using namespace rapidjson;
//using namespace std;


//#define CreateTrainData
//#define Training
//#define TmpTraining
//#define TryIt
#define TestCrossover



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
	for (int i = 2000; i > 0; --i)
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

int main()
{
#ifdef CreateTrainData
	TrainingData newTrainData;
	newTrainData.CreateTrainingDataFile("TrainingData.txt");
#endif // CreateTrainData

#ifdef TmpTraining
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

	//freopen_s(&s

	freopen_s(&stream, "Result.txt", "w", stdout);

	//WIP generation module
	double tmpError = 100.0f;
	bool isTheBest;
	Net<double> tmpNet(topology);
	//Net<double> currentNet = tmpNet;


	for (size_t i = 0; i < 1; ++i)
	{

		while (!trainData.isEof())
		{
			//get new input data and feed forward
			trainData.GetNextInputs(inputVals);
			if (inputVals.size() != topology[0])
			{
				break;
			}

			++trainingPass;
			//std::cout << std::endl << "Pass " << trainingPass;

			//ShowVectorVals(": Inputs: ", inputVals);
			myNet.FeedForward(inputVals);

			//collect net's actual results
			myNet.GetResults(resultVals);
			//ShowVectorVals("Outputs: ", resultVals);

			//train net what outputs should have been
			trainData.GetTargetOutputs(targetVals);
			//ShowVectorVals("Targets:", targetVals);


			assert(targetVals.size() == topology.back());

			myNet.BackProp(targetVals);

			auto i1 = inputVals[0];
			auto i2 = inputVals[1];


			auto z = targetVals[0];
			auto x = resultVals[0];

			//report how well the training is working, average over recent samples
			//std::cout << "Net recent average error: " << myNet.GetRecentAverageError() << std::endl;

			if (fabs(myNet.GetRecentAverageError()) < fabs(tmpError))
			{
				tmpError = myNet.GetRecentAverageError();
				tmpNet = myNet;
				isTheBest = true;
				std::cout << std::endl << "Pass " << trainingPass;
			}
		}

		/*auto a = fabs(myNet.GetRecentAverageError());
		auto b = fabs(tmpError);

		auto z = 6.6215111956237598e-15;
		auto x = 4.0647951209393796e-06;
		auto c = z < x;
		long double a = 0.000000081372668603307829;
		a *= a;*/

		auto a = fabs(myNet.GetRecentAverageError());
		auto b = fabs(tmpError);

		tmpNet.SetGeneration(i);
		if (isTheBest)
		{
			myNet = tmpNet;
			std::cout << std::endl << "Generation " << myNet.GetGeneration() << std::endl;
			std::cout << "Min Error! " << myNet.GetRecentAverageError() << std::endl;
			myNet.SerializeToJSON("BestNet.json");

		}
		trainData.RewindDatatFile();

		isTheBest = false;

	}

	//std::cout << std::endl << "Generation " << myNet.GetGeneration() << std::endl;
	//std::cout << "Min Error! " << myNet.GetRecentAverageError() << std::endl;
	//myNet.DeserializeFromJSON("BestNet.json");

	//Net<double> newNet("BestNet.json");

	fclose(stdout);

#endif // TmpTraining

#ifdef Training

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

	//freopen_s(&s

	freopen_s(&stream, "Result.txt", "w", stdout);

	//WIP generation module
	double tmpError = 100.0f;
	bool isTheBest;
	Net<double> tmpNet(topology);


	for (size_t i = 0; i < 10; ++i)
	{

		while (!trainData.isEof())
		{
			//get new input data and feed forward
			trainData.GetNextInputs(inputVals);
			if (inputVals.size() != topology[0])
			{
				break;
			}

			++trainingPass;
			trainData.GetTargetOutputs(targetVals);

			myNet.TrainingInvariant(inputVals, targetVals, resultVals);

			if (fabs(myNet.GetRecentAverageError()) < fabs(tmpError))
			{
				tmpError = myNet.GetRecentAverageError();
				tmpNet = myNet;
				isTheBest = true;
				std::cout << std::endl << "Pass " << trainingPass;
			}
		}

		tmpNet.SetGeneration(i);
		if (isTheBest)
		{
			myNet = tmpNet;
			std::cout << std::endl << "Generation " << myNet.GetGeneration() << std::endl;
			std::cout << "Min Error! " << myNet.GetRecentAverageError() << std::endl;
			myNet.SerializeToJSON("BestNet2.json");

		}
		trainData.RewindDatatFile();

		isTheBest = false;

	}
	myNet.GetRecentAverageError();
	fclose(stdout);

#endif //Training
	
#ifdef TestCrossover

	std::vector<double> inputVals;
	std::vector<double> targetVals;
	std::vector<double> resultVals;
	std::vector<unsigned int> topology;

	TrainingData trainData("TrainingData.txt");
	trainData.GetTopology(topology);
	trainData.GetNextInputs(inputVals);
	trainData.GetTargetOutputs(targetVals);
	trainData.RewindDatatFile();

	Net<double> net1("BestNet.json");
	Net<double> net2("BestNet2.json");

	net1.Crossover(net2);

	net1.TrainingInvariant(inputVals, targetVals, resultVals);
	
	net1.SerializeToJSON("BestNetCross.json");

#endif TestCrossover

#ifdef TryIt

	Net<double> newNet("BestNetCross.json");

	while (std::cin)
	{
		unsigned int a, b;
		std::cout << "Input 2 bin numbers for neuroXOR!))" << std::endl;
		std::cin >> a >> b;

		std::vector<double> inputVals;
		std::vector<double> resultVals;

		inputVals.push_back(a);
		inputVals.push_back(b);

		newNet.FeedForward(inputVals);

		newNet.GetResults(resultVals);

		for (const auto &n : resultVals)
		{
			std::cout << "Answer is: " << fabs(roundf(n)) << " check: " << (a^b) << " actual output: " << n << std::endl;
		}
		std::cout << std::endl;
	}
#endif //TryIt


	system("pause");
	return 0;
}