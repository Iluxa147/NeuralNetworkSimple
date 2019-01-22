#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>

#include <thread>
#include <mutex>

#include <ctime>

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
#define DebugMultiThreadingTraining
#define MultiThreadingTraining
//#define TestCrossover
#define TryIt

#ifdef MultiThreadingTraining

void GetTrainingDataValuesFromFile(TrainingData& trainData, std::vector<double>& inputVals, std::vector<double>& targetVals, bool& isTrained1, bool& isTrained2, bool& isTrainingFinished)
{
	for (size_t i = 0; i < 1; ++i)
	{
		int j=0;
		while (!trainData.isEof())
		{
			//if (isTrained1 && isTrained2)
			if (isTrained1)
			{
				++j;

				trainData.GetNextInputs(inputVals);
				trainData.GetTargetOutputs(targetVals);
				isTrained1 = false;
				isTrained2 = false;
			}
		}
		trainData.RewindDatatFile();
	}
	isTrainingFinished = true;
}

void SingleTrainingCycle(TrainingData& trainData, Net<double>& net, std::vector<double>& inputVals, std::vector<double>& targetVals, bool& isTrained, bool& isTrainingFinished)
{
	Net<double> tmpNet = net;
	double tmpError = 100.0f;
	bool isTheBest = false;
	std::vector<double> resultVals;
	int trainingPass = 0;


#ifdef DebugMultiThreadingTraining
	std::vector<double> debugTmpInput;
	std::vector<double> debugTmpOutput;
	Net<double> debugNet = net;

	int isTheBestCount = 0;
#endif //DebugMT

	while (!isTrainingFinished)
	{

		while (!isTrained && !trainData.isEof())
		{
			++trainingPass;

			net.TrainingInvariant(inputVals, targetVals, resultVals);

			if (fabs(net.CalculateError(targetVals)) < fabs(tmpError))
			{
				tmpError = net.CalculateError(targetVals);
				tmpNet = net;
				isTheBest = true;
#ifdef DebugMultiThreadingTraining
				++isTheBestCount;
				std::cout << "   +++" << isTheBestCount << "+++";
#endif //DebugMT

			}

			///tmpNet.SetGeneration(i);
			if (isTheBest)
			{
				net = tmpNet;
#ifdef DebugMultiThreadingTraining
				std::cout << "---" << isTheBestCount << "---Best--- ";
				std::cout << "        " << net.CalculateError(targetVals) << " " << net.GetCurrentRecentAverageError() << " input: " << inputVals[0] << " " << inputVals[1] << " output: " << resultVals[0] << std::endl;
				
				debugTmpInput = inputVals;
				debugTmpOutput = resultVals;
				debugNet = net;
				//debugNet.TrainingInvariant(debugTmpInput, targetVals, debugTmpOutput);
				debugNet.FeedForward(debugTmpInput);
				debugNet.GetResults(debugTmpOutput);
				std::cout << "*********************************" << std::endl;
				std::cout << "after TrainingInvariant:  " << debugNet.CalculateError(targetVals) << " " << debugNet.GetCurrentRecentAverageError() << " input: " << debugTmpInput[0] << " " << debugTmpInput[1] << " output: " << debugTmpOutput[0] << std::endl << std::endl;

				debugNet.FeedForward(debugTmpInput);
				debugNet.GetResults(debugTmpOutput);
				std::cout << "*********************************" << std::endl;
				std::cout << "after TrainingInvariant:  " << debugNet.CalculateError(targetVals) << " " << debugNet.GetCurrentRecentAverageError() << " input: " << debugTmpInput[0] << " " << debugTmpInput[1] << " output: " << debugTmpOutput[0] << std::endl << std::endl;



#endif //DebugMT

				//std::cout << std::endl << "Generation " << net.GetGeneration() << std::endl;
				//std::cout << "Min Error! " << net.GetRecentAverageError() << std::endl;
				///myNet.SerializeToJSON("BestNet.json");
			}

			isTheBest = false;
			isTrained = true;
		}
	}
	net = tmpNet;
}

void f1(int& a, std::mutex& mtx)
{
	std::cout << "f1" << std::endl;
}

void f2(int& a, std::mutex& mtx)
{
	std::cout << "f2" << std::endl;
}

#endif // MultiThreadingTraining


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
	std::clock_t start;
	double duration;
	start = std::clock();

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
			//std::cout << std::endl << "Pass " << trainingPass;

			//train net what outputs should have been
			trainData.GetTargetOutputs(targetVals);
			//ShowVectorVals("Targets:", targetVals);

			//ShowVectorVals(": Inputs: ", inputVals);
			myNet.FeedForward(inputVals);

			//collect net's actual results
			myNet.GetResults(resultVals);
			//ShowVectorVals("Outputs: ", resultVals);

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
			myNet.SerializeToJSON("BestNet.json");

		}
		trainData.RewindDatatFile();

		isTheBest = false;

	}
	myNet.GetRecentAverageError();
	fclose(stdout);

#endif //Training
	

#ifdef MultiThreadingTraining

	//std::mutex mtx1;
	//std::mutex mtx2;
	
	bool isTrained1 = true;
	bool isTrained2 = true;
	bool isTrainingFinished = false;
	//double tmpError1 = 100.0f;
	//double tmpError2 = 100.0f;

	TrainingData trainData("TrainingData.txt");

	///number of neurons on each layer from start to end (excluding bias neuron). Vector size is a layer count
	std::vector<unsigned int> topology;
	trainData.GetTopology(topology);

	Net<double> net1(topology);
	Net<double> net2(topology);
	Net<double> netCross1(topology);
	//Net<double> netCross2(topology);

	std::vector<double> inputVals;
	std::vector<double> targetVals;
	std::vector<double> resultVals; //for crossover's TrainingInvariant

	std::thread threadTrainData(GetTrainingDataValuesFromFile, std::ref(trainData), std::ref(inputVals), std::ref(targetVals), std::ref(isTrained1), std::ref(isTrained2), std::ref(isTrainingFinished));

	std::thread SingleTrainingCycle1(SingleTrainingCycle, std::ref(trainData), std::ref(net1), std::ref(inputVals), std::ref(targetVals), std::ref(isTrained1), std::ref(isTrainingFinished));
	//std::thread SingleTrainingCycle2(SingleTrainingCycle, std::ref(trainData), std::ref(net2), std::ref(inputVals), std::ref(targetVals), std::ref(isTrained2), std::ref(isTrainingFinished));

	threadTrainData.join();
	SingleTrainingCycle1.join();
	//SingleTrainingCycle2.join();

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << '\n' << "execution time: " << duration << '\n';


	net1.SerializeToJSON("BestNetMT1.json");
	net2.SerializeToJSON("BestNetMT2.json");

	std::cout << std::endl << "net1 " << net1.GetCurrentRecentAverageError() << std::endl;

	//net1.Crossover(net2);

	netCross1 = net1;
	//netCross2 = net2.Crossover(net1);

	//net1.TrainingInvariant(inputVals, targetVals, resultVals);
	netCross1.TrainingInvariant(inputVals, targetVals, resultVals);

	std::cout << std::endl << "net2 " << net2.GetCurrentRecentAverageError() << std::endl;
	std::cout << std::endl << "net1 Crossed " << net1.GetCurrentRecentAverageError() << std::endl;
	std::cout << std::endl << "netCross1 " << netCross1.GetCurrentRecentAverageError() << std::endl;
	
	netCross1.SerializeToJSON("BestNetMTCross.json");

	//std::cout << std::endl << "netCross2 " << netCross2.GetRecentAverageError() << std::endl;

	//netCross.SerializeToJSON("BestNetMTCross.json");
	/*GetTrainingDataValuesFromFile(trainData, inputVals, targetVals);
	SingleTrainingCycle(net1, inputVals, targetVals);
	SingleTrainingCycle(net2, inputVals, targetVals);

	net1.Crossover(net2);

	net1.SerializeToJSON("BestNetMT1.json");
	net1.SerializeToJSON("BestNetMT2.json");*/

	/*int a = 2;
	std::mutex mtx;

	//std::thread t1(f1, std::ref(a), std::ref(mtx), f2, std::ref(a), std::ref(mtx)); //f1 - pointer to function

	std::thread t1(f1, std::ref(a), std::ref(mtx)); //f1 - pointer to function

	std::cout << "main" << std::endl;*/
	//t1.join(); //wait for t1 end
	//std::cout << a;
	//t2.join();


#endif //MultiThreadingTraining


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
	std::string filename = "BestNetMT1.json";
	std::cout << std::endl << filename << std::endl;
	Net<double> newNet(filename);

	while (std::cin)
	{
		unsigned int a, b;
		std::cout << std::endl << "Input 2 bin numbers for neuroXOR!))" << std::endl;
		std::cin >> a >> b;

		std::vector<double> inputVals;
		std::vector<double> resultVals;
		std::vector<double> targetVals(1);
		targetVals[0] = (a^b);

		inputVals.push_back(a);
		inputVals.push_back(b);

		net1.FeedForward(inputVals);

		net1.GetResults(resultVals);

		for (const auto &n : resultVals)
		{
			std::cout << "Answer is: " << fabs(roundf(n)) << " check: " << (a^b) << " actual output: " << n << " error: "<< net1.CalculateError(targetVals) << " " << net1.GetCurrentRecentAverageError() << std::endl;
		}
		std::cout << std::endl;
	}
#endif //TryIt

	system("pause");

	return 0;
}