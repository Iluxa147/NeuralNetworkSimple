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


//#define TmpDataCreate
//#define TmpDataCreateJSON
#define Training
//#define TryIt

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

#ifdef TmpDataCreateJSON
void CreateTrainingDataJSON()
{
	std::FILE *f;
	fopen_s(&f, "TrainingData.json", "wb");

	rapidjson::StringBuffer strBuf;
	rapidjson::Writer<rapidjson::StringBuffer> stringWriter(strBuf);

	rapidjson::Document doc;
	rapidjson::Value json_val;
	auto& allocator = doc.GetAllocator();

	stringWriter.StartObject();
	stringWriter.Key("topology");
	stringWriter.StartArray();
	stringWriter.Uint(2);
	stringWriter.Uint(4);
	stringWriter.Uint(1);
	stringWriter.EndArray();

	stringWriter.Key("IN_OUT");
	stringWriter.StartArray();

	for (int i = 2; i > 0; --i)
	{
		int n1 = rand() % 2;
		int n2 = rand() % 2;
		int t = n1^n2; // 0 or 1

		stringWriter.StartObject();
		stringWriter.Key("in");
		stringWriter.StartArray();
		stringWriter.Double(n1);
		stringWriter.Double(n2);
		stringWriter.EndArray();

		stringWriter.Key("out");
		stringWriter.Double(t);
		stringWriter.EndObject();

	}
	stringWriter.EndArray();
	stringWriter.EndObject();

	fwrite(strBuf.GetString(), 1, strBuf.GetSize(), f);

	fclose(f);

}
#endif //TmpDataCreateJSON

void f1(int& a, std::mutex& mtx)
{
	std::lock_guard<std::mutex> lock(mtx);
	a*=a;
	std::cout << "f1 ";
	std::cout << "ID is " << std::this_thread::get_id() << std::endl;

}

void f2()
{
	std::cout << "f2" << std::endl;
}

int main()
{

#ifdef TmpDataCreate
	CreateTrainingDataFile();
#endif // TmpDataCreate

#ifdef TmpDataCreateJSON
	CreateTrainingDataJSON();
#endif // TmpDataCreateJSON

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
	freopen_s(&stream, "Result.txt", "w", stdout);

	//WIP generation module
	double tmpError = 100.0f;
	bool isTheBest;
	Net<double> tmpNet(topology);
	Net<double> currentNet = tmpNet;


	for (size_t i = 0; i < 70000; ++i)
	{

		while (!trainData.isEof())
		{

			//get new input data and feed forward
			if (trainData.GetNextInputs(inputVals) != topology[0])
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

			//report how well the training is working, average over recent samples
			//std::cout << "Net recent average error: " << myNet.GetRecentAverageError() << std::endl;

			if (fabs(myNet.GetRecentAverageError()) < fabs(tmpError))
			{
				tmpError = myNet.GetRecentAverageError();
				tmpNet = myNet;
				isTheBest = true;
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

#endif //Training
	
#ifdef TryIt

	Net<double> newNet("BestNet.json");

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

	int a = 2;
	std::mutex mtx;

	std::thread t1(f1, std::ref(a), std::ref(mtx)); //f1 - pointer to function


	t1.join(); //wait for t1 end
	std::cout << a;
	//t2.join();


	system("pause");
	return 0;
}