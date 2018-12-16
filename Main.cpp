#include <iostream>
#include <vector>
#include "Net.h"

int main()
{
	//number of neurons on each layer from start to end (excluding bias neuron). Vector size is a layer count
	std::vector<unsigned int> topology; 

	std::vector<double> inputVals;
	std::vector<double> targetVals;
	std::vector<double> resultVals;
	
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);

	Net<double> myNet(topology);

	//myNet.FeedForward(inputVals);
	//myNet.BackProp(targetVals);
	//myNet.GetResults(resultVals);


	
	
	
	
	/*char c[2] = { '2','5' };


	std::cout << "c " << c << std::endl;
	std::cout << "*c " << *c << std::endl;
	std::cout << "&c " << &c << std::endl;


	std::cout << "*(c+1) " << *(c+1) << std::endl;
	std::cout << "*c+1 " << *c + 1 << std::endl;*/


	system("pause");
	return 0;
}