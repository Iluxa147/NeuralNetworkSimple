#pragma once
#include <vector>
#include <iostream>
#include "Neuron.h"

template <typename T>
class Net
{
public:
	Net(const std::vector<unsigned int>& topology);

	void FeedForward(const std::vector<T> &inputVals );
	void BackProp(const std::vector<T>& targetVals);
	void GetResults(std::vector<T>& resultVals) const;

private:
	std::vector<Layer> layers_; //[numLayer] [numNeuron]
};

template<typename T>
inline Net<T>::Net(const std::vector<unsigned int>& topology)
{
	unsigned int numLayers = topology.size();

	for (size_t numLayer = 0; numLayer < numLayers; ++numLayer)
	{
		//fill net with layers
		layers_.push_back(Layer());

		//outputs connections quantity for each neuron in that particular layer (numLayer)
		unsigned int numOutputs = numLayer == numLayers - 1 ? 0 : topology[numLayer + 1];

		std::cout << "Layer " << numLayer << std::endl;

		//fill current layer with neurons + bias neuron
		for (size_t numNeuron = 0; numNeuron <= topology[numLayer]; ++numNeuron)
		{
			layers_.back().push_back(Neuron());
			std::cout << "Hello Neuron number " << numNeuron << std::endl;
		}
	}
}

template<typename T>
inline void Net<T>::FeedForward(const std::vector<T>& inputVals)
{
	//assign inp vals to neurons

	//foward propagate to next layers
}