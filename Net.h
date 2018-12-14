#pragma once
#include <vector>
#include <iostream>

class Neuron {};

typedef std::vector<Neuron> Layer;

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
		
		std::cout << "Layer " << numLayer << std::endl;

		//fill current layer with neurons + bias neuron
		for (size_t numNeuron = 0; numNeuron <= topology[numLayer]; ++numNeuron)
		{
			layers_.back().push_back(Neuron());
			std::cout << "Hello Neuron number " << numNeuron << std::endl;
		}
	}
}
