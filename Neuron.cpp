#include "Neuron.h"

Neuron::Neuron(unsigned int numOutputs)
{
	for (size_t i = 0; i < numOutputs; ++i)
	{
		outputWeights_.push_back(Connection());
		outputWeights_.back().weight = RandomWeight();
	}
}

double Neuron::RandomWeight()
{
	return rand() / double(RAND_MAX);
}

Neuron::Neuron()
{
}
