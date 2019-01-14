#include "Neuron.h"
#include <iostream>

#define DEBUG

//[0.0; 1.0] overall net training rate
const double Neuron::eta_ = 0.15f;

// [0.0 n] multtiplier of last weight change (momentum)
const double Neuron::alpha_ = 0.5f; 

Neuron::Neuron(unsigned int numOutputs, unsigned int neuronIndex)
{
	for (size_t i = 0; i < numOutputs; ++i)
	{
		outputWeights_.push_back(Connection());
		outputWeights_.back().weight = RandomWeight();
	}
	neuronIndex_ = neuronIndex;

#ifdef DEBUG
	std::cout << numOutputs << " output(s) " << "at index " << neuronIndex_ << " ";
#endif // DEBUG
}

void Neuron::SetOutputVal(double val)
{
	outputVal_ = val;
}

double Neuron::GetOutputVal() const
{
	return outputVal_;
}

double Neuron::SumDOW(const Layer & nextLayer) const
{
	double sum = 0.0f;

	//sum of contributions of the errors at the feeded neurons
	for (size_t n = 0; n < nextLayer.size()-1; ++n)
	{
		sum += outputWeights_[n].weight*nextLayer[n].gradient_;
	}

	return sum;
}

void Neuron::FeedForward(const Layer & prevLayer)
{
	double sum = 0.0f;

	//sum previous layer's outputs (it is neuron inputs) for each neuron including bias
	for (size_t n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].outputVal_ * prevLayer[n].outputWeights_[neuronIndex_].weight;

/*#ifdef DEBUG
		std::cout << "layer number " << n << "at index " << neuronIndex_ << " ";
#endif // DEBUG*/
	}

	outputVal_ = Neuron::TransferFunction(sum);
}

void Neuron::CalcOutputGradients(double targetVal)
{
	double delta = targetVal - outputVal_;
	gradient_ = delta * Neuron::TransferFunctionDerivative(outputVal_);
}

void Neuron::CalcHiddenGradients(const Layer& nextLayer)
{
	double dow = SumDOW(nextLayer);
	gradient_ = dow * Neuron::TransferFunctionDerivative(outputVal_);
}

void Neuron::UpdateInputWeights(Layer & prevLayer)
{
	for (size_t n = 0; n < prevLayer.size(); ++n)
	{
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.outputWeights_[neuronIndex_].deltaWeight;
		double newDeltaWeight =
			// individual input, magnified by the gradient and train rate
			eta_ * neuron.outputVal_ * gradient_
			//add a momentum (a fraction of the previous delta weight)
			+ alpha_ * oldDeltaWeight;

		neuron.outputWeights_[neuronIndex_].deltaWeight = newDeltaWeight;
		neuron.outputWeights_[neuronIndex_].weight += newDeltaWeight;
	}

}


double Neuron::RandomWeight()
{
	return rand() / double(RAND_MAX);
}

double Neuron::TransferFunction(double val)
{
	//[-1.0; 1.0]
	return tanh(val);
}

double Neuron::TransferFunctionDerivative(double val)
{
	//fas aproximate tanh function
	return 1.0 - val*val;
}

//Neuron::Neuron() {}
