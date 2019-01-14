#pragma once
#include <vector>
#include <iostream>
#include <assert.h>
#include "Neuron.h"

template <typename T>
class Net
{
public:
	Net(const std::vector<unsigned int>& topology);

	void FeedForward(const std::vector<T> &inputVals );
	void BackProp(const std::vector<T>& targetVals);
	void GetResults(std::vector<T>& resultVals) const;
	double GetRecentAverageError() const;

private:
	std::vector<Layer> layers_; //[numLayer] [numNeuron]
	double error_;
	double recentAverageError_;
	double recentAverageSmoothingFacor_;
};

template<typename T>
inline Net<T>::Net(const std::vector<unsigned int>& topology)
{
	unsigned int numLayers = topology.size();

	for (size_t layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		//fill net with layers
		layers_.push_back(Layer());

		//outputs connections quantity for each neuron in that particular layer. Ofc, no outputs at output layer
		unsigned int numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		std::cout << "Layer " << layerNum << std::endl;

		//fill current layer with neurons + bias neuron
		for (size_t neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
		{
			layers_.back().push_back(Neuron(numOutputs, neuronNum));
			std::cout << "Hello Neuron number " << neuronNum << std::endl;
		}
		layers_.back().back().SetOutputVal(1.0f); //set value for bias neuron in output layer
	}
}

template<typename T>
inline void Net<T>::FeedForward(const std::vector<T>& inputVals)
{
	assert(inputVals.size() == layers_[0].size() - 1);
	
	//assign input vals to neurons
	for (size_t i = 0; i < inputVals.size(); ++i)
	{
		layers_[0][i].SetOutputVal(inputVals[i]);
	}
	
	//foward propagate to next layers
	for (size_t layerNum = 1; layerNum < layers_.size(); ++layerNum)
	{
		Layer &prevLayer = layers_[layerNum - 1];
		for (size_t n = 0; n < layers_[layerNum].size()-1; ++n)
		{
			layers_[layerNum][n].FeedForward(prevLayer);
		}
	}
}

template<typename T>
inline void Net<T>::BackProp(const std::vector<T>& targetVals)
{
	Layer &outputLayer = layers_.back();
	error_ = 0.0f;

	//calculate overall net error (RMS of output neuron errors), excluding bias
	for (size_t n = 0; n < outputLayer.size()-1; ++n)
	{
		double delata = targetVals[n] - outputLayer[n].GetOutputVal();
		error_ += delata * delata;
	}
	error_ /= outputLayer.size()-1;
	error_ = sqrtf(error_);

	//recent average measurement
	recentAverageError_ = (recentAverageError_ * recentAverageSmoothingFacor_ + error_) / (recentAverageSmoothingFacor_ + 1.0f);

	//calculate output layer gradients, excluding bias
	for (size_t n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].CalcOutputGradients(targetVals[n]);
	}

	//calculate gradients on hidden layers
	for (size_t layerNum = layers_.size()-2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = layers_[layerNum];
		Layer &nextLayer = layers_[layerNum+1];

		for (size_t n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].CalcHiddenGradients(nextLayer);
		}
	}
	
	//update connection weights for all layers, excluding output layer ofc
	for (size_t layerNum = layers_.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = layers_[layerNum];
		Layer &prevLayer = layers_[layerNum-1];

		for (size_t n = 0; n < layer.size()-1; ++n)
		{
			layer[n].UpdateInputWeights(prevLayer);
		}
	}
}

template<typename T>
inline void Net<T>::GetResults(std::vector<T>& resultVals) const
{
	resultVals.clear();

	for (size_t n = 0; n < layers_.back().size() -1; ++n)
	{
		resultVals.push_back(layers_.back()[n].GetOutputVal());
	}

}

template<typename T>
inline double Net<T>::GetRecentAverageError() const
{
	return recentAverageError_;
}
