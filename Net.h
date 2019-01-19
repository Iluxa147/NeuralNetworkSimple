#pragma once
#include <vector>
#include <iostream>
#include <assert.h>
#include "Neuron.h"

#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include <rapidjson/filereadstream.h>
#include "rapidjson/writer.h"
#include "rapidjson/reader.h"

//#define Debug

template <typename T>
class Net
{
public:
	Net(const std::vector<unsigned int>& topology);
	Net(std::string filename); //constructor from JSON (deserializer)

	void FeedForward(const std::vector<T> &inputVals );
	void BackProp(const std::vector<T>& targetVals);
	void GetResults(std::vector<T>& resultVals) const;
	void SerializeToJSON(std::string filename) const;
	void Crossover(Net<T>& net);
	double GetRecentAverageError() const { return recentAverageError_; };
	void SetGeneration(unsigned int num) { generation_ = num; };
	unsigned int GetGeneration() const { return generation_; };
	//void DeserializeFromJSON(std::string filename);

private:
	std::vector<Layer> layers_; //[numLayer] [numNeuron]
	double error_;
	double recentAverageError_;
	double recentAverageSmoothingFacor_;
	unsigned int generation_ = 0;
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
inline Net<T>::Net(std::string filename)
{
	std::FILE *f;
	fopen_s(&f, filename.c_str(), "rb");

	char buffer[256];
	rapidjson::FileReadStream is(f, buffer, sizeof(buffer));
	rapidjson::Document doc;
	doc.ParseStream(is);

	generation_ = doc["generation_"].GetUint();
	recentAverageError_ = doc["recentAverageError_"].GetDouble();

	unsigned int numLayers = doc["Layers"].Size();

	for (size_t layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		//fill net with layers
		layers_.push_back(Layer());

		//outputs connections quantity for each neuron in that particular layer. Ofc, no outputs at output layer
		unsigned int numOutputs = layerNum == numLayers - 1 ? 0 : doc["Layers"][layerNum+1].Size()-1;
		
		//fill current layer with neurons + bias neuron
		for (size_t neuronNum = 0; neuronNum < doc["Layers"][layerNum].Size(); ++neuronNum)
		{
			auto a = doc["Layers"][layerNum][neuronNum]["outputWeights_"].Size();
			assert(numOutputs == doc["Layers"][layerNum][neuronNum]["outputWeights_"].Size());
			assert(neuronNum == doc["Layers"][layerNum][neuronNum]["neuronIndex_"].GetUint());

			layers_.back().push_back(Neuron(doc, layerNum, neuronNum, numOutputs));
		}
	}
	
	/*std::vector<unsigned int> topology;

	for (size_t i = 0; i < doc["Layers"].Size(); ++i)
	{
		//add neurons, excluding bias neuron
		topology.push_back(doc["Layers"][i].Size() - 1);
	}*/

	//doc.ParseStream<0, UTF8<>, rapidjson::FileReadStream>(is);

	fclose(f);

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
#ifdef Debug
		if (fabs(delata) <= 0.0000000813727)
		{
			auto a = 1;
		}
#endif
		error_ += delata * delata;
	}
	error_ /= outputLayer.size()-1;
	error_ = sqrtf(error_);

	//recent average measurement
	recentAverageError_ = (recentAverageError_ * recentAverageSmoothingFacor_ + error_) / (recentAverageSmoothingFacor_ + 1.0f);
#ifdef Debug
	if (fabs(recentAverageError_) <= 0.0000000813727)
	{
		auto a = 1;
	}
#endif

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
inline void Net<T>::SerializeToJSON(std::string filename) const
{
	rapidjson::Value json_val;
	rapidjson::Document doc;

	auto& allocator = doc.GetAllocator();
	doc.SetObject();

	json_val.SetUint(generation_);
	doc.AddMember("generation_", json_val, allocator);

	json_val.SetDouble(recentAverageError_);
	doc.AddMember("recentAverageError_", json_val, allocator);

	rapidjson::Value layersArray(rapidjson::kArrayType);

	for (size_t i = 0; i < layers_.size(); ++i)
	{
		rapidjson::Value neuronsArray(rapidjson::kArrayType);
		for (size_t n = 0; n < layers_[i].size(); ++n)
		{
			json_val.CopyFrom(layers_[i][n].SerializeToJSON(), allocator);
			neuronsArray.PushBack(json_val, allocator);
		}
		layersArray.PushBack(neuronsArray, allocator);
	}
	doc.AddMember("Layers", layersArray, allocator);

	std::FILE *f;
	fopen_s(&f, filename.c_str(), "wb");

	char writeBuffer[65536];
	FileWriteStream os(f, writeBuffer, sizeof(writeBuffer));
	Writer<FileWriteStream> writer(os);

	doc.Accept(writer);
	fclose(f);
}

template<typename T>
inline void Net<T>::Crossover(Net<T>& net)
{
	//topology must be the same (layers and neurons count)
	assert(net.layers_.size() == layers_.size());
	for (size_t i = 0; i < layers_.size(); ++i)
	{
		assert(net[i].size() == layers_[i].size());
	}

	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<double> dst(0, 1);
	dst(rng);
}
