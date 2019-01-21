#pragma once
#include <vector>
#include <random>

#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"
#include "rapidjson/reader.h"


class Neuron;
typedef std::vector<Neuron> Layer;

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	//Neuron();
	Neuron(unsigned int numOutputs, unsigned int neuronIndex);
	Neuron(const rapidjson::Document& doc, unsigned int layerNum, unsigned int neuronIndex, unsigned int numOutputs); //from JSON constructor (deserializer)
	/*Neuron(const Neuron& newNeuron);
	Neuron(Neuron&& newNeuron);

	Neuron& operator = (const Neuron& newNeuron);*/

	void SetOutputVal(double val);
	double GetOutputVal() const;
	double SumDOW(const Layer& nextLayer) const;
	void FeedForward(const Layer& prevLayer);
	void CalcOutputGradients(double targetVal);
	void CalcHiddenGradients(const Layer& nextLayer);
	void UpdateInputWeights(Layer& prevLayer);
	rapidjson::Document SerializeToJSON() const;
	// Neuron DeserializeFromJSON(const rapidjson::Value& doc);

private:
	static double RandomWeight();
	static double TransferFunction(double val);
	static double TransferFunctionDerivative(double val);

private:
	const static double eta_;
	const static double alpha_;
	unsigned int neuronIndex_;
	double outputVal_;
	double gradient_;
	std::vector<Connection> outputWeights_;
};

