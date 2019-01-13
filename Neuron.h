#pragma once
#include <vector>
#include <random>

class Neuron;
typedef std::vector<Neuron> Layer;

struct Connection //TODO mb a class instead?
{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	//Neuron();
	Neuron(unsigned int numOutputs, unsigned int neuronIndex);
	void SetOutputVal(double val);
	double GetOutputVal() const;
	double SumDOW(const Layer& nextLayer) const;
	void FeedForward(const Layer& prevLayer);
	void CalcOutputGradients(double targetVal);
	void CalcHiddenGradients(const Layer& nextLayer);
	void UpdateInputWeights(Layer& prevLayer);

private:
	static double RandomWeight(); //TODO more uniform distribution
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

