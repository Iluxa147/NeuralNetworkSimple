#pragma once
#include <vector>
#include <random>


struct Connection //TODO mb a class insted?
{
	double weight;
	double deltaWeight;
};


class Neuron
{
public:
	Neuron();
	Neuron(unsigned int numOutputs);

private:
	static double RandomWeight(); //TODO more uniform distribution

private:
	double outputVal_;
	std::vector<Connection> outputWeights_;
};

typedef std::vector<Neuron> Layer;
