#include "Neuron.h"
#include <iostream>
#include <utility>

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

Neuron::Neuron(const rapidjson::Document & doc, unsigned int layerNum, unsigned int neuronIndex, unsigned int numOutputs)
{
	neuronIndex_ = neuronIndex;
	outputVal_ = doc["Layers"][layerNum][neuronIndex]["outputVal_"].GetDouble();
	gradient_ = doc["Layers"][layerNum][neuronIndex]["gradient_"].GetDouble();
	
	auto a = doc["Layers"][layerNum][neuronIndex]["outputWeights_"].Size();
	for (size_t i = 0; i < numOutputs; ++i)
	{
		outputWeights_.push_back(Connection());
		outputWeights_[i].weight = doc["Layers"][layerNum][neuronIndex]["outputWeights_"][i]["weight"].GetDouble();
		outputWeights_[i].deltaWeight = doc["Layers"][layerNum][neuronIndex]["outputWeights_"][i]["deltaWeight"].GetDouble();
	}
}

/*Neuron::Neuron(const Neuron & newNeuron) :
	neuronIndex_(newNeuron.neuronIndex_),
	outputVal_(newNeuron.outputVal_),
	gradient_(newNeuron.gradient_),
	outputWeights_(newNeuron.outputWeights_)
{}

Neuron::Neuron(Neuron && newNeuron):
	neuronIndex_(std::move(newNeuron.neuronIndex_)),
	outputVal_(std::move(newNeuron.outputVal_)),
	gradient_(std::move(newNeuron.gradient_)),
	outputWeights_(std::move(newNeuron.outputWeights_))
{}

Neuron & Neuron::operator=(const Neuron & newNeuron)
{
	neuronIndex_ = newNeuron.neuronIndex_;
	outputVal_ = newNeuron.neuronIndex_;
	gradient_ = newNeuron.gradient_;
	outputWeights_ = newNeuron.outputWeights_;
	return *this;
}*/

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

rapidjson::Document Neuron::SerializeToJSON() const
{
	rapidjson::Value json_val;
	rapidjson::Document doc;

	auto& allocator = doc.GetAllocator();
	doc.SetObject();

	json_val.SetDouble(eta_);
	doc.AddMember("eta_", json_val, allocator);

	json_val.SetDouble(alpha_);
	doc.AddMember("alpha_", json_val, allocator);

	json_val.SetUint(neuronIndex_);
	doc.AddMember("neuronIndex_", json_val, allocator);

	json_val.SetDouble(outputVal_);
	doc.AddMember("outputVal_", json_val, allocator);

	json_val.SetDouble(gradient_);
	doc.AddMember("gradient_", json_val, allocator);


	rapidjson::Value weightsArray(rapidjson::kArrayType);
	for (size_t i = 0; i < outputWeights_.size(); ++i)
	{
		rapidjson::Value weightArray(rapidjson::kArrayType);
		weightArray.SetObject();
		json_val.SetDouble(outputWeights_[i].weight);
		weightArray.AddMember("weight", json_val, allocator);

		json_val.SetDouble(outputWeights_[i].deltaWeight);
		weightArray.AddMember("deltaWeight", json_val, allocator);


		/*rapidjson::Value weightArray(rapidjson::kArrayType);
		json_val.SetDouble(outputWeights_[i].weight);
		weightArray.PushBack(json_val, allocator);

		json_val.SetDouble(outputWeights_[i].deltaWeight);
		weightArray.PushBack(json_val, allocator);*/

		weightsArray.PushBack(weightArray, allocator);

		/*json_val.SetDouble(outputWeights_[i].deltaWeight);
		json_val.AddMember("deltaWeight", json_val, allocator);
		weightArray.PushBack(json_val, allocator);


		/*json_val.SetObject();
		json_val.AddMember("weight", outputWeights_[i].weight, allocator);
		weightArray.PushBack(json_val, allocator);

		json_val.SetObject();
		json_val.AddMember("deltaWeight", outputWeights_[i].deltaWeight, allocator);
		weightArray.PushBack(json_val, allocator);

		weightsArray.PushBack(weightArray, allocator);*/
	}
	doc.AddMember("outputWeights_", weightsArray, allocator);
	return doc;
}

/*Neuron Neuron::DeserializeFromJSON(const rapidjson::Value & doc)
{
}*/


double Neuron::RandomWeight()
{
	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<double> dst(0, 1);
	return dst(rng);
	//return rand() / double(RAND_MAX);
}

double Neuron::TransferFunction(double val)
{
	//[-1.0; 1.0]
	return tanh(val);
}

double Neuron::TransferFunctionDerivative(double val)
{
	//fast aproximate tanh function
	return 1.0 - tanh(val)*tanh(val);
	//return 1.0 - val*val;

}

//Neuron::Neuron() {}
