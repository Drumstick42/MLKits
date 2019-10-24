const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegressions
{
	constructor(features, labels, options) {
		this.labels = tf.tensor(labels);

		this.features = this.processFeatures(features);

		this.options = Object.assign(
			{ learningRate: 0.1, iterations: 1000, batchSize: 5 }, 
			options
			);

		this.weights = tf.zeros([this.features.shape[1], 1]);
		this.mseHistory = [];

	}

	train()
	{
		batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);
		
		for (let i = 0; i < this.options.iterations; i++)
		{
			for (let j = 0; j < batchQuantity; j++)
			{
				const startIndex = j * this.options.batchSize;
				const { batchSize } = this.options;


				const featureSlice = this.features.slice(
					[startIndex, 0],
					[batchSize, -1]);

				const labelSlice = this.labels.slice([startIndex, 0],[batchSize, -1]);

				
			}
			this.recordMSE();
			this.updateLearningRate();


		}
	}


// Using array of arrays

gradientDescent2() 
{
	// Get m*x_i + b
	const currentGuessesForMPG = this.features.map(row => {
		return this.m * row[0] + this.b;
	});

	// Calculate slope w.r.t b
	const bSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
		return guess - this.labels[i][0];
	})) * 2 / this.features.length;

	// Calculate slope w.r.t. m
	const mSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
		return -1 * this.features[i][0] * (this.labels[i][0] - guess); 
	})) * 2 / this.features.length;



	// Multiply slopes by learning rate, subtract from b and m
	this.m = this.m - mSlope*this.options.learningRate;
	this.b = this.b - bSlope*this.options.learningRate;

}


// Using TensorFlow
gradientDescent(features, labels) 
{
	const currentGuesses = features.matMul(this.weights);
	const differences = currentGuesses.sub(labels);
	const slopes = features
		.transpose()
		.matMul(differences) 
		.div(features.shape[0]);

	this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
}

 test(testFeatures, testLabels) {
 	testLabels = tf.tensor(testLabels);
 	testFeatures = this.processFeatures(testFeatures)

	const predictions = testFeatures.matMul(this.weights);

	// Calc SS_res
	const res = testLabels.sub(predictions)
		.pow(2)
		.sum()
		.get(); // Sum returns a tensor

	//Calc SS_tot
	const tot = testLabels.sub(testLabels.mean())
		.pow(2)
		.sum()
		.get();

	return 1.0 - res / tot;
 }

 processFeatures (features)
 {
	features = tf.tensor(features);
	
	
	if(this.mean && this.variance)
	{
		features = features.sub(this.mean)
			.div(this.variance.pow(0.5))
	}
	else
	{
		features = this.standardize(features);
	}

	features = tf.ones([features.shape[0], 1]).concat(features, 1);

	return features
 }

 standardize(features)
 {
	 const { mean, variance} = tf.moments(features, 0);
	 
	 // Define these first time we standartize, and make sure we only use them when testing
	 this.mean = mean;
	 this.variance = variance;

	 return features.sub(mean)
	 	.div(variance.pow(0.5));
 }

 recordMSE()
 {
	 const MSE = this.features.matMul(this.weights)
		 .sub(this.labels)
		 .pow(2)
		 .sum()
		 .div(this.features.shape[0])
		 .get();

	this.mseHistory.unshift(MSE);
 }

 updateLearningRate()
 {
	 if (this.mseHistory.length < 2)
	{
		return;
	}

	if (this.mseHistory[0] > this.mseHistory[1])
	{
		this.options.learningRate /= 2;
	}
	else
	{
		this.options.learningRate *= 1.05;
	}
 }
}


// Export class
module.exports = LinearRegressions;