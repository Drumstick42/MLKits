// Uses linear knn w/ standardization
require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

function knn(features, labels, predictionPoint, k) {
	const { mean, variance } = tf.moments(features, 0);

	const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

	return features
		.sub(mean)
		.div(variance.pow(0.5))
		.sub(scaledPrediction) //Subtract point from elementwise
		.pow(2)
		.sum(1)
		.pow(0.5)
		.expandDims(1)
		.concat(labels, 1)
		.unstack() // Covnverts to nroal array of tensors
		.sort((a, b) => 
			a.get(0) > b.get(0) ? 1 : -1
		)
		.slice(0, k)
		.reduce((acc, pair) => acc + pair.get(1), 0) / k;
	
}

let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
	shuffle: true, // Shuffle, so that test and training draw on random data
	splitTest: 10,
	dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'], // Only extract these columns for features
	labelColumns: ['price'] // Only extract price for labels
});

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
	const result = knn(features, labels, tf.tensor(testPoint), 10);
	const err = (testLabels[i][0] - result)/testLabels[i][0]*100;
	console.log('Guess ', result, testLabels[i][0]);
	console.log('Error: ', err);
});