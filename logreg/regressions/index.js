require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LogisticRegression = require('./logistic-regressions')

const {features, labels, testFeature, testLabels} 
= loadCSV('./cars.csv', {
	dataColumns: [
		'horsepower',
		'displacement',
		'weight'
	], 
	labelColumns: ['passedemissions'],
	shuffle: true,
	spliTest: 50,
	converters: {
		passedemssions: (value) => {
			return value === 'TRUE' ? 1 : 0 // Encode labels
		}
	}
});

const regression = new LogisticRegression(features, labels, {
	learningRate: 0.5,
	iterations: 100,
	batchSize: 50
});

regression.train();
regression.predict([
	[130, 307, 1.75]	
]).print();
