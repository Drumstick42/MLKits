require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs-node');
const loadCSV = require('./load-csv');
const LinearRegressions = require('./linear-regressions');


let { features, labels, testFeatures, testLabels } = 
	loadCSV('./cars.csv', {
		shuffle: true,
		splitTest: 50,
		dataColumns: ['horsepower', 'weight', 'displacement'],
		labelColumns: ['mpg']
	});

//console.log(features, labels);

const regression = new LinearRegressions(features, labels, {
	learningRate: 10,
	iterations: 100
});

regression.train();

//console.log('Update M is: ', regression.weights.get(1, 0), 'Updated B is: ', regression.weights.get(0, 0));

const r2 = regression.test(testFeatures, testLabels);


console.log('R2 is: ', r2);

