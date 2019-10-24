// Declare new array
const outputs = [];


function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}



// Algorithm determines relationship between Features (inputs) and labels (outputs)
// Will use K-Nearest Neighbor(knn)
// "Birds of a feather flock together"
// If you have a bunch of things similar sitting together, they are probably the same type
// Looks at similar initial conditions to determine the output
// 1.) Drop a bunch of balls all over board, record the bucket #
// 2.) For each observation, subtract drop point from 300 px, take abs value, which determines
//	   how similar drops are to the IC we care about
// 3.) Sort results from least to greatest
// 4.) look at the top k records. WHat was the most common bucket?
//		No strict rules for k.
// 5.) Whichever bucket came up most frequenty is probably where it will go
function runAnalysisVaryK() {
	const testSetSize = 100; // Don't run too high
	const [testSet, trainingSet] = splitDataset(minMax(outputs, 3) , testSetSize); // Assign results to separate arrays  

	let numberCorrect = 0;
	//for (let i = 0; i < testSet.length; i++)
	//{
	//	const bucket = knn(trainingSet, testSet[i][0]); // Prediction of the bucket based on training set
	//	if (bucket === testSet[i][3]) {
	//		numberCorrect++;
	//	}
	//}

	//console.log('Accuracy: ', numberCorrect / testSetSize)

	_.range(1, 20).forEach(k => {
		const accuracy = _.chain(testSet)
			.filter(testPoint => // Removes entries that are incorrect
				knn(trainingSet, _.initial(testPoint), k) === testPoint[3]
			)
			.size()
			.divide(testSetSize)
			.value();

		console.log('For k of ', k," Accuracy is: ", accuracy);
	});
}

const k = 10;

// Test features separately for a single k to test accuracy
function runAnalysis() {
	const testSetSize = 100; // Don't run too high

	let numberCorrect = 0;
	//for (let i = 0; i < testSet.length; i++)
	//{
	//	const bucket = knn(trainingSet, testSet[i][0]); // Prediction of the bucket based on training set
	//	if (bucket === testSet[i][3]) {
	//		numberCorrect++;
	//	}
	//}

	//console.log('Accuracy: ', numberCorrect / testSetSize)


	// feature 1 == 0, etc.
	_.range(0, 3).forEach(feature => {
		const data = _.map(outputs, row => [row[feature], _.last(row)]);
		const [testSet, trainingSet] = splitDataset(minMax(data, 1) , testSetSize); // Assign results to separate arrays  
		const accuracy = _.chain(testSet)
			.filter(testPoint => // Removes entries that are incorrect
				knn(trainingSet, _.initial(testPoint), k) === _.last(testPoint)
			)
			.size()
			.divide(testSetSize)
			.value();

		console.log('For feature of ', feature," Accuracy is: ", accuracy);
	});
}


// We add extra features by changing the distance function
// Everything else remains the same, as the distance determines the prediction made by knn
function knn(data, point, k) {
	// Assume point doens't contain label, so we can use knn later for make predictions
	return _.chain(data)
		.map(row => {
			return [
				distance(_.initial(row), point),
				_.last(row)
			]
		})
		.sortBy(row => row[0])
		.slice(0, k)
		.countBy(row => row[1])
		.toPairs() //Convert key-value pairs as arrays
		.sortBy(row => row[1])
		.last() // Get array w/ most counts
		.first() //Get the bucket number
		.parseInt() // Convert to Int
		.value();
}

function distance(pointA, pointB) {
  return _.chain(pointA)
	.zip(pointB) // Now, corresponding indices in both arrays are contained in the same array
	.map(([a, b]) => (a - b) ** 2)
	.sum()
	.value() ** 0.5;
}

function distanceOneVar(pointA, pointB) {
  return Math.abs(pointA - pointB);
}

// Finding ideal K
// 1.) Record a bunch of data
// 2.) Split data into a training set and a test set
// 3.) For each test record, run KNN using the training data
// 4. Does the result of KNN equal the 'test' record bucket?
//    If it does, we know if the change is an improvement for the overall algorithm.

function splitDataset(data, testCount) {
	const shuffled = _.shuffle(data); // We want to shuffle, to ensure that the split datasets contain point all across the board

	const testSet = _.slice(shuffled, 0, testCount);
	const trainingSet = _.slice(shuffled, testCount);

	return [testSet, trainingSet];
}

// featureCount tells us how many columns of features we have and want to normalize
function minMax(data, featureCount)
{
	const clonedData = _.cloneDeep(data);

	for (let i = 0; i < featureCount; i++)
	{
		const column = clonedData.map(row => row[i]); // column is just an array of numebrs

		const min = _.min(column);
		const max = _.max(column);

		for (let j = 0; j < clonedData.length; j++)
		{
			clonedData[j][i] = (clonedData[j][i] - min) / (max - min);
		}
	}

	return clonedData;
}