const fs = require('fs'); // filesystem
const _ = require('lodash')
const shuffleSeed = require('shuffle-seed'); // Lets us use a seed string to shuffle labels and features in the same way (shuffle observations)
                                             // Also allows us to keep the same shuffling across runs on the same data

// Extract only specified columns
function extractColumns(data, columnNames) {
    const headers = _.first(data);

    const indexes = _.map(columnNames, column => headers.indexOf(column));
    const extracted = _.map(data, row => _.pullAt(row, indexes));

    return extracted;
}

function loadCSV(filename, { converters = {}, dataColumns = [], labelColumns = [], 
    shuffle = true, splitTest = false }) {
    let data = fs.readFileSync(filename, { encoding: 'utf-8'});
    // Split the read string into a 2D array
    data = data.split('\n').map(row => row.split(','));
    // Get rid of carriage return/empty columns
    data = data.map(row => 
        _.dropRightWhile(row, val => (val === '') || (val === '\r')));
    
    // Get a ref to column headers
    const headers = _.first(data); 
    
    data = data.map((row, index) => {
        if (index === 0) {
            return row; // If header, do nothing
        }

        // Parse individual  values in each row
        return row.map((element, index) => {
            if (converters[headers[index]]) {
                const converted = converters[headers[index]](element);
                return _.isNaN(converted) ? element : converted;
            }

            const result =  parseFloat(element);
            return _.isNaN(result) ? element : result;
        });
    });

    let labels = extractColumns(data, labelColumns);
    data = extractColumns(data, dataColumns);

    // Can now dump headers
    data.shift();
    labels.shift()

    if(shuffle) {
        data = shuffleSeed.shuffle(data, 'phrase');
        labels = shuffleSeed.shuffle(labels, 'phrase');
    }

    if (splitTest) {
        const trainSize = _.isNumber(splitTest) ? splitTest : Math.floor(data.length/2);
        
        return {
            features: data.slice(0, trainSize),
            labels: labels.slice(0, trainSize),
            testFeatures: data.slice(trainSize),
            testLabels: labels.slice(trainSize)
        }
    } else {
        return {features: data, labels: labels};
    }
}

const{ features, labels, testFeatures, testLabels} = loadCSV('data.csv',
{
    dataColumns: ['height', 'value'],
    labelColumns: ['passed'],
    shuffle: true,
    splitTest: true,
    converters: {
        passed: val => (val === 'TRUE' ? 1 : 0 ) // if column is named passed, use converter instead of trying to parse as float
    }
});

console.log(features);
console.log(labels);
console.log(testFeatures);
console.log(testLabels);