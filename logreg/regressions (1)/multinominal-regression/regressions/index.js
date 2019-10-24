require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LogisticRegression = require('./logistic-regression');
//const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');

const mnistData = mnist.training(0, 1);

console.log(mnistData);