const _ = require('lodash');

const loadData = () => {
    const randoms = _.range(0, 999999); // Creates array w/ elements from 0 to 999...

    return randoms;
};

const data = loadData();

debugger;