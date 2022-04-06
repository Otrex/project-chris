const brain = require('brain.js');
const XLSX = require('xlsx');
const fs = require('fs');
const path = require('path');


var workbook = XLSX.readFile(`${__dirname}/data/d1.xlsx`);
var jsa = XLSX.utils.sheet_to_json(workbook.Sheets['Table 1'], {raw: false});


const network = new brain.recurrent.LSTM();
const trainingData = jsa.map((d) => ({
  input: String(d.FULLNAME.replace('\r\n', ' ')).toLowerCase(),
  output: String(d.SEX)
}))

// fs.writeFileSync(path.join(__dirname, 'data', 'd1.json'), JSON.stringify(trainingData, null, 2), 'utf8');

network.train(trainingData, {
  iterations: 200
});

const output = network.run('prince nna')

console.log(output);