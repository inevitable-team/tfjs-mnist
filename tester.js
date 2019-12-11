const mnist = require("./mnist.js");

runner(1, { epochs: 20 });
runner(2, { epochs: 20, imageLimiter: 0.5 });

async function runner(id, config) {
    let mnistTester = new mnist(config);
    await mnistTester.setup();
    console.log(`${id} Training Data Size: `, mnistTester.trainingData.length);
    console.log(`${id} Testing Data Size: `, mnistTester.testingData.length);
    await mnistTester.train();
    await mnistTester.evaluate();
    console.table(mnistTester.prettyConfusionMatrix());
    console.log(`${id} Accuracy: `, mnistTester.accuracy() + "%");

    console.log(`${id} Benchmark: `, mnistTester.benchmarkResults());

    let onePrediction = await mnistTester.predictOne(`${__dirname}/example_dataset/0/img_5.jpg`);
    console.log(`${id} Predict One Prediction: `, onePrediction);
}