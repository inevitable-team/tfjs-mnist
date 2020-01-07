const sharp = require("sharp"), fs = require("fs"), datasetWrapper = require("./datasetWrapper");

class mnist {
    async run() {
        await this.setup();
        await this.train();
        await this.evaluate();
        let results = this.benchmarkResults();
        return {
            setUpTime: results.setUpTime,
            trainTime: results.trainTime,
            evaluateTime: results.evaluateTime,
            confusionMatrix: results.confusionMatrix,
            accuracy: results.accuracy
        };
    }

    async predictOne(imageBuffer) {
        if (this.trained) {
            let imageTensorData = await this._generateTensorData(this.classes, [{ location: imageBuffer }]);
            let results = this.model.predict(imageTensorData.xs);
            let argMax = results.argMax(1);
            let predictedIndex = argMax.dataSync()[0];
            return this.classes[predictedIndex];
        } else { throw new Error("Model needs to be trained before it can predict!"); }
    }

    // MODIFY 

    constructor(config) {
        this.tf = config.tf || require("@tensorflow/tfjs-node"); // Optional: TF, enables the gpu package to be passed in
        this.onlyTesting = config.onlyTesting || false; // Optional: Boolean, true if you want to test via other means and use the "predictOne" function
        this.imageLimiter = config.imageLimiter || false; // Optional: Number, % of images to use, 0.9 turns 100 images to 90 images to use (then being split into training and testing data)
        this.split = config.split || 0.75; // Optional: Float, vary the difference in training and testing data, 0.75 = 75% of the images will be used for training
        this.oldModelImageSize = config.modelImageSize || 28;  // Optional: Number, specify the input width/height of the old model
        this.modelImageShape = [this.oldModelImageSize, this.oldModelImageSize, 1];  // Using the input size to get the shape
        this.imagesUrl = config.imagesUrl || `${__dirname}/example_dataset`;  // Optional: String, specify the location of where the source folder is of the images
        this.lossFunction = config.lossFunction || 'categoricalCrossentropy';  // Optional: String, loss function for the models training phase
        this.optimizer = config.optimizer || 'rmsprop';  // Optional: tf.train / String, optimizer for the models training phase
        this.epochs = config.epochs || 5;  // Optional: Number, specify the amount of epoches to be run during the training phase
        this.batchSize = config.batchSize || 8; // Optional: Number, specify the size of batchs to be run during the training phase
        // Internal Values
        this.classes = null;
        this.trained = false;
        this.confusionMatrix = null;
        this.trainingImageTensorData = null;
        // Benchmarking Values
        this.setUpTimeSecs = null;
        this.trainTimeSecs = null;
        this.evaluateTimeSecs = null;
    }

    async setup() {
        let setUpStart = new Date();
        this.getImageData();
        await this.getTrainingImages();
        this.generateModel();
        this.setUpTimeSecs = (new Date() - setUpStart) / 1000;
    }

    async train() {
        if (this.classes == null) {
            await this.setup();
        }   await this.trainModel();
        return null;
    }

    getImageData() {
        if (fs.existsSync(this.imagesUrl)) {
            let sourceFolderFileArr = fs.readdirSync(this.imagesUrl);
            if (sourceFolderFileArr.includes("training") && sourceFolderFileArr.includes("testing") && sourceFolderFileArr.length == 2) { // Already split into training / testing data
                // Get Image Meta and Check Class Names Match
                let trainingImagesData = this._getImages(`${this.imagesUrl}/training`), testingImagesData = this._getImages(`${this.imagesUrl}/testing`);
                if (JSON.stringify(trainingImagesData.classes) != JSON.stringify(testingImagesData.classes)) throw new Error('Classes file name missmatch in "training" and "testing" Folder!');
                this.classes = trainingImagesData.classes;
                // Get Training Images
                this.trainingData = this._shuffleArray(trainingImagesData.images);
                // Get Testing Images
                this.testingData = this._shuffleArray(testingImagesData.images);
            } else {
                let imagesData = this._getImages(this.imagesUrl);
                let imageMetaData = this._shuffleArray(imagesData.images);
                this.classes = imagesData.classes;
                if (this.onlyTesting) { // Do not split into training data
                    this.trainingData = imageMetaData;
                } else {
                    this.trainingData = imageMetaData.slice(0, Math.floor(imageMetaData.length * this.split));
                    this.testingData = imageMetaData.slice(Math.ceil(imageMetaData.length * this.split), imageMetaData.length);
                }
            }   this._limitImageData();
            // If root folder doesn't exist
        } else { throw new Error('Filepath not found, please update the "imagesUrl" to the correct filepath.'); }
    }

    async getTrainingImages() {
        if (this.classes != null) {
            this.trainingImageTensorData = await this._generateTensorData(this.classes, this.trainingData);
        } else { throw new Error("Setup needs to be performed to get training images!"); }
        return null;
    }

    generateModel() {
        if (this.classes != null) {
            this.model = this._createModel(this.classes.length);
        } else { throw new Error("Setup needs to be performed to generate a model!"); }
    }

    async trainModel() {
        if (this.model != undefined && this.trainingImageTensorData != null) {
            let trainStart = new Date();
            this.trainingHistory = await this.model.fit(this.trainingImageTensorData.xs, this.trainingImageTensorData.ys, { batchSize: this.batchSize, epochs: this.epochs });
            this.trainTimeSecs = (new Date() - trainStart) / 1000;
            this.trainingImageTensorData.xs.dispose();
            this.trainingImageTensorData.ys.dispose();
            this.trained = true;
        } else { throw new Error("Model needs to be generated before it can be trained!"); }
        return null;
    }

    async evaluate() {
        return new Promise(resolve => {
            if (this.trained) {
                if (!this.onlyTesting) {
                    let evaluateStart = new Date();
                    this._eval(this.model, this.testingData, this.classes).then(matrix => {
                        this.evaluateTimeSecs = (new Date() - evaluateStart) / 1000;
                        this.confusionMatrix = matrix;
                        resolve(matrix);
                    });
                } else { throw new Error(`No testing data in order to evaluate the model, try "evaluateFromImageFolder", "evaluateFromImageUrls" or "PredictOne"!`); }
            } else { throw new Error("Model needs to be trained in order to evaluate it!"); }
        });
    }

    prettyConfusionMatrix() {
        if (this.confusionMatrix != null) {
            let matrixObj = {};
            this.classes.forEach((item, i) => { 
                matrixObj[`"${item}" Actual`] = {};
                this.confusionMatrix[i].forEach((prediction, index) => {
                    matrixObj[`"${item}" Actual`][`"${this.classes[index]}" Prediction`] = prediction;
                });
            }); return matrixObj;
        } else { throw new Error("No confusion matrix to fetch!"); }
    }

    accuracy() {
        if (this.confusionMatrix != null) {
            let correct = this.confusionMatrix.reduce((sum, curr, index) => sum += curr[index], 0);
            let total = this.confusionMatrix.reduce((a, b) => a.concat(b)).reduce((a, b) => a + b);
            return parseFloat(((correct / total) * 100).toFixed(2));
        } else { throw new Error("No confusion matrix to fetch!"); }
    }

    benchmarkResults() {
        if (this.trained) {
            return {
                setUpTime: this.setUpTimeSecs,
                trainTime: this.trainTimeSecs,
                evaluateTime: this.evaluateTimeSecs,
                confusionMatrix: this.confusionMatrix,
                confusionMatrixObj: this.prettyConfusionMatrix(),
                accuracy: this.accuracy(),
                allClasses: this.classes,
                trainingImages: this._countClasses(this.classes, this.trainingData),
                totalTrainingImages: this.trainingData.length,
                testingImages: this.testingData ? this._countClasses(this.classes, this.testingData) : {},
                totalTestingImages: this.testingData ? this.testingData.length : 0,
                epochs: this.epochs,
                split: this.split,
                batchSize: this.batchSize,
                optimizer: this.optimizer
            }
        } else {
            console.log("Please train the model before trying to benchmark!");
            return null;
        }
    }

    async predictOneFromImageUrl(imageUrl) {
        if (this.trained) {
            if (fs.existsSync(imageUrl)) {
                let imageTensorData = await this._generateTensorData(this.classes, [{ location: imageUrl }]);
                let results = this.model.predict(imageTensorData.xs);
                let argMax = results.argMax(1);
                let predictedIndex = argMax.dataSync()[0];
                return this.classes[predictedIndex];
            } else { throw new Error("Image does not exist!"); }
        } else { throw new Error("Model needs to be trained before it can predict!"); }
    }

    // Other Functions

    _limitImageData() {
        if (this.imageLimiter) {
            this.trainingData = this._limitClassesByPercentage(this.trainingData, this.classes, this.imageLimiter);
            if (!this.onlyTesting) this.testingData = this._limitClassesByPercentage(this.testingData, this.classes, this.imageLimiter);
        }
    }

    _limitClassesByPercentage(data, classes, percentage) {
        let splitClasses = classes.map(c => data.filter(d => d.model == c));
        let limitedSplitClasses = splitClasses.map(classArr => classArr.splice(0, Math.floor(classArr.length * percentage)));
        let flattered = [].concat.apply([], limitedSplitClasses);
        return this._shuffleArray(flattered);
    }

    _createModel(classesNum) {
        let m = this.tf.sequential();

        m.add(this.tf.layers.conv2d({
            inputShape: this.modelImageShape,
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
        }));

        m.add(this.tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
        }));
        
        m.add(this.tf.layers.maxPooling2d({poolSize: [2, 2]}));

        m.add(this.tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
        }));

        m.add(this.tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
        }));

        m.add(this.tf.layers.maxPooling2d({poolSize: [2, 2]}));
        m.add(this.tf.layers.flatten());
        m.add(this.tf.layers.dropout({rate: 0.25}));
        m.add(this.tf.layers.dense({units: 512, activation: 'relu'}));
        m.add(this.tf.layers.dropout({rate: 0.5}));
        m.add(this.tf.layers.dense({units: classesNum, activation: 'softmax'}));

        m.compile({
            optimizer: this.optimizer,
            loss: this.lossFunction,
            metrics: ['accuracy'],
        });

        return m;
    }

    async _generateTensorData(classes, imageMetas) {
        let dataset = new datasetWrapper(this.tf);

        for (let i = 0; i < imageMetas.length; i++) {
            dataset.addExample(
                this.tf.tensor4d([...await this._imgSrcToBuffer(imageMetas[i].location)].map(this._scaleColourToOne), [1].concat(this.modelImageShape)), 
                classes.map(cat => cat == imageMetas[i].model ? 1 : 0), 
                classes.length
            );
        }

        return { xs: dataset.xs, ys: dataset.ys };
    }

    _scaleColourToOne(value) {
        return (value - 0) * (1 - 0) / (255 - 0) + 0;
    }

    async _imgSrcToBuffer(src) {
        return await sharp(src).resize({
            width: this.oldModelImageSize,
            height: this.oldModelImageSize,
            fit: sharp.fit.fill
        }).removeAlpha().toColourspace('b-w').raw().toBuffer();
    }

    _getImages(sourceFolder) {
        if (fs.existsSync(sourceFolder)) {
            let data = { classes: [], images: [] };
            fs.readdirSync(sourceFolder).forEach((model, i) => {
                // Find Models
                data.classes.push(model);
                // Find Image Examples
                fs.readdirSync(`${sourceFolder}/${model}`)
                .filter(image => image.includes(".jpg") || image.includes(".jpeg") || image.includes(".png"))
                .forEach(image => data.images.push({ modelIndex: i, model: model, location: `${sourceFolder}/${model}/${image}` }));
            }); return data;
        } else {
            return null;
        }
    }

    async _eval(model, testingData, classes) {
        return new Promise(resolve => {
            let matrix = new Array(classes.length).fill(0).map(() => new Array(classes.length).fill(0));
            Promise.all(testingData.map(item => this._generateTensorData(classes, [item]))).then(inputs => {
                testingData.forEach(async (item, i) => {
                    let results = model.predict(inputs[i].xs);
                    let argMax = results.argMax(1);
                    let index = argMax.dataSync()[0];
                    // Adding Result to Confusion Matrix
                    matrix[parseInt(item.modelIndex)][parseInt(index)]++;
                }); resolve(matrix);
            });
        }) 
    }

    _shuffleArray(a) {
        for (let i = a.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [a[i], a[j]] = [a[j], a[i]];
        } return a;
    }

    _allFilesExist(urlsArray) {
        for (let url of urlsArray) {
            if (!fs.existsSync(url)) return false;
        } return true;
    }

    _countClasses(classes, imageData) {
        return classes.map(item => {
            return { class: item, count: imageData.filter(image => image.model == item).length }
        });
    }
}

module.exports = mnist;