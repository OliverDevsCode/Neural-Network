class NeuralNetwork{

  //Neurons
  input_neurons;

  //Activation Neurons /Hidden Layers
  layer_one_neurons;
  layer_two_neurons;

  //Weights
  layer_one_weights;
  layer_two_weights;
  output_weights;

  //biases
  layer_one_biases;
  layer_two_biases;
  output_biases;

  //output
  output_neurons;

  //stats
  correct_guesses;

  constructor(input_size,hiddenSize1, hiddenSize2, outputSize){
    this.layer_one_weights = this.createWeights(hiddenSize1,input_size);
    this.layer_two_weights = this.createWeights(hiddenSize2,hiddenSize1);
    this.output_weights = this.createWeights(outputSize,hiddenSize2);

    this.layer_one_biases = new Array(hiddenSize1).fill(0);
    this.layer_two_biases = new Array(hiddenSize2).fill(0);
    this.output_biases = new Array(outputSize).fill(0);

  }

  async train(iterations,expectedLabel,learningRate,datasetFrame){
    this.correct_guesses = 0
    for(let i =0; i < iterations; i++){
        //get inputs
        await this.loadnextData(i,datasetFrame)
        // console.log("Input Neurons:")//for debug
        // console.log(JSON.stringify(this.input_neurons))//for debug

        //first layer * by weights + biases
        this.layer_one_neurons = await this.forward(this.input_neurons,this.layer_one_weights.length,this.layer_one_weights,this.layer_one_biases)
        // console.log("First Layer Neurons:")//for debug
        // console.log(JSON.stringify(this.layer_one_neurons))//for debug

        //first layer to second layer 
        this.layer_two_neurons = await this.forward(this.layer_one_neurons,this.layer_two_weights.length,this.layer_two_weights,this.layer_two_biases)
        // console.log("Second Layer Neurons:")//for debug
        // console.log(JSON.stringify(this.layer_two_neurons))//for debug

        //second to final layer
        this.output_neurons = await this.forward(this.layer_two_neurons,this.output_weights.length,this.output_weights,this.output_biases)
        // console.log("Output Neurons:")//for debug
        // console.log(JSON.stringify(this.output_neurons))//for debug

        const normalizedOutputs = this.softmax(this.output_neurons);
        // console.log("Output Normalised Neurons:")//for debug
        // console.log(JSON.stringify(normalizedOutputs))//for debug
        const prediction = this.getPrediction(normalizedOutputs);
        // console.log("Predicted class:", prediction.index, "with probability:", prediction.probability);

        // console.log("predicted",prediction)
        if(prediction.index == expectedLabel){
          this.correct_guesses ++
        }

        //learning

        // Compute error (Assume target is available)
        let target = new Array(normalizedOutputs.length).fill(0);
        target[expectedLabel] = 1; // Expected label should be set per dataset

        let error = normalizedOutputs.map((out, i) => out - target[i]);

        const outputMatrix = new Matrix(this.output_neurons.length,1,this.output_neurons);
        const targetMatrix = new Matrix(target.length,1,target)
        const errorMatrix = new Matrix(this.output_neurons.length,1,subtract(targetMatrix,outputMatrix))
        if(i == iterations-1){
          console.log("Output Neurons")
          console.table(outputMatrix.matrix)
          console.table(errorMatrix.matrix)
        }

        // Backpropagation (Basic Gradient Descent)
        this.updateWeights(this.output_weights, errorMatrix.matrix, this.layer_two_neurons, learningRate);
        // this.updateWeights(this.layer_two_weights, error, this.layer_one_neurons, learningRate);
        // this.updateWeights(this.layer_one_weights, error, this.input_neurons, learningRate);

        // Backpropagation MY DESIGN TEST

        this.output_biases = new Array(this.output_neurons.length).fill(0);
        clear()
        this.input_neurons = []
    }
    console.log(`Correct ${this.correct_guesses}/${iterations}`);
    console.log(JSON.stringify(this.softmax(this.output_neurons)));
  }

  async test(iterations,expectedLabel,datasetFrame){
    this.correct_guesses = 0
    for(let i =0; i < iterations; i++){
        //get inputs
        await this.loadnextData(i,datasetFrame)
        // console.log("Input Neurons:")//for debug
        // console.log(JSON.stringify(this.input_neurons))//for debug

        //first layer * by weights + biases
        this.layer_one_neurons = await this.forward(this.input_neurons,this.layer_one_weights.length,this.layer_one_weights,this.layer_one_biases)
        // console.log("First Layer Neurons:")//for debug
        // console.log(JSON.stringify(this.layer_one_neurons))//for debug

        //first layer to second layer 
        this.layer_two_neurons = await this.forward(this.layer_one_neurons,this.layer_two_weights.length,this.layer_two_weights,this.layer_two_biases)
        // console.log("Second Layer Neurons:")//for debug
        // console.log(JSON.stringify(this.layer_two_neurons))//for debug

        //second to final layer
        this.output_neurons = await this.forward(this.layer_two_neurons,this.output_weights.length,this.output_weights,this.output_biases)
        // console.log("Output Neurons:")//for debug
        // console.log(JSON.stringify(this.output_neurons))//for debug

        const normalizedOutputs = this.softmax(this.output_neurons);
        // console.log("Output Normalised Neurons:")//for debug
        // console.log(JSON.stringify(normalizedOutputs))//for debug
        const prediction = this.getPrediction(normalizedOutputs);
        // console.log("Predicted class:", prediction.index, "with probability:", prediction.probability);

        // console.log("predicted",prediction)
        if(prediction.index == expectedLabel){
          this.correct_guesses ++
        }
        if(iterations == 1){
          console.log("Output Neurons")
          console.table(this.output_neurons)
          console.log(`Answer: ${prediction.index}, Probability: ${prediction.probability}`)
        }
        this.output_biases = new Array(this.output_neurons.length).fill(0);
        clear()
        this.input_neurons = []
    }
    console.log(`Correct ${this.correct_guesses}/${iterations}`);
    console.log(JSON.stringify(this.softmax(this.output_neurons)));
    return this.correct_guesses
  }

  async forward(neurons,neurons_2,weights,biases){
    let weightedNeurons = [];
    for(let i =0; i < neurons_2;i++){
      let sum = 0;
      for(let j =0; j < neurons.length;j++){
        let current_neuron = (neurons[j]*weights[i][j])
        sum += current_neuron
      }
      sum = this.Sigmoid(sum + biases[i]); 
      weightedNeurons.push(sum)
    }
    return weightedNeurons
  }

  async backpropagation(neuron,next_neurons,errors,learningRate){

  }

    /**
   * Adjusts weights based on error rates.
   * @param {array} outputs output neurons array
   * @param {array} errors loss calculation array
   */


  updateWeights(weights, error, previousLayerNeurons, learningRate) {
    // console.log(error)//for debug
    for (let i = 0; i < weights.length; i++) {
        for (let j = 0; j < previousLayerNeurons.length; j++) {
            weights[i][j] += learningRate * error[i] * previousLayerNeurons[j];
        }
    }
}

  getPrediction(outputNeurons) {
    // Find the index of the neuron with the maximum value
    let maxIndex = outputNeurons.indexOf(Math.max(...outputNeurons));
    return { index: maxIndex, probability: outputNeurons[maxIndex] };
  }

  softmax(arr) {
    const max = Math.max(...arr); // For numerical stability
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(exp => exp / sum);
  }

  normaliseOutputs(arr){
    let total =0;
    arr.forEach(activation => {
      total+= activation
    });
    let normalised = []
    arr.forEach(activation => {
      normalised.push(activation/total)
    });
    return normalised
  }

  derivativeSigmoid(val){
    let y = this.Sigmoid(val) * (1-this.Sigmoid(val))
    return y
  }

  Sigmoid(val){
    let y = 1/(1+exp(-val));
    let value = parseFloat(y.toFixed(3));
    return value
  }


  createWeights(hiddenSize,neuron_num){
    let arr = [];
    for(let i =0; i<hiddenSize;i++){
    let weightArr = []
    for(let j=0;j<(neuron_num);j++){
      // let weight = Math.random() * 0.01 Math.round((Math.random()*6))-3
      let weight = Math.round((Math.random()*6))-3
      weightArr.push(weight)
    }
    arr.push(weightArr)
    }
    return arr
  }


  loadnextImage(num,datasetFrame) {
    return new Promise((resolve, reject) => {
      loadImage(`dataset/${datasetFrame}/${num}.png`, (img) => {
        // Assign the image to a property if needed
        dataset = img;
        // Process the image once loaded
        this.processImage();
        resolve();
      }, (err) => {
        console.error("Error fetching file", err);
        reject(err);
      });
    });
  }
  
  async loadnextData (num,datasetFrame){
      await this.loadnextImage(num,datasetFrame)
  }

  processImage(){
    console.log("Processing Image")
    background(255);
    image(dataset, 0, 0, 28, 28)
    loadPixels();
    updatePixels()
    let grayscaleValues = [];
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let c = get(x, y); // Get color at (x, y)
        let grayscale = 1-(((c[0] + c[1] + c[2])/3)/255); 
        grayscaleValues.push(grayscale);
      }
    }
    this.input_neurons = grayscaleValues;
    // console.log(JSON.stringify(this.input_neurons))
    }
}