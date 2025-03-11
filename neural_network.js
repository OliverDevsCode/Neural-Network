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

  async train(iterations,learningRate){
    this.correct_guesses = 0
    for(let i =0; i < iterations; i++){

        //random number class - label
        let expectedLabel = randomInt(0,9);
        //random number png
        let png_position = randomInt(0,10000);

        //get inputs
        await this.loadnextData(png_position,expectedLabel)

        //first layer * by weights + biases
        this.layer_one_neurons = await this.forward(this.input_neurons,this.layer_one_weights.length,this.layer_one_weights,this.layer_one_biases)

        //first layer to second layer 
        this.layer_two_neurons = await this.forward(this.layer_one_neurons,this.layer_two_weights.length,this.layer_two_weights,this.layer_two_biases)

        //second to final layer
        this.output_neurons = await this.forward(this.layer_two_neurons,this.output_weights.length,this.output_weights,this.output_biases)

        const normalizedOutputs = this.softmax(this.output_neurons);

        const prediction = this.getPrediction(normalizedOutputs);

        // console.log("predicted",prediction)
        if(prediction.index == expectedLabel){
          this.correct_guesses ++
        }

        //learning

        // Compute error (Assume target is available)
        let target = new Array(normalizedOutputs.length).fill(0);
        target[expectedLabel] = 1; // Expected label should be set per dataset

        let error = normalizedOutputs.map((out, i) => out - target[i]);

        // if(iterations == 1 ){
        //   console.log("Output Neurons")
        //   console.table(outputMatrix.matrix)
        //   console.table(errorMatrix.matrix)
        // }

        // Backpropagation MY DESIGN TEST
        this.backpropagation(error,learningRate,target);

        this.output_biases = new Array(this.output_neurons.length).fill(0);
        clear()
        this.input_neurons = []
    }
    console.log(`Correct ${this.correct_guesses}/${iterations}`);
    console.log(JSON.stringify(this.output_neurons));
  }

  async test(iterations,expectedLabel,datasetFrame){
    this.correct_guesses = 0
    for(let i =0; i < iterations; i++){
        //get inputs
        await this.loadnextData(i,datasetFrame)
 
        //first layer * by weights + biases
        this.layer_one_neurons = await this.forward(this.input_neurons,this.layer_one_weights.length,this.layer_one_weights,this.layer_one_biases)

        //first layer to second layer 
        this.layer_two_neurons = await this.forward(this.layer_one_neurons,this.layer_two_weights.length,this.layer_two_weights,this.layer_two_biases)
  
        //second to final layer
        this.output_neurons = await this.forward(this.layer_two_neurons,this.output_weights.length,this.output_weights,this.output_biases)

        const normalizedOutputs = this.softmax(this.output_neurons);

        const prediction = this.getPrediction(normalizedOutputs);

        if(prediction.index == expectedLabel){
          this.correct_guesses ++
        }
        // if(iterations == 1){
        //   console.log("Output Neurons")
        //   console.table(this.output_neurons)
        //   console.log(`Answer: ${prediction.index}, Probability: ${prediction.probability}`)
        // }
        this.output_biases = new Array(this.output_neurons.length).fill(0);
        clear()
        this.input_neurons = []
    }
    console.log(`Correct ${this.correct_guesses}/${iterations}`);
    console.log(JSON.stringify(this.output_neurons));
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

  async backpropagation(errors, learningRate,target) {
  let output_errors = [];
  for(let i =0; i < this.output_neurons.length;i++){
    let t = target[i];
    let activation = this.output_neurons[i];
    let error = (t-activation)*this.derivativeSigmoid(activation);
    output_errors.push(error)
  }
  // console.log('output_errors')
  // console.table(output_errors)//debug

  //calculate hidden layer 2
  let hidden_2_errors = [];
  for(let i =0; i < this.layer_two_neurons.length;i++){
    let activation = this.layer_two_neurons[i]
    let inverse = this.derivativeSigmoid(activation)
    let sum = 0;
    for(let j =0; j < this.output_neurons.length; j++){
      let weight_at_neuron = this.output_weights[j][i];
      let error_at_neuron = output_errors[j];
      let product = weight_at_neuron*error_at_neuron
      sum += product
    };
    let neuron_error = inverse*sum
    hidden_2_errors.push(neuron_error);
  }

  // console.log('layer 2 errors');
  // console.table(hidden_2_errors);//debug

   //calculate hidden layers 1
   let hidden_1_errors = [];
   for(let i =0; i < this.layer_one_neurons.length;i++){
     let activation = this.layer_one_neurons[i]
     let inverse = this.derivativeSigmoid(activation)
     let sum = 0;
     for(let j =0; j < this.layer_two_neurons.length; j++){
       let weight_at_neuron = this.layer_two_weights[j][i];
       let error_at_neuron = hidden_2_errors[j];
       let product = weight_at_neuron*error_at_neuron
       sum += product
     };
     let neuron_error = inverse*sum
     hidden_1_errors.push(neuron_error);
   }
 
  //  console.log('layer 1 errors');
  //  console.table(hidden_1_errors);//debug

   //update weights based on errors
   //output weights

   for (let k = 0; k < this.output_neurons.length; k++) {
    for (let j = 0; j < this.layer_two_neurons.length; j++) {
      let delta = output_errors[k]; // δ for output neuron k
      let activation = this.layer_two_neurons[j]; // activation from hidden layer 2
      let weightDelta = learningRate * delta * activation;
      this.output_weights[k][j] += weightDelta;
    }
    // Optionally update the bias for the output neuron as well
    this.output_biases[k] += learningRate * output_errors[k];
    }

    for (let k = 0; k < this.layer_two_neurons.length; k++) {
      for (let j = 0; j < this.layer_one_neurons.length; j++) {
        let delta = hidden_2_errors[k]; // δ for output neuron k
        let activation = this.layer_one_neurons[j]; // activation from hidden layer 2
        let weightDelta = learningRate * delta * activation;
        this.layer_one_weights[k][j] += weightDelta;
      }
      // Optionally update the bias for the output neuron as well
      this.layer_two_biases[k] += learningRate * hidden_2_errors[k];
      }

      for (let k = 0; k < this.layer_one_neurons.length; k++) {
        for (let j = 0; j < this.input_neurons.length; j++) {
          let delta = hidden_1_errors[k]; // δ for output neuron k
          let activation = this.input_neurons[j]; // activation from hidden layer 2
          let weightDelta = learningRate * delta * activation;
          this.layer_one_weights[k][j] += weightDelta;
        }
        // Optionally update the bias for the output neuron as well
        this.layer_one_biases[k] += learningRate * hidden_1_errors[k];
        }

}

  updateWeight(neuron,weights,learningRate,error,pos){
    let weight_delta = learningRate * error * neuron
    weights[pos][pos] -= weight_delta
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