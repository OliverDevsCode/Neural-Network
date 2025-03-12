let dataset;
let network;
let existing_weights;

function preload(){
  dataset = loadImage('dataset/0/0.png');
  existing_weights = loadJSON('weights.json');
}

let trainDigit = 1;
let testDigit = 1;

async function setup() {
  createCanvas(400, 400);
  createCanvas(28, 28);
  pixelDensity(1);
  background(255);
  image(dataset, 0, 0, 28, 28)
  network = new NeuralNetwork(784,16,16,10)
  network.loadWeights(existing_weights)
  console.log("Training....")
  // await network.train(15000,0.10)//learning usually 0.01
  // network.exportWeights()
  console.log("Testing....")
  // await network.test(1000,testDigit,testDigit)
  // await startTraining(100);
  await testRandom(5000)

  // await network.test(1,0,'mineDigits')

}

function draw() {
  // background(220);
  // background(255);
  noSmooth()
  image(dataset, 0, 0, 28, 28)
}

