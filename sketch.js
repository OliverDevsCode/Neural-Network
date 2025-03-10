let dataset;
let network;

function preload(){
  dataset = loadImage('dataset/0/0.png')
}

let trainDigit = 1;
let testDigit = 1;

async function setup() {
  createCanvas(28, 28);
  pixelDensity(1);
  background(255);
  image(dataset, 0, 0, 28, 28)
  network = new NeuralNetwork(784,16,16,10)
  console.log("Training....")
  await network.train(1000,trainDigit,0.01,trainDigit)
  console.log("Testing....")
  await network.test(100,testDigit,testDigit)
  // await network.test(1,7,'mineDigits')

}

function draw() {
  // background(220);
  // background(255);
  image(dataset, 0, 0, 28, 28)
}

