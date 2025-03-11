let dataset;
let network;

function preload(){
  dataset = loadImage('dataset/0/0.png')
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
  console.log("Training....")
  await network.train(1000,0.01)
  console.log("Testing....")
  await network.test(1,testDigit,testDigit)

  // await startTraining(100);
  // await testRandom(100)

  // await network.test(1,0,'mineDigits')

}

function draw() {
  // background(220);
  // background(255);
  image(dataset, 0, 0, 28, 28)
}

