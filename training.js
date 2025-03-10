async function startTraining(sampleSize){
  for(let i =0 ; i < sampleSize; i++){
    let trainDigit = randomInt(0,9)
    console.log(`Training ${trainDigit}`)
    await network.train(40,trainDigit,0.01,trainDigit)
  }
}

async function testRandom(sampleSize){
  let success =0;
  for(let i =0; i < sampleSize; i++){
    let testDigit = randomInt(0,9)
    console.log(`Testing ${trainDigit}`)
    success += await network.test(40,testDigit,testDigit)
  }
  console.log((success/sampleSize)*(40*sampleSize),"%")
}

function randomInt(min,max){
  let val = Math.round(Math.random()*(max-min)) + min
  return val
}