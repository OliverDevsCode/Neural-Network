async function startTraining(sampleSize){
  console.log(`Training ${trainDigit}`)
  for(let i =0 ; i < sampleSize; i++){
    let trainDigit = randomInt(0,9)
    await network.train(5,trainDigit,0.01,trainDigit)
  }
}

async function testRandom(sampleSize){
  let success =0;
  console.log(`Testing ${trainDigit}`)
  for(let i =0; i < sampleSize; i++){
    let testDigit = randomInt(0,9)
    success += await network.test(1,testDigit,testDigit)
  }
  console.log((success))
}

function randomInt(min,max){
  let val = Math.round(Math.random()*(max-min)) + min
  return val
}