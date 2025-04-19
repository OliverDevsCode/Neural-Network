let submit;
let networkGuess;

function userInterface() {
    mode = -1;
    let canvas = createCanvas(28, 28);
    canvas.position(10, 10); // Offset from top-left
    canvas.style('border', '1px solid black'); // Add border around canvas
    pixelDensity(1);
    background(255);

    submit.mousePressed(guessUserDigit);
}



async function guessUserDigit(){
    console.log("Loading User Pixels")
    let result = await network.guessUserDigit()
    networkGuess = createP(`Guess: ${result.index}, Certainity: ${Math.round(result.probability*100)}%`)
    networkGuess.position(0,height + 30)
    console.log(result)
}

function mouseDragged(){
  noSmooth()
    if(mode == -1){
        noSmooth()
        stroke(0);     // Solid black stroke
        strokeWeight(1); // 1 pixel stroke
        point(mouseX, mouseY); // Use point instead of circle for a sharp 1px dot
    }
}