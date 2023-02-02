

const MODEL_PATH = 'https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4';
const EXAMPLE_IMG= new Image();
EXAMPLE_IMG.src = './hello.png';
let movenet = undefined;

async function loadAndRunModel() {
    movenet = await tf.loadGraphModel(MODEL_PATH, {fromTFHub: true});

    let exampleInputTensor = tf.zeros([1,192,192,3],'int32');
    let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG);
    console.log(imageTensor.shape);

    let cropStartPoint = [15, 170, 0];
    let cropSize = [345, 345, 3];
    let coppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize);

    let resizedTensor = tf.image.resizeBilinear(croppedTensor, [192,192], true).toInt();
    console.log(resizedTensor.shape);

    let tensorOutput = movenet.predict(tf.expandDims(resizedTensor));
    let arrayOutput = await tensorOutput.array();
    console.log(arrayOutput);
}
loadAndRunModel();
/*
console.log('hi2');
//import * as poseDetection from '@tensorflow-models/pose-detection';
//import * as tf from '@tensorflow/tfjs-core';
//import '@tensorflow/tfjs-backend-webgl';
// Create a detector.
const detector = poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER});
const image = new Image();
image.src = './hello.png';
console.log('hi');
const poses = detector.estimatePoses(image);
//const video = document.getElementById('video');
//const poses = await detector.estimatePoses(video);
//console.log(poses[0].keypoints);
*/