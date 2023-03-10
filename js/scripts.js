var detector;
const SCORESET = 0.5;
async function init() {
    const detectorConfig = {
        modelType: poseDetection.movenet.modelType.MULTIPOSE_THUNDER,
        enableTracking: true,
        trackerType: poseDetection.TrackerType.BoundingBox
    };
    localStorage.clear();
    // init tensorflow
    await tf.setBackend('webgl');
    await tf.enableProdMode();
    await tf.ENV.set('DEBUG', false);
    await tf.ready();
    detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
    document.getElementById("webcam").style.display = "none";
}
var flag = true;
var timearr = [];
async function myfunction(video) {
    if (flag) {
        const poses = await detector.estimatePoses(video);
        if (poses[0] != [] && poses[0] != undefined) {
            var myObject = {
                keypoints: poses[0].keypoints
            }
            localStorage.setItem(video.currentTime.toString(), JSON.stringify(myObject));
            timearr.push(video.currentTime);
        }
    }
}

let myReq;
function addVideo() {
    const inputFile = document.getElementById("input");
    const video = document.getElementById("video");

    inputFile.addEventListener("change", function () {
        const file = inputFile.files[0];
        const videourl = URL.createObjectURL(file);
        video.setAttribute("src", videourl);
        document.getElementById("video").style.display = "none";
        const loading = document.getElementById("loading");
        loading.style.display = "inline-block";
        video.addEventListener('loadeddata', function () {
            video.play().then(draw);
        });
    })
}
function findTime() {
    const video = document.getElementById("video");
    var goal = video.currentTime;
    var closest = timearr.reduce(function (prev, curr) {
        return (Math.abs(curr - goal) < Math.abs(prev - goal) ? curr : prev);
    });
    let minus = timearr.map((v) => {
        return Math.abs(v - closest); 
      }); 
    let second = minus.sort()[1];
    //console.log("hhhhhhhhh",closest);
    //console.log("hhhhhhhhhhh", (Math.abs(timearr - closest)));
    //console.log("hhhhhhhhhhh", (Math.abs(timearr - closest)).sort(function(a, b) { return b - a; })[1]);
    return [closest, second];
}

function turnonVideo() {
    const video = document.getElementById("video");
    if (video.videoHeight > 360){
        video.height = (window.screen.availHeight - (window.outerHeight - window.innerHeight))/2; 
        //video.width = video.videoWidth/2;
    }
    video.style.display = "inline";
}

const bodyParts = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'];

function draw() {
    //ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
    myfunction(video).then(requestAnimationFrame(draw));
}

async function drawSkeleton(element, keypoints, oks_score, width, height, booldraw) {
    var canvas = document.getElementById(element);
    var ctx = canvas.getContext("2d");
    const webcam = document.getElementById("webcam");
    ctx.canvas.width = width;
    ctx.canvas.height = height;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    if (booldraw) {
        ctx.drawImage(webcam, 0, 0,
            webcam.videoWidth,
            webcam.videoHeight);
    }
    ctx.lineWidth = 10;
    async function connectParts(parts, color) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        for (let i = 0; i < parts.length; i++) {
            const part = keypoints.find((a) => a.name == parts[i] && a.score > SCORESET);

            if (part) {
                if (i === 0) {
                    ctx.moveTo(part.x, part.y);
                }
                else {
                    ctx.lineTo(part.x, part.y);
                }
            }
        }
        ctx.stroke();
    }

    let c = new Array(6).fill('#F55050'); //'#F55050' '#0081B4'
    oks_score.forEach(function (e, i) {
        if (e > 0.8) {
            c[i] = '#0081B4';
        }
    })
    //console.log(c);
    await connectParts(['nose', 'left_eye', 'right_eye', 'nose'], c[0]);
    await connectParts(['right_shoulder', 'right_elbow', 'right_wrist'], c[1]);
    await connectParts(['left_shoulder', 'left_elbow', 'left_wrist'], c[2]);
    await connectParts(['right_hip', 'right_knee', 'right_ankle'], c[3]);
    await connectParts(['left_hip', 'left_knee', 'left_ankle'], c[4]);
    await connectParts(['right_shoulder', 'left_shoulder', 'left_hip', 'right_hip', 'right_shoulder'], c[5]);
}

function div(a, b) {
    return a.map((e, i) => e / b[i]);
}

function normalize(kps) {
    // find between left and right eye
    var norm = Math.sqrt((kps[1].x - kps[2].x) ** 2 + (kps[1].y - kps[2].y));
    return norm
}

function parallel(kps) {
    // find between left and right eye
    var itp = {
        x: (kps[1].x + kps[2].x) / 2,
        y: (kps[1].y + kps[2].y) / 2
    };
    return itp
}

function setPlaySpeed() {
    const video = document.getElementById("video");
    video.playbackRate = Number(document.getElementById("speed").value);
}

function calvec(kps, parts) {
    var vecarr = [];
    for (let i = 0; i < parts.length; i++) {
        const part = kps.find((a) => a.name == parts[i]);
        if (part) {
            if (i === 0) {
                var prevec = [part.x, part.y];
            }
            else if (prevec == undefined || prevec == [] || part.score < SCORESET)
            {
                var vec = [NaN, NaN];
                vecarr.push(vec);
            }
            else {
                var vec = [part.x - prevec[0], part.y - prevec[1]];
                var prevec = [part.x, part.y];
                vecarr.push(vec);
            }
        }
    }
    return vecarr;
}

function extractvector(kps) {

    const face = ['nose', 'left_eye', 'right_eye', 'nose'];
    const right_arm = ['right_shoulder', 'right_elbow', 'right_wrist'];
    const left_arm = ['left_shoulder', 'left_elbow', 'left_wrist'];
    const right_leg = ['right_hip', 'right_knee', 'right_ankle'];
    const left_leg = ['left_hip', 'left_knee', 'left_ankle'];
    const body = ['right_shoulder', 'left_shoulder', 'left_hip', 'right_hip', 'right_shoulder'];
    const sixpart = [face, right_arm, left_arm, right_leg, left_leg, body];
    var vecarr = [];
    for (let i = 0; i < sixpart.length; i++) {
        vecarr.push(calvec(kps, sixpart[i]));
    }
    return vecarr;
}

function cosinesim(A, B) {
    var dotproduct = 0;
    var mA = 0;
    var mB = 0;
    for (let i = 0; i < A.length; i++) { // here you missed the i++
        dotproduct += (A[i] * B[i]);
        mA += (A[i] * A[i]);
        mB += (B[i] * B[i]);
    }
    mA = Math.sqrt(mA);
    mB = Math.sqrt(mB);
    var similarity = (dotproduct) / ((mA) * (mB)) // here you needed extra brackets
    return similarity;
}

async function calsim(kps1, kps2) {
    var A = extractvector(kps1);
    var B = extractvector(kps2);

    var similarity = new Array(6).fill(0);
    for (let i = 0; i < A.length; i++) {
        var vec1 = A[i];
        var vec2 = B[i];
        var vec1len = vec1.length;
        for (let j = 0; j < vec1.length; j++) {
            if (vec1[j]!=NaN && vec2[j]!=NaN){
                similarity[i]+=(cosinesim(vec1[j], vec2[j]));
            }
            else{
                similarity[i] += 0;
                vec1len -=1;
            }
        }
        similarity[i] = similarity[i]/vec1len;
    }
    
    return similarity;
}

function oks(kps1, kps2) {

    const KAPPA = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089];
    const SCALE = 1;

    var numerator = new Array(6).fill(0);
    var denominator = new Array(6).fill(0);

    const face = ['nose', 'left_eye', 'right_eye', 'nose'];
    const right_arm = ['right_shoulder', 'right_elbow', 'right_wrist'];
    const left_arm = ['left_shoulder', 'left_elbow', 'left_wrist'];
    const right_leg = ['right_hip', 'right_knee', 'right_ankle'];
    const left_leg = ['left_hip', 'left_knee', 'left_ankle'];
    const body = ['right_shoulder', 'left_shoulder', 'left_hip', 'right_hip', 'right_shoulder'];
    const sixpart = [face, right_arm, left_arm, right_leg, left_leg, body];

    for (let i = 0; i < kps1.length; i++) {
        var kp1 = kps1[i];
        var kp2 = kps2[i];
        var kp1x = kp1.x;
        var kp1y = kp1.y;
        var kp2x = kp2.x;
        var kp2y = kp2.y;
        // var kp1x = (kp1.x-itp1.x)/norm1;
        // var kp1y = (kp1.y-itp1.y)/norm1;
        // var kp2x = (kp2.x-itp2.x)/norm2;
        // var kp2y = (kp2.y-itp2.y)/norm2;

        var distances = Math.sqrt((kp1x - kp2x) ** 2 + (kp1y - kp2y) ** 2);
        exp_vector = Math.exp(-(distances ** 2) / (2 * (SCALE ** 2) * (KAPPA[i] ** 2)))
        for (let j = 0; j < sixpart.length; j++) {
            if (sixpart[j].includes(kp1.name)) {
                numerator[j] = numerator[j] + (exp_vector * Number(kp1.score > SCORESET));
                denominator[j] = denominator[j] + Number(kp1.score > SCORESET);
            };
        }
    }
    return div(numerator, denominator);

}

async function translate(kps, width, height) {
    var norm = normalize(kps);
    var itp = parallel(kps);
    for (let i = 0; i < kps.length; i++) {
        var kp = kps[i];
        kp.x = width / 30 * (kp.x - itp.x) / norm + width / 2;
        kp.y = width / 30 * (kp.y - itp.y) / norm + height / 2;
    }
}

class App {
    constructor() {
        const webcam = document.querySelector("#webcam");
        if (navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then((stream) => { // function ??? this??? ????????? ????????? this ??? ??????
                    webcam.srcObject = stream;
                })
                .catch(function (error) {
                    console.log("Something went wrong!");
                    console.log(error);
                    return;
                });
        }

        webcam.addEventListener("loadedmetadata", () => {
            window.requestAnimationFrame(this.draw.bind(this));
        });
    }

    async draw(t) {
        const myReq = window.requestAnimationFrame(this.draw.bind(this));
        const webcam = document.getElementById("webcam");
        const video = document.getElementById("video");
        webcamRun();
        async function webcamRun() {
            var closeTime = findTime();
            var dbkps = JSON.parse(localStorage.getItem(closeTime[0].toString()));
            
            if (dbkps != null && dbkps != undefined) {
                var dbkps2 = JSON.parse(localStorage.getItem(closeTime[1].toString()));
                if (dbkps2 != null && dbkps2 != undefined) {
                await translate(dbkps2.keypoints, 240, 240);
                await drawSkeleton("dancereal", dbkps2.keypoints, [1, 1, 1, 1, 1, 1], 240, 240, false);
                }
                await translate(dbkps.keypoints, 240, 240);
                await drawSkeleton("dancedata", dbkps.keypoints, [1, 1, 1, 1, 1, 1], 240, 240, false);
                
                
                const poses = await detector.estimatePoses(webcam);
                if (poses[0] != [] && poses[0] != undefined) {
                    //????????????
                    await drawSkeleton("mirrored", poses[0].keypoints, [1, 1, 1, 1, 1, 1], webcam.videoWidth, webcam.videoHeight, true)
                    //normalize
                    //var oks_score = oks(poses[0].keypoints, dbkps.keypoints);
                    await translate(poses[0].keypoints, 240, 240);
                    var similarity = await calsim(poses[0].keypoints, dbkps.keypoints);
                    await drawSkeleton("dancereal", poses[0].keypoints, similarity, 240, 240, false);
                }
            }
            // else {
            //     const poses = await detector.estimatePoses(webcam);
            //     if (poses[0] != [] && poses[0] != undefined) {
            //         const oks_score = [1, 1, 1, 1, 1, 1];
            //         await drawSkeleton("mirrored", poses[0].keypoints, oks_score, webcam.videoWidth, webcam.videoHeight, true);
            //     }
            //     //cancelAnimationFrame(myReq);
            // }
        }
    }
}

window.addEventListener('DOMContentLoaded', event => {

    // Activate Bootstrap scrollspy on the main nav element
    const mainNav = document.body.querySelector('#mainNav');
    if (mainNav) {
        new bootstrap.ScrollSpy(document.body, {
            target: '#mainNav',
            offset: 74,
        });
    };

    // Collapse responsive navbar when toggler is visible
    const navbarToggler = document.body.querySelector('.navbar-toggler');
    const responsiveNavItems = [].slice.call(
        document.querySelectorAll('#navbarResponsive .nav-link')
    );
    responsiveNavItems.map(function (responsiveNavItem) {
        responsiveNavItem.addEventListener('click', () => {
            if (window.getComputedStyle(navbarToggler).display !== 'none') {
                navbarToggler.click();
            }
        });
    });

});


var input = document.getElementById("input");
   var initLabel = document.getElementById("label");
 
   input.addEventListener("change", (event) => {
      const files = changeEvent(event);
      handleUpdate(files);
   });
 
   initLabel.addEventListener("mouseover", (event) => {
      event.preventDefault();
      const label = document.getElementById("label");
      label?.classList.add("label--hover");
   });
 
   initLabel.addEventListener("mouseout", (event) => {
      event.preventDefault();
      const label = document.getElementById("label");
      label?.classList.remove("label--hover");
   });
 
   document.addEventListener("dragenter", (event) => {
      event.preventDefault();

      if (event.target.className === "inner") {
         event.target.style.background = "#616161";
      }
   });
 
   document.addEventListener("dragover", (event) => {

    event.preventDefault();
   });
 
   document.addEventListener("dragleave", (event) => {
      event.preventDefault();

      if (event.target.className === "inner") {
         event.target.style.background = "#3a3a3a";
      }
   });
 
   document.addEventListener("drop", (event) => {
      event.preventDefault();

      if (event.target.className === "inner") {
         const files = event.dataTransfer?.files;
         event.target.style.background = "#3a3a3a";
         handleUpdate([...files]);
         const file = files[0];
        const videourl = URL.createObjectURL(file);
        video.setAttribute("src", videourl);
        document.getElementById("video").style.display = "none";
        const loading = document.getElementById("loading");
        loading.style.display = "inline-block";
        video.addEventListener('loadeddata', function () {
            video.play().then(draw);
        });
      }
   });
 
   function changeEvent(event) {
      const { target } = event;
      return [...target.files];
   };
 
   function handleUpdate(fileList) {
 
      fileList.forEach((file) => {
         const reader = new FileReader();
         reader.readAsDataURL(file);
      });
   };
 
   function el(nodeName, attributes, ...children) {
      const node =
         nodeName === "fragment"
            ? document.createDocumentFragment()
            : document.createElement(nodeName);
 
      Object.entries(attributes).forEach(([key, value]) => {
         if (key === "events") {
            Object.entries(value).forEach(([type, listener]) => {
               node.addEventListener(type, listener);
            });
         } else if (key in node) {
            try {
               node[key] = value;
            } catch (err) {
               node.setAttribute(key, value);
            }
         } else {
            node.setAttribute(key, value);
         }
      });
 
      children.forEach((childNode) => {
         if (typeof childNode === "string") {
            node.appendChild(document.createTextNode(childNode));
         } else {
            node.appendChild(childNode);
         }
      });
 
      return node;
   }

document.addEventListener("DOMContentLoaded", () => {
    tf.tidy(() => {
        init().then(addVideo());
    });
})
const video = document.getElementById("video");
const loading = document.getElementById("loading");
video.addEventListener('ended', (event) => {
    if(flag){
        loading.style.display = "none";
        turnonVideo();
        new App();
        //findTime();
        location.href = "#about";
    }
    flag = false;
    // document.getElementById("myBtn").addEventListener("click", function () {
    //     turnonVideo();
    //     new App();
    //     findTime();
    // });
    document.getElementById("submit").addEventListener("click", function () {
        setPlaySpeed();
    });
    //video.currentTime = 0;
    //video.play();
});