const video = document.getElementById("video");

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri(`/models`),
  faceapi.nets.faceLandmark68Net.loadFromUri(`/models`),
  faceapi.nets.faceRecognitionNet.loadFromUri(`/models`),
  faceapi.nets.faceExpressionNet.loadFromUri(`/models`),
  faceapi.nets.ssdMobilenetv1.loadFromUri(`/models`),
  faceapi.nets.mtcnn.loadFromUri(`/models`),
])
  .then(getImages)
  .then(startVideo);

function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    (stream) => (video.srcObject = stream),
    (err) => console.error(err)
  );
}

let labeledFaceDescriptors;
async function getImages() {
  labeledFaceDescriptors = await loadLabeledImages();
}

video.addEventListener("play", () => {
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);
  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions()
      .withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    faceapi.draw.drawDetections(canvas, resizedDetections);
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);

    faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

    const results = resizedDetections.map((d) =>
      faceMatcher.findBestMatch(d.descriptor)
    );
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: result.toString(),
        boxColor: "#27AE60",
      });
      if (drawBox.options.label.includes("Simon")) {
        document.getElementById("alarm").play();
        drawBox.options.boxColor = "#C0392B";
        drawBox.options.drawLabelOptions.backgroundColor = "#C0392B";
      }
      if (drawBox.options.label.includes("Uzi")) {
        document.getElementById("alarm").play();
        drawBox.options.boxColor = "#C0392B";
        drawBox.options.drawLabelOptions.backgroundColor = "#C0392B";
      }
      drawBox.draw(canvas);
    });
  }, 100);
});

function loadLabeledImages() {
  const labels = [
    "Black Widow",
    "Captain America",
    "Captain Marvel",
    "Hawkeye",
    "Jim Rhodes",
    "Thor",
    "Simon",
    "Uzi",
    "Beni",
    "John Cena",
    "Tony Stark",
  ];
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(
          //`https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/${label}/${i}.jpg`
          `./labeled_images/${label}/${i}.jpg`
        );
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}
