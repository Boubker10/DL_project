let model;
const modelPath = "assets/model.json"; // Path to the TensorFlow.js model
const classes = ['carrot', 'eggplant', 'peas', 'potato', 'sweetcorn', 'tomato', 'turnip']; // Class names

(async function loadModel() {
  try {
    console.log("Loading model...");
    model = await tf.loadGraphModel(modelPath);
    console.log("Model loaded successfully!");
  } catch (error) {
    console.error("Error loading the model:", error);
  }
})();

const dropZone = document.getElementById("drop-zone");
dropZone.addEventListener("dragover", (e) => e.preventDefault());
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  handleImageUpload(file);
});
document.getElementById("image-upload").addEventListener("change", (e) => {
  const file = e.target.files[0];
  handleImageUpload(file);
});

function handleImageUpload(file) {
  if (!file) return;
  const img = new Image();
  img.src = URL.createObjectURL(file);
  img.onload = async () => {
    document.getElementById("output-image-upload").innerHTML = `<img src="${img.src}" alt="Uploaded Image">`;
    const predictions = await makePrediction(img);
    displayPredictions(predictions);
  };
}

async function makePrediction(image) {
  try {
    const tensor = tf.browser
      .fromPixels(image)
      .resizeNearestNeighbor([128, 128])
      .toFloat()
      .div(255.0)
      .sub([0.485, 0.456, 0.406])
      .div([0.229, 0.224, 0.225])
      .expandDims()
      .transpose([0, 3, 1, 2]);

    const logits = await model.predict(tensor).dataSync();
    const probabilities = softmax(Array.from(logits));
    const predictedClass = probabilities.indexOf(Math.max(...probabilities));
    return { probabilities, predictedClass };
  } catch (error) {
    console.error("Error during prediction:", error);
    return null;
  }
}

function softmax(logits) {
  const expScores = logits.map((x) => Math.exp(x));
  const sumExpScores = expScores.reduce((a, b) => a + b, 0);
  return expScores.map((x) => x / sumExpScores);
}

function displayPredictions(predictionData) {
    const predictionOutput = document.getElementById("prediction-output");
    const probabilityBars = document.getElementById("probability-bars");
  
    // Clear previous content
    predictionOutput.innerHTML = "";
    probabilityBars.innerHTML = "";
  
    if (!predictionData) {
      predictionOutput.innerText = "Error in prediction.";
      return;
    }
  
    const { probabilities, predictedClass } = predictionData;
  
    // Predicted class
    predictionOutput.innerHTML = `
      <h4>Predicted Class: <span style="color: #007bff; font-weight: bold;">${classes[predictedClass]}</span></h4>
    `;
  
    // Create unified probability bars
    probabilities.forEach((prob, i) => {
      const isMax = i === predictedClass; // Highlight the max probability
  
      // Bar wrapper
      const barWrapper = document.createElement("div");
      barWrapper.style = `
        display: flex;
        align-items: center;
        margin-bottom: 10px;
      `;
  
      // Class name
      const className = document.createElement("div");
      className.innerText = classes[i].toUpperCase();
      className.style = `
        width: 20%;
        font-size: 14px;
        font-weight: ${isMax ? "bold" : "normal"};
        color: ${isMax ? "#007bff" : "#000"};
      `;
  
      // Bar container
      const barContainer = document.createElement("div");
      barContainer.style = `
        flex-grow: 1;
        height: 20px;
        background-color: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        margin-left: 10px;
        margin-right: 10px;
        position: relative;
      `;
  
      // Filled bar
      const barFill = document.createElement("div");
      barFill.style = `
        width: ${Math.round(prob * 100)}%;
        height: 100%;
        background-color: ${isMax ? "#007bff" : "#6c757d"};
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
      `;
  
      // Append bar fill to container
      barContainer.appendChild(barFill);
  
      // Probability percentage
      const probPercent = document.createElement("div");
      probPercent.innerText = `${Math.round(prob * 100)}%`;
      probPercent.style = `
        width: 10%;
        text-align: right;
        font-size: 14px;
        font-weight: ${isMax ? "bold" : "normal"};
        color: ${isMax ? "#007bff" : "#000"};
      `;
  
      // Append elements
      barWrapper.appendChild(className);
      barWrapper.appendChild(barContainer);
      barWrapper.appendChild(probPercent);
      probabilityBars.appendChild(barWrapper);
    });
  }
  