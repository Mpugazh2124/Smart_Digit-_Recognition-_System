function predict() {
  const image = canvas.toDataURL("image/png");

  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ image: image })
  })
  .then(res => res.json())
  .then(data => {
    console.log("Response from server:", data);

    if (data.character !== undefined) {
      document.getElementById("result").innerText =
        "Prediction: " + data.character + " (Confidence: " + data.confidence + "%)";
    } else {
      document.getElementById("result").innerText = "Prediction failed.";
    }
  })
  .catch(err => {
    console.error("Prediction error:", err);
    document.getElementById("result").innerText = "Error during prediction.";
  });
}
