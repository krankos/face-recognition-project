import { createSignal } from "solid-js";

import For from "solid-js/web";

// import many from "../assets/many.jpeg";

export function Prediction() {
  const [predictions, setPredictions] = createSignal(null);
  const [resultImage, setResultImage] = createSignal(null);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal(null);

  const getPrediction = async () => {
    setLoading(true);
    setError(null);
    try {
      // load image from local file to send to the server
      const image = await loadImage(many);
      // send image to the server
      const response = await fetch("127.0.0.1:8000/api/predict", {
        method: "POST",
        body: image,
      });
      // get the prediction from the server
      const data = await response.json();
      console.log(data);
      setPredictions(data.predictions);
      setResultImage(data.result_image);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    // <div className="prediction">
    //   <h1>Prediction</h1>
    //   <button onClick={getPrediction}>Predict</button>
    //   {loading && <p>Loading...</p>}
    //   {error && <p>{error.message}</p>}
    //     {predictions() && (
    //         <div className="predictions">
    //             <h2>Predictions</h2>
    //             <For each={predictions()}>
    //                 {prediction => (
    //                     <div className="prediction">
    //                         <p>{prediction.label}</p>
    //                         <p>{prediction.score}</p>

    //   {resultImage && <img src={resultImage} alt="result" />}
    // </div>

    <div>
      <button onClick={getPrediction}>Predict</button>
      {loading && <p>Loading...</p>}
      {error && <p>{error.message}</p>}
    </div>
  );
}
