import { createSignal } from "solid-js";

function UploadImage() {
  const [uploadedImage, setUploadedImage] = createSignal(null);

  const [predictions, setPredictions] = createSignal(null);
  const [resultImage, setResultImage] = createSignal(null);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal(null);

  function setImage(responseAsBlob) {
    const objectURL = URL.createObjectURL(responseAsBlob);
    console.log(objectURL);
    return objectURL;
  }

  const getPrediction = async () => {
    setLoading(true);
    setError(null);
    console.log("sending request");
    try {
      // send image to the server
      const formData = new FormData();
      formData.append("file", uploadedImage());
      const response = await fetch("http://127.0.0.1:8000/api/predict", {
        method: "POST",
        body: formData,
      });
      // get the prediction from the server
      const data = await response.json();
      console.log("data", data);
      console.log("data.result", typeof data.result_image);
      console.log("image: ", setImage(await data.result_image.blob()));
      setPredictions(data.predictions);
      setResultImage(setImage(await data.result_image.blob()));
    } catch (err) {
      setError(err);
      console.log(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Upload and Display Image usign React Hook's</h1>
      {uploadedImage() && (
        <div>
          <img
            alt="not found"
            width={"250px"}
            src={URL.createObjectURL(uploadedImage())}
          />
          <br />
          <button onClick={() => setUploadedImage(null)}>Remove</button>
          <br />
          <button onClick={() => getPrediction()}>Predict</button>
          {loading && <p>Loading...</p>}
          {error && <p>{error.message}</p>}
          {resultImage() && (
            <div>
              <img alt="not found" width={"250px"} src={resultImage()} />
            </div>
          )}
        </div>
      )}
      <br />

      <br />
      <input
        type="file"
        name="myImage"
        onChange={(event) => {
          // console.log(event.target.files[0]);
          setUploadedImage(event.target.files[0]);
        }}
      />
    </div>
  );
}
export default UploadImage;
