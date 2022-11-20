import { createSignal, onCleanup } from "solid-js";

function UploadImage() {
  const [uploadedImage, setUploadedImage] = createSignal(null);

  const [predictions, setPredictions] = createSignal(null);
  const [show, setShow] = createSignal(false);
  const [resultImage, setResultImage] = createSignal(null);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal(null);
  const [noface, setNoFace] = createSignal(false);

  function setImage(responseAsBlob) {
    const objectURL = URL.createObjectURL(responseAsBlob);
    console.log(objectURL);
    return objectURL;
  }

  const getPredictionImage = async () => {
    setLoading(true);
    setError(null);
    console.log("sending request");
    try {
      // send image to the server
      const formData = new FormData();
      formData.append("file", uploadedImage());
      const response = await fetch("http://127.0.0.1:8000/api/predict/image", {
        method: "POST",
        body: formData,
      });
      // get the prediction from the server
      const data = await response;
      const contentType = response.headers.get("content-type");
      console.log("data", data);
      // console.log("data.result", typeof data.result_image);
      // console.log("image: ", setImage(await data.blob()));
      // if data has a message key, then there is an error
      if (contentType && contentType.indexOf("application/json") !== -1) {
        setNoFace(true);
        setResultImage(null);
      } else {
        setNoFace(false);
        setResultImage(setImage(await data.blob()));
      }
      setLoading(false);
    } catch (err) {
      setError(err);
      console.log(err);
    } finally {
      setLoading(false);
    }
  };
  const getPredictionsList = async () => {
    setLoading(true);
    setError(false);
    try {
      const formData = new FormData();
      formData.append("file", uploadedImage());
      const response = await fetch("http://127.0.0.1:8000/api/predict/list", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      console.log("data", data);
      if (data.predictions.length === 0) {
        console.log("no face");
        setNoFace(true);
        setPredictions(null);
      } else {
        setNoFace(false);
        setPredictions(data.predictions);
      }
      setLoading(false);
    } catch (err) {
      setError(err);
      console.log(err);
    } finally {
      setLoading(false);
    }
  };
  const resetInput = (el) => {
    document.getElementById("rm").addEventListener("click", () => {
      el.target.value = null;
      setUploadedImage(null);
      setPredictions(null);
      setResultImage(null);
    });
    onCleanup(() => {
      document.getElementById("rm").removeEventListener("click");
    });
  };

  return (
    <div>
      <h1>Who Are You?</h1>
      {uploadedImage() && (
        <div>
          <img
            alt="not found"
            width={"250px"}
            src={URL.createObjectURL(uploadedImage())}
          />
          <br />
          <button
            id="rm"
            onClick={() => {
              setUploadedImage(null);
              setPredictions(null);
              setResultImage(null);
              setNoFace(null);
              document.getElementById("selectedFile").value = null;
            }}
          >
            Remove
          </button>
          <br />
          <button onClick={() => getPredictionImage()}>Predict</button>

          {resultImage() && (
            <div>
              <img alt="not found" width="500px" src={resultImage()} />
            </div>
          )}

          <button onClick={() => getPredictionsList()}>Predict List</button>
          {noface() && <div>No Face Detected</div>}
          {loading() ? <div>Loading...</div> : null}
          {error() && <p>{error().message}</p>}
          {predictions() && (
            <div>
              <ul>
                {predictions().map((prediction) => (
                  <li>{`${prediction[0]} ${prediction[1]}%`}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
      <br />
      {predictions() ? (
        predictions().length == 1 && predictions()[0][0] == "Khalil" ? (
          <p style={"color:magenta"}>Hello Khalil</p>
        ) : null
      ) : null}
      <br />
      <input
        type="file"
        name="myImage"
        id="selectedFile"
        accept="image/jpeg, image/jpg"
        // style={{ display: "none" }}
        onChange={(event) => {
          // console.log(event.target.files[0]);

          setUploadedImage(event.target.files[0]);
          setNoFace(null);
        }}
      />

      {/* <input
        type="button"
        value="Browse..."
        onclick="document.getElementById('selectedFile').click();"
      /> */}
    </div>
  );
}
export default UploadImage;
