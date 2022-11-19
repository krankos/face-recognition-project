import logo from "./logo.svg";
import styles from "./App.module.css";

// import Prediction from "./components/Prediction";
import HelloWorld from "./components/HelloWorld";
import UploadImage from "./components/UploadImage";

function App() {
  return (
    <div class={styles.App}>
      <header class={styles.header}>
        <HelloWorld />
        <UploadImage />
        {/* <Prediction /> */}
        <img src={logo} class={styles.logo} alt="logo" />
        <p>
          Edit <code>src/App.jsx</code> and save to reload.
        </p>
        <a
          class={styles.link}
          href="https://github.com/solidjs/solid"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn Solid
        </a>
      </header>
    </div>
  );
}

export default App;
