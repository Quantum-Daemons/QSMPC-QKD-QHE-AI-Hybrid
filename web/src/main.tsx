import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { refreshConfigFromApi } from "./lib/config";
import "./styles.css";

// In full mode, pull the live config (cost model / tiers / weights) from /api/config so a runtime
// config.yaml override flows through; the demo keeps the baked config.defaults.json (zero /api).
void refreshConfigFromApi();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
