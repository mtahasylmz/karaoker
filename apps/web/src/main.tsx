import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App.tsx";
import { Manual } from "./manual/Manual.tsx";
import "./styles.css";

// One extra page, one branch. Not worth a router.
const Root = window.location.pathname.startsWith("/manual") ? Manual : App;

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
