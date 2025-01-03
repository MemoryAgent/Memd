import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router";
import "@/App.css";
import Chat from "@/chat";
import Graph from "./graph";
import Layout from "./layout";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <React.StrictMode>
        <BrowserRouter>
            <Layout>
                <Routes>
                    <Route path="/chat" element={<Chat />} />
                    <Route path="/graph" element={<Graph />} />
                </Routes>
            </Layout>
        </BrowserRouter>
    </React.StrictMode>
);
