import React from "react";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Dashboard from "./pages/Dashboard";

function App() {
  return (
    <div style={{ background: "#F7F9FA", minHeight: "100vh" }}>
      <Navbar />
      <main>
        <Dashboard />
      </main>
      <Footer />
    </div>
  );
}

export default App;