import React from "react";
import logo from "../assets/logo.svg";
import { colors } from "../theme";

export default function Navbar() {
  return (
    <nav style={{
      background: colors.primary,
      padding: "1rem",
      display: "flex",
      alignItems: "center"
    }}>
      <img src={logo} alt="Vitanza Logo" style={{ height: "40px", marginRight: "1rem" }} />
      <span style={{ color: colors.background, fontWeight: "bold", fontSize: "1.5rem" }}>Vitanza</span>
    </nav>
  );
}