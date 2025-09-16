import React from "react";
import { Link } from "react-router-dom";
import logo from "../assets/Logo.png";
import { colors } from "../theme";
import LanguageSwitcher from "./LanguageSwitcher";

export default function Navbar() {
  return (
    <nav style={{
      background: "white",
      padding: "0.75rem 1rem",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      borderBottom: "1px solid #EEF2F7",
      position: "sticky",
      top: 0,
      zIndex: 10
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <img src={logo} alt="Vitanza Logo" style={{ height: "32px" }} />
        <span style={{ color: colors.primary, fontWeight: "bold", fontSize: "1.25rem" }}>Vitanza</span>
      </div>
      <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
        <Link to="/" style={linkStyle}>Home</Link>
        <Link to="/dashboard" style={linkStyle}>Dashboard</Link>
        <Link to="/reports" style={linkStyle}>Reports</Link>
        <Link to="/community" style={linkStyle}>Community</Link>
        <Link to="/assistant" style={linkStyle}>AI Assistant</Link>
      </div>
      <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
        <button aria-label="Toggle theme" style={iconBtn}>🌓</button>
        <LanguageSwitcher />
        <Link to="/login" style={{ ...ghostBtn, textDecoration: "none", color: "#111827" }}>Sign In</Link>
        <Link to="/signup" style={{ ...solidBtn, textDecoration: "none" }}>Sign Up</Link>
      </div>
    </nav>
  );
}

const linkStyle = {
  color: "#4A5568",
  textDecoration: "none",
  fontWeight: 600,
};

const iconBtn = {
  background: "transparent",
  border: "1px solid #E5E7EB",
  borderRadius: 10,
  padding: "0.35rem 0.5rem",
  cursor: "pointer",
};

const ghostBtn = {
  background: "transparent",
  border: "1px solid #CBD5E1",
  borderRadius: 12,
  padding: "0.5rem 0.9rem",
  fontWeight: 700,
  cursor: "pointer",
};

const solidBtn = {
  background: "#2D9CDB",
  color: "white",
  border: 0,
  borderRadius: 12,
  padding: "0.5rem 0.9rem",
  fontWeight: 700,
  cursor: "pointer",
  boxShadow: "0 8px 18px rgba(45,156,219,0.25)",
};