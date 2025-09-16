import React from "react";

export default function Footer() {
  return (
    <footer style={{ padding: "2rem 1rem", textAlign: "center", color: "#6B7280", marginTop: 24 }}>
      <div>© {new Date().getFullYear()} Vitanza. Built with ❤️ for global health monitoring.</div>
      <div style={{ fontSize: 12, marginTop: 6 }}>Built with v0</div>
    </footer>
  );
}