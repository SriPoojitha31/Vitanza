import React from "react";
import { Link } from "react-router-dom";

export default function CommunityDashboard() {
  return (
    <section style={{ padding: "1.5rem" }}>
      <h2>Community Dashboard</h2>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 12 }}>
        <div style={card}><div style={title}>Report Symptoms</div><Link to="/reports">Open</Link></div>
        <div style={card}><div style={title}>Check Water Status</div><Link to="/water">Open</Link></div>
        <div style={card}><div style={title}>Awareness</div><Link to="/community">Open</Link></div>
      </div>
    </section>
  );
}

const card = { background: "white", borderRadius: 16, padding: 16, boxShadow: "0 6px 16px rgba(0,0,0,0.06)" };
const title = { color: "#6B7280", fontSize: 12, marginBottom: 8 };
