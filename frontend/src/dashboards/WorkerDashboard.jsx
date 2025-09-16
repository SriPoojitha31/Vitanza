import React from "react";
import { Link } from "react-router-dom";

export default function WorkerDashboard() {
  return (
    <section style={{ padding: "1.5rem" }}>
      <h2>Worker Dashboard</h2>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 12 }}>
        <div style={card}><div style={title}>Submit Health Report</div><Link to="/reports">Go to Reports</Link></div>
        <div style={card}><div style={title}>Submit Water Quality</div><Link to="/water">Go to Water</Link></div>
        <div style={card}><div style={title}>View Alerts</div><Link to="/alerts">Go to Alerts</Link></div>
      </div>
    </section>
  );
}

const card = { background: "white", borderRadius: 16, padding: 16, boxShadow: "0 6px 16px rgba(0,0,0,0.06)" };
const title = { color: "#6B7280", fontSize: 12, marginBottom: 8 };
