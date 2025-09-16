import React, { useEffect, useState } from "react";
import { fetchAlerts, fetchHeatmap, sendSmsAlert } from "../services/api";

export default function GISMap() {
  const [points, setPoints] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [form, setForm] = useState({ message: "", severity: "medium", phone: "", lang: "en" });
  useEffect(() => {
    fetchHeatmap().then(setPoints);
    fetchAlerts().then(setAlerts);
  }, []);
  return (
    <section style={{ padding: "1.5rem" }}>
      <h2 style={{ marginBottom: "1rem" }}>Alerts Heatmap</h2>
      <div style={{ height: 360, borderRadius: 16, background: "#E6F0FA", position: "relative" }}>
        {points.map((p, idx) => (
          <div key={idx} title={`Intensity ${p.intensity}`}
            style={{
              position: "absolute",
              left: `${20 + (idx * 20) % 70}%`,
              top: `${20 + (idx * 15) % 60}%`,
              width: 24, height: 24,
              borderRadius: "50%",
              background: `rgba(242, 153, 74, ${0.3 + 0.5 * p.intensity})`,
              boxShadow: `0 0 40px rgba(242, 153, 74, ${0.6 * p.intensity})`
            }} />
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 16 }}>
        <div style={{ background: "white", borderRadius: 16, padding: 16, boxShadow: "0 6px 16px rgba(0,0,0,0.06)" }}>
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Send SMS Alert</div>
          <div style={{ display: "grid", gap: 8 }}>
            <input placeholder="Message" value={form.message} onChange={(e)=>setForm({ ...form, message: e.target.value })} style={input} />
            <select value={form.severity} onChange={(e)=>setForm({ ...form, severity: e.target.value })} style={input}>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
            <input placeholder="Phone" value={form.phone} onChange={(e)=>setForm({ ...form, phone: e.target.value })} style={input} />
            <select value={form.lang} onChange={(e)=>setForm({ ...form, lang: e.target.value })} style={input}>
              <option value="en">English</option>
              <option value="hi">हिंदी</option>
              <option value="te">తెలుగు</option>
            </select>
            <button onClick={async ()=>{ const r = await sendSmsAlert(form); alert(r.sent ? "Sent" : "Failed"); }} style={btn}>Send SMS</button>
          </div>
        </div>
        <div style={{ background: "white", borderRadius: 16, padding: 16, boxShadow: "0 6px 16px rgba(0,0,0,0.06)" }}>
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Recent Alerts</div>
          <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: 8 }}>
            {alerts.map((a, i) => (
              <li key={i} style={{ background: "#F8FAFC", padding: 10, borderRadius: 10 }}>
                <div style={{ fontWeight: 700 }}>{a.severity?.toUpperCase()}</div>
                <div style={{ color: "#6B7280" }}>{a.message}</div>
                <div style={{ fontSize: 12, color: "#9CA3AF" }}>{a.timestamp}</div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}

const input = { padding: 10, borderRadius: 10, border: "1px solid #E5E7EB" };
const btn = { background: "#2D9CDB", color: "white", border: 0, borderRadius: 12, padding: 10, fontWeight: 700, cursor: "pointer" };

