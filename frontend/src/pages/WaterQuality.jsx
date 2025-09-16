import React, { useEffect, useState } from "react";
import { fetchWaterReports } from "../services/api";
import { authHeader } from "../services/auth";

function statusFrom(ph, turbidity) {
  if (ph < 6.5 || ph > 8.5 || turbidity > 5) return { label: "Unsafe", color: "#EB5757" };
  if (turbidity > 3) return { label: "Caution", color: "#F2C94C" };
  return { label: "Safe", color: "#27AE60" };
}

export default function WaterQuality() {
  const [items, setItems] = useState([]);
  useEffect(() => {
    fetchWaterReports().then((data)=>{
      if (Array.isArray(data) && data.length) { setItems(data); return; }
      setItems([
        { sensor_id: "sensor-001", ph: 7.2, turbidity: 2.5, location: "Well 3, Village A", timestamp: new Date().toISOString() },
        { sensor_id: "sensor-002", ph: 6.3, turbidity: 5.2, location: "River, Village B", timestamp: new Date().toISOString() },
      ]);
    });
  }, []);
  return (
    <section style={{ padding: "1.5rem" }}>
      <h2 style={{ marginBottom: "1rem" }}>Water Quality</h2>
      <form onSubmit={async (e)=>{ e.preventDefault();
        const form = new FormData(e.currentTarget);
        const payload = { sensor_id: form.get("sensor_id"), ph: Number(form.get("ph")), turbidity: Number(form.get("turbidity")), location: form.get("location"), timestamp: new Date().toISOString() };
        await fetch("/api/water", { method: "POST", headers: { "Content-Type": "application/json", ...authHeader() }, body: JSON.stringify(payload) });
        const next = await fetchWaterReports();
        setItems(next);
      }} style={{ background: "white", borderRadius: 12, padding: 12, boxShadow: "0 6px 16px rgba(0,0,0,0.06)", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 8, marginBottom: 12 }}>
        <input name="sensor_id" placeholder="Sensor ID" style={{ padding: 10, borderRadius: 10, border: "1px solid #E5E7EB" }} />
        <input name="ph" placeholder="pH" type="number" step="0.1" style={{ padding: 10, borderRadius: 10, border: "1px solid #E5E7EB" }} />
        <input name="turbidity" placeholder="Turbidity" type="number" step="0.1" style={{ padding: 10, borderRadius: 10, border: "1px solid #E5E7EB" }} />
        <input name="location" placeholder="Location" style={{ padding: 10, borderRadius: 10, border: "1px solid #E5E7EB" }} />
        <button type="submit" style={{ padding: 10, borderRadius: 10, background: "#2D9CDB", color: "white", border: 0, fontWeight: 700, cursor: "pointer" }}>Submit</button>
      </form>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16 }}>
        {items.map((w, i) => {
          const s = statusFrom(w.ph, w.turbidity);
          return (
            <div key={i} style={{ background: "white", borderRadius: 16, padding: 16, boxShadow: "0 6px 16px rgba(0,0,0,0.06)" }}>
              <div style={{ fontSize: 14, color: "#6B7280" }}>{w.location}</div>
              <div style={{ marginTop: 8, display: "flex", gap: 12, alignItems: "center" }}>
                <span style={{ fontWeight: 700, color: s.color }}>{s.label}</span>
                <span>pH {w.ph}</span>
                <span>Turbidity {w.turbidity} NTU</span>
                {w.risk && <span>Risk: {w.risk}</span>}
              </div>
              <div style={{ fontSize: 12, color: "#9CA3AF", marginTop: 8 }}>{w.timestamp}</div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

