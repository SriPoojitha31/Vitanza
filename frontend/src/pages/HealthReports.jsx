import React, { useEffect, useMemo, useState } from "react";
import { fetchHealthReports } from "../services/api";
import { authHeader } from "../services/auth";

export default function HealthReports() {
  const [reports, setReports] = useState([]);
  const [sortKey, setSortKey] = useState("timestamp");
  useEffect(() => {
    fetchHealthReports().then((data)=>{
      if (Array.isArray(data) && data.length) { setReports(data); return; }
      // inject demo when empty
      setReports([
        { patient_id: "demo-001", symptoms: ["fever","diarrhea"], location: "Village A", timestamp: new Date().toISOString() },
        { patient_id: "demo-002", symptoms: ["nausea"], location: "Village B", timestamp: new Date().toISOString() },
      ]);
    });
  }, []);
  const sorted = useMemo(() => {
    return [...reports].sort((a,b) => String(a[sortKey]).localeCompare(String(b[sortKey])));
  }, [reports, sortKey]);
  return (
    <section style={{ padding: "1.5rem" }}>
      <h2 style={{ marginBottom: "1rem" }}>Reports</h2>
      <form onSubmit={async (e)=>{ e.preventDefault();
        const form = new FormData(e.currentTarget);
        const payload = { patient_id: form.get("patient_id"), symptoms: String(form.get("symptoms")).split(",").map(s=>s.trim()).filter(Boolean), location: form.get("location"), timestamp: new Date().toISOString() };
        await fetch("/api/health", { method: "POST", headers: { "Content-Type": "application/json", ...authHeader() }, body: JSON.stringify(payload) });
        const next = await fetchHealthReports();
        setReports(next);
      }} style={{ background: "white", borderRadius: 12, padding: 12, boxShadow: "0 6px 16px rgba(0,0,0,0.06)", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 8, marginBottom: 12 }}>
        <input name="patient_id" placeholder="Patient ID" style={tdStyle} />
        <input name="symptoms" placeholder="Symptoms (comma)" style={tdStyle} />
        <input name="location" placeholder="Location" style={tdStyle} />
        <button type="submit" style={{ ...tdStyle, cursor: "pointer", background: "#27AE60", color: "white", border: 0, borderRadius: 8 }}>Submit</button>
      </form>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "separate", borderSpacing: 0 }}>
          <thead>
            <tr>
              {headers.map((h) => (
                <th key={h.key} onClick={() => setSortKey(h.key)} style={thStyle}>{h.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => (
              <tr key={i} style={{ background: i % 2 ? "#FAFBFC" : "white" }}>
                <td style={tdStyle}>{r.patient_id}</td>
                <td style={tdStyle}>{(r.symptoms || []).join(", ")}</td>
                <td style={tdStyle}>{r.location}</td>
                <td style={tdStyle}>{r.timestamp}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

const headers = [
  { key: "patient_id", label: "Patient" },
  { key: "symptoms", label: "Symptoms" },
  { key: "location", label: "Location" },
  { key: "timestamp", label: "Timestamp" },
];

const thStyle = {
  textAlign: "left",
  padding: "0.75rem 0.75rem",
  background: "#EDF2F7",
  cursor: "pointer",
  position: "sticky",
  top: 0
};

const tdStyle = {
  padding: "0.75rem 0.75rem",
  borderBottom: "1px solid #EEF2F7",
};

