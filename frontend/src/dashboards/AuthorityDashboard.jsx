import React, { useEffect, useState } from "react";
import { fetchAuthoritySummary, fetchWaterSummary } from "../services/api";

export default function AuthorityDashboard() {
  const [summary, setSummary] = useState(null);
  const [water, setWater] = useState(null);
  useEffect(()=>{ fetchAuthoritySummary().then(setSummary); fetchWaterSummary().then(setWater); },[]);
  return (
    <section style={{ padding: "1.5rem" }}>
      <h2>Authority Dashboard</h2>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
        {summary && (
          <div style={card}><div style={title}>Monitors</div><div style={value}>{summary.active_monitors}</div></div>
        )}
        {summary && (
          <div style={card}><div style={title}>Communities</div><div style={value}>{summary.communities}</div></div>
        )}
        {summary && (
          <div style={card}><div style={title}>Safe Areas</div><div style={value}>{summary.safe_areas_pct}%</div></div>
        )}
        {water && (
          <div style={card}><div style={title}>Water Status</div><div style={value}>Safe {water.safe} / Caution {water.caution} / Unsafe {water.unsafe}</div></div>
        )}
      </div>
    </section>
  );
}

const card = { background: "white", borderRadius: 16, padding: 16, boxShadow: "0 6px 16px rgba(0,0,0,0.06)" };
const title = { color: "#6B7280", fontSize: 12 };
const value = { fontWeight: 800, fontSize: 22 };
