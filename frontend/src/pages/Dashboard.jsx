import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import ripple from "../assets/Logo.png";
import { fetchAuthoritySummary, fetchWaterSummary, listCommunities } from "../services/api";
import { colors } from "../theme";

export default function Dashboard() {
  const [summary, setSummary] = useState({ active_monitors: 2847, communities: 156, safe_areas_pct: 98.2 });
  const [water, setWater] = useState({ safe: 142, caution: 23, unsafe: 8 });
  const [communities, setCommunities] = useState([]);
  useEffect(()=>{
    fetchAuthoritySummary().then((d)=>{ if (d && d.active_monitors) setSummary(d); });
    fetchWaterSummary().then((d)=>{ if (d && ("safe" in d)) setWater(d); });
    listCommunities().then((d)=>{ if (Array.isArray(d)) setCommunities(d); });
  },[]);
  return (
    <section style={{ padding: "2rem" }}>
      {/* Hero */}
      <div style={{
        position: "relative",
        overflow: "hidden",
        borderRadius: "24px",
        background: "linear-gradient(135deg, #E8F6F3 0%, #F7F9FA 100%)",
        padding: "3rem 2rem",
        boxShadow: "0 10px 30px rgba(0,0,0,0.06)",
        marginBottom: "2rem"
      }}>
        <img src={ripple} alt="Ripple" style={{
          position: "absolute", right: "-40px", bottom: "-40px", width: "260px", opacity: 0.15, filter: "blur(1px)"
        }} />
        <h1 style={{ color: colors.primary, fontSize: "2.2rem", margin: 0 }}>Vitanza - Smart Health Monitoring for Safer Communities</h1>
        <p style={{ color: "#4A5568", maxWidth: "640px", marginTop: "0.5rem" }}>Monitor water quality, track health symptoms, and stay informed with real-time alerts. Together, we build healthier communities through data-driven insights powered by Vitanza.</p>
        <div style={{ display: "flex", gap: "1rem", marginTop: "1.5rem", flexWrap: "wrap" }}>
          <Link to="/reports" style={ctaStyle}>Report Symptoms</Link>
          <Link to="/water" style={ctaStyleSecondary}>Check Water Quality</Link>
          <Link to="/alerts" style={ctaStyleTertiary}>View Alerts</Link>
        </div>
      </div>

      {/* Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "1rem", marginBottom: "2rem" }}>
        {[{ label: "Active Monitors", value: summary.active_monitors.toLocaleString() }, { label: "Communities", value: summary.communities }, { label: "Safe Areas", value: `${summary.safe_areas_pct}%` }].map((s) => (
          <div key={s.label} style={statCard}>
            <div style={{ fontSize: 28, fontWeight: 800 }}>{s.value}</div>
            <div style={{ fontSize: 12, color: "#6B7280" }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Features */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1rem", marginTop: "1rem" }}>
        {featureCards.map((f) => (
          <div key={f.title} style={cardStyle}>
            <div style={{ fontSize: "1.8rem" }}>{f.icon}</div>
            <h3 style={{ margin: "0.5rem 0", color: colors.primary }}>{f.title}</h3>
            <p style={{ margin: 0, color: "#4A5568" }}>{f.desc}</p>
          </div>
        ))}
      </div>

      {/* Water status */}
      <div style={{ marginTop: 24, background: "white", borderRadius: 16, padding: 16, boxShadow: "0 6px 16px rgba(0,0,0,0.06)" }}>
        <h3 style={{ marginTop: 0 }}>Current Water Quality Status</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12 }}>
          {[{ label: "Safe", value: water.safe }, { label: "Caution", value: water.caution }, { label: "Unsafe", value: water.unsafe }].map((w) => (
            <div key={w.label} style={statPill}>
              <div style={{ fontSize: 22, fontWeight: 800 }}>{w.value}</div>
              <div style={{ fontSize: 12, color: "#6B7280" }}>{w.label} Communities</div>
            </div>
          ))}
        </div>
      </div>

      {/* Impact metrics and contributors placeholders */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16, marginTop: 24 }}>
        {impact.map((m) => (
          <div key={m.label} style={cardStyle}>
            <div style={{ fontSize: 24, fontWeight: 800 }}>{m.value}</div>
            <div style={{ color: "#6B7280" }}>{m.label}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 24, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 16 }}>
        {contributors.map((c) => (
          <div key={c.name} style={cardStyle}>
            <div style={{ fontWeight: 700 }}>{c.name}</div>
            <div style={{ color: "#6B7280" }}>{c.title}</div>
          </div>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 16, marginTop: 24 }}>
        <aside style={{ background: "white", borderRadius: 16, padding: 16, boxShadow: "0 6px 16px rgba(0,0,0,0.06)" }}>
          <h4 style={{ marginTop: 0 }}>Health Worker</h4>
          <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: 8 }}>
            <li><Link to="/reports">Reports</Link></li>
            <li><Link to="/water">Water Quality</Link></li>
            <li><Link to="/alerts">Alerts Heatmap</Link></li>
          </ul>
          <div style={{ marginTop: 12 }}>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>Communities</div>
            <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: 6 }}>
              {communities.slice(0,5).map((c)=> (
                <li key={c.id} style={{ color: "#6B7280" }}>{c.name} — {c.district}</li>
              ))}
            </ul>
          </div>
        </aside>
        <div style={{ display: "grid", gap: 16 }}>
          <div style={{ background: "white", borderRadius: 16, padding: 16, boxShadow: "0 6px 16px rgba(0,0,0,0.06)" }}>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Cases Over Time</div>
            <div style={{ height: 160, background: "linear-gradient(180deg, #E8F6F3, #FFFFFF)", borderRadius: 12 }} />
          </div>
          <div style={{ background: "white", borderRadius: 16, padding: 16, boxShadow: "0 6px 16px rgba(0,0,0,0.06)" }}>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Village Heatmap</div>
            <div style={{ height: 160, background: "#E6F0FA", borderRadius: 12 }} />
          </div>
        </div>
      </div>
    </section>
  );
}

const ctaBase = {
  padding: "0.85rem 1.1rem",
  borderRadius: "14px",
  fontWeight: 600,
  textDecoration: "none",
  transition: "transform 200ms ease, box-shadow 200ms ease",
  boxShadow: "0 6px 16px rgba(0,0,0,0.08)",
};

const ctaStyle = {
  ...ctaBase,
  background: colors.accent,
  color: "white",
};

const ctaStyleSecondary = {
  ...ctaBase,
  background: "#2D9CDB",
  color: "white",
};

const ctaStyleTertiary = {
  ...ctaBase,
  background: "#F2994A",
  color: "white",
};

const cardStyle = {
  background: "white",
  borderRadius: "18px",
  padding: "1.2rem",
  boxShadow: "0 8px 20px rgba(0,0,0,0.06)",
};

const featureCards = [
  { title: "Community Reporting", desc: "Multilingual symptom reports from field workers and citizens.", icon: "👥" },
  { title: "Water Quality Status", desc: "Live safety labels: Safe, Caution, Unsafe.", icon: "💧" },
  { title: "Real-time Alerts", desc: "Instant notifications to officials and workers.", icon: "🚨" },
  { title: "AI Predictions", desc: "Detect patterns and predict outbreaks early.", icon: "🤖" },
];

const stats = [
  { label: "Active Monitors", value: "2,847" },
  { label: "Communities", value: "156" },
  { label: "Safe Areas", value: "98.2%" },
];

const waterStats = [
  { label: "Safe", value: 142 },
  { label: "Caution", value: 23 },
  { label: "Unsafe", value: 8 },
];

const statCard = { background: "white", borderRadius: 16, padding: 16, textAlign: "center", boxShadow: "0 6px 16px rgba(0,0,0,0.06)" };
const statPill = { background: "#F8FAFC", borderRadius: 12, padding: 12 };

const impact = [
  { label: "Active Community Members", value: "12,847" },
  { label: "Monitored Locations", value: "156" },
  { label: "Reports This Month", value: "2,394" },
  { label: "Average Response Time", value: "4.2 min" },
];

const contributors = [
  { name: "Dr. Sarah Johnson", title: "Lead Health Researcher" },
  { name: "Michael Chen", title: "Full Stack Developer" },
  { name: "Dr. Amara Okafor", title: "Community Health Expert" },
  { name: "James Rodriguez", title: "Data Scientist" },
];