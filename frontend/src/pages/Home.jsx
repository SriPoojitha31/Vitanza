import React, { useContext } from "react";
import { Link } from "react-router-dom";
import logo from "../assets/Logo.png";
import { I18nContext } from "../i18n/I18nProvider";

export default function Home() {
  const { t } = useContext(I18nContext);
  return (
    <div style={{ padding: "0 1rem" }}>
      <div style={heroWrap}>
        <div style={heroInner}>
          <div style={heroGrid}>
            <div>
              <h1 style={heroTitle}>{t("home.hero_title")}</h1>
              <p style={heroSub}>{t("home.hero_sub")}</p>
              <div style={ctaRow}>
                <Link to="/signup" style={primaryBtn}>{t("home.cta_get_started")}</Link>
                <Link to="/login" style={ghostBtn}>{t("home.cta_sign_in")}</Link>
              </div>
            </div>
            <div style={logoWrap}>
              <div style={logoCard}>
                <img src={logo} alt="Vitanza" style={{ width: "100%", height: "auto", filter: "drop-shadow(0 20px 30px rgba(0,0,0,0.25))" }} />
              </div>
            </div>
          </div>
        </div>
      </div>

      <section style={{ maxWidth: "1100px", margin: "2rem auto", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1.25rem" }}>
        {features(t).map((f) => (
          <div key={f.title} style={card}>
            <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "#111827", marginBottom: "0.5rem" }}>{f.title}</div>
            <p style={{ margin: 0, color: "#6B7280", lineHeight: 1.6 }}>{f.desc}</p>
          </div>
        ))}
      </section>

      <section style={{ maxWidth: "1100px", margin: "2rem auto", background: "#F9FAFB", border: "1px solid #E5E7EB", borderRadius: 16, padding: "1.5rem" }}>
        <h2 style={{ marginTop: 0, color: "#1F2937" }}>{t("home.why_title")}</h2>
        <p style={{ margin: 0, color: "#4B5563" }}>{t("home.why_desc")}</p>
        <div style={{ marginTop: "1rem", display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          <Link to="/reports" style={chipLink}>{t("health_reports")}</Link>
          <Link to="/alerts" style={chipLink}>Alerts</Link>
          <Link to="/dashboard/community" style={chipLink}>Communities</Link>
          <Link to="/water" style={chipLink}>{t("water_quality")}</Link>
        </div>
      </section>
    </div>
  );
}

const heroWrap = {
  width: "100%",
  background: "linear-gradient(135deg, #2C3E50 0%, #27AE60 100%)",
  borderBottom: "1px solid #E5E7EB",
};

const heroInner = {
  maxWidth: "1100px",
  margin: "0 auto",
  padding: "4rem 1rem 3rem 1rem",
  color: "white",
  textAlign: "center",
  position: "relative",
};

const heroGrid = {
  display: "grid",
  gridTemplateColumns: "1.2fr 0.8fr",
  gap: "2rem",
  alignItems: "center",
};

const heroTitle = {
  margin: 0,
  fontSize: "3rem",
  fontWeight: 800,
  letterSpacing: 1,
};

const heroSub = {
  margin: "0.75rem 0 0 0",
  opacity: 0.95,
  fontSize: "1.1rem",
};

const ctaRow = {
  marginTop: "1.5rem",
  display: "flex",
  gap: "0.75rem",
  justifyContent: "center",
};

const logoWrap = {
  perspective: "1000px",
};

const logoCard = {
  background: "rgba(255,255,255,0.08)",
  border: "1px solid rgba(255,255,255,0.2)",
  borderRadius: 20,
  padding: "1rem",
  transform: "rotateY(-12deg) rotateX(6deg)",
  boxShadow: "0 25px 60px rgba(0,0,0,0.35)",
};

const primaryBtn = {
  background: "#2D9CDB",
  color: "white",
  textDecoration: "none",
  padding: "0.6rem 1rem",
  borderRadius: 12,
  fontWeight: 700,
};

const ghostBtn = {
  background: "transparent",
  color: "#111827",
  textDecoration: "none",
  padding: "0.6rem 1rem",
  borderRadius: 12,
  border: "1px solid #CBD5E1",
  fontWeight: 700,
};

const card = {
  background: "white",
  border: "1px solid #E5E7EB",
  borderRadius: 16,
  padding: "1rem",
  boxShadow: "0 2px 6px rgba(0,0,0,0.04)",
};

const chipLink = {
  textDecoration: "none",
  background: "white",
  border: "1px solid #E5E7EB",
  color: "#1F2937",
  padding: "0.4rem 0.7rem",
  borderRadius: 999,
  fontWeight: 600
};

const features = (t) => [
  { title: t("home.features.alerts"), desc: t("home.features.alerts_desc") },
  { title: t("home.features.reports"), desc: t("home.features.reports_desc") },
  { title: t("home.features.iot"), desc: t("home.features.iot_desc") },
  { title: t("home.features.ai"), desc: t("home.features.ai_desc") },
];
