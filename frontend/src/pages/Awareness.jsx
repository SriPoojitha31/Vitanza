import React, { useEffect, useState } from "react";

export default function Awareness() {
  const [items, setItems] = useState([]);
  useEffect(()=>{
    // fallback demo content
    setItems([
      { title: "Hygiene Tips", body: "Wash hands, keep surroundings clean." },
      { title: "Boil Water", body: "Boil for 10 minutes before drinking." },
    ]);
  },[]);
  return (
    <section style={{ padding: "1.5rem" }}>
      <h2>Community</h2>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 16 }}>
        {items.map((a, i) => (
          <div key={i} style={{ background: "white", borderRadius: 16, padding: 16, boxShadow: "0 6px 16px rgba(0,0,0,0.06)" }}>
            <div style={{ fontWeight: 700 }}>{a.title}</div>
            <div style={{ color: "#6B7280" }}>{a.body}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

