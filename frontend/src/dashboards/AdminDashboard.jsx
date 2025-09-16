import React, { useEffect, useState } from "react";
import { listCommunities } from "../services/api";

export default function AdminDashboard() {
  const [communities, setCommunities] = useState([]);
  useEffect(()=>{ listCommunities().then((d)=>Array.isArray(d)&&setCommunities(d)); },[]);
  return (
    <section style={{ padding: "1.5rem" }}>
      <h2>Admin Dashboard</h2>
      <div style={{ marginTop: 12 }}>
        <div style={{ fontWeight: 700, marginBottom: 6 }}>Communities</div>
        <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: 6 }}>
          {communities.map((c)=> (
            <li key={c.id} style={{ background: "white", padding: 10, borderRadius: 10, boxShadow: "0 6px 16px rgba(0,0,0,0.06)" }}>{c.name} — {c.district}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}
