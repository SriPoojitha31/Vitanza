import React, { useState } from "react";

export default function Signup() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("worker");
  const [message, setMessage] = useState("");

  async function submit(e) {
    e.preventDefault();
    try {
      const res = await fetch("/api/auth/register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username, password, role }) });
      const data = await res.json();
      if (res.ok) {
        setMessage("Account created. You can sign in now.");
      } else {
        setMessage(data.detail || "Sign up failed");
      }
    } catch (_) {
      setMessage("Network error");
    }
  }

  return (
    <section style={{ padding: "2rem", display: "grid", placeItems: "center" }}>
      <form onSubmit={submit} style={{ background: "white", padding: 24, borderRadius: 16, boxShadow: "0 10px 24px rgba(0,0,0,0.06)", width: 360, maxWidth: "90%" }}>
        <h3 style={{ marginTop: 0 }}>Sign Up</h3>
        <label>Username</label>
        <input value={username} onChange={(e)=>setUsername(e.target.value)} style={input} />
        <label>Password</label>
        <input type="password" value={password} onChange={(e)=>setPassword(e.target.value)} style={input} />
        <label>Role</label>
        <select value={role} onChange={(e)=>setRole(e.target.value)} style={input}>
          <option value="community">Community User</option>
          <option value="worker">Health Worker</option>
          <option value="officer">Health Authority</option>
          <option value="admin">Admin</option>
        </select>
        <button type="submit" style={btn}>Create Account</button>
        {message && <div style={{ marginTop: 8, color: "#6B7280" }}>{message}</div>}
      </form>
    </section>
  );
}

const input = { width: "100%", padding: 10, borderRadius: 10, border: "1px solid #E5E7EB", marginBottom: 12 };
const btn = { width: "100%", background: "#27AE60", color: "white", border: 0, borderRadius: 12, padding: 10, fontWeight: 700, cursor: "pointer" };


