import React, { useState } from "react";

export default function Feedback() {
  const [input, setInput] = useState("");
  const [msgs, setMsgs] = useState([{ role: "assistant", content: "Hi! Ask me about water quality or reports." }]);
  function send() {
    if (!input.trim()) return;
    const userMsg = { role: "user", content: input };
    const botMsg = { role: "assistant", content: "This is a demo response. Connect LLM backend to answer contextually." };
    setMsgs((m)=>[...m, userMsg, botMsg]);
    setInput("");
  }
  return (
    <section style={{ padding: "1.5rem" }}>
      <h2>AI Assistant</h2>
      <div style={{ background: "white", borderRadius: 16, padding: 16, boxShadow: "0 6px 16px rgba(0,0,0,0.06)", display: "grid", gap: 12 }}>
        <div style={{ maxHeight: 300, overflow: "auto", display: "grid", gap: 8 }}>
          {msgs.map((m, i)=> (
            <div key={i} style={{ alignSelf: m.role === "user" ? "end" : "start", background: m.role === "user" ? "#E0F2FE" : "#F3F4F6", padding: 10, borderRadius: 10, maxWidth: "75%" }}>
              {m.content}
            </div>
          ))}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <input value={input} onChange={(e)=>setInput(e.target.value)} placeholder="Type a message..." style={{ flex: 1, padding: 10, borderRadius: 10, border: "1px solid #E5E7EB" }} />
          <button onClick={send} style={{ background: "#2D9CDB", color: "white", border: 0, borderRadius: 12, padding: "0 16px", fontWeight: 700, cursor: "pointer" }}>Send</button>
        </div>
      </div>
    </section>
  );
}

