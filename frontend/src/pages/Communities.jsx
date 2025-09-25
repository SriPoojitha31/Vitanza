import React, { useEffect, useState } from "react";
import { createCommunity, deleteCommunity, listCommunities } from "../services/api";

export default function Communities() {
  const [items, setItems] = useState([]);
  const [name, setName] = useState("");
  const [district, setDistrict] = useState("");
  const [wsStatus, setWsStatus] = useState("disconnected");

  useEffect(() => {
    (async () => {
      const data = await listCommunities();
      setItems(Array.isArray(data) ? data : []);
    })();

    const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host.replace(/:\\d+$/, ':8000') + '/api/communities/ws';
    const ws = new WebSocket(wsUrl);
    ws.onopen = () => setWsStatus("connected");
    ws.onclose = () => setWsStatus("disconnected");
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'created') setItems((prev) => [msg.data, ...prev]);
        if (msg.type === 'updated') setItems((prev) => prev.map(it => it._id === msg.data._id ? msg.data : it));
        if (msg.type === 'deleted') setItems((prev) => prev.filter(it => it._id !== msg.id));
      } catch {}
    };
    return () => ws.close();
  }, []);

  const onCreate = async (e) => {
    e.preventDefault();
    if (!name) return;
    const body = { name, district };
    const res = await createCommunity(body);
    if (res && res._id) setItems((prev) => [res, ...prev]);
    setName("");
    setDistrict("");
  };

  const onDelete = async (id) => {
    await deleteCommunity(id);
    setItems((prev) => prev.filter((it) => it._id !== id));
  };

  return (
    <section style={{ padding: "1.5rem", maxWidth: 900, margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>Communities</h2>
        <div style={{ fontSize: 12, color: wsStatus === 'connected' ? '#10B981' : '#EF4444' }}>WS: {wsStatus}</div>
      </div>

      <form onSubmit={onCreate} style={{ display: 'grid', gridTemplateColumns: '2fr 2fr auto', gap: 8, background: 'white', padding: 12, borderRadius: 12, boxShadow: '0 4px 10px rgba(0,0,0,0.06)', marginBottom: 12 }}>
        <input placeholder="Community name" value={name} onChange={(e)=>setName(e.target.value)} style={input} />
        <input placeholder="District" value={district} onChange={(e)=>setDistrict(e.target.value)} style={input} />
        <button type="submit" style={btnPrimary}>Add</button>
      </form>

      <div style={{ background: 'white', borderRadius: 12, boxShadow: '0 4px 10px rgba(0,0,0,0.06)' }}>
        {items.length === 0 ? (
          <div style={{ padding: 24, color: '#6B7280' }}>No communities yet.</div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: 0 }}>
            <thead>
              <tr>
                <th style={th}>Name</th>
                <th style={th}>District</th>
                <th style={th}></th>
              </tr>
            </thead>
            <tbody>
              {items.map((c) => (
                <tr key={c._id}>
                  <td style={td}>{c.name}</td>
                  <td style={td}>{c.district || '-'}</td>
                  <td style={{ ...td, textAlign: 'right' }}>
                    <button onClick={()=>onDelete(c._id)} style={btnDanger}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}

const input = { padding: '0.6rem 0.75rem', border: '1px solid #E5E7EB', borderRadius: 8 };
const th = { textAlign: 'left', padding: '0.75rem', background: '#F3F4F6', fontWeight: 600 };
const td = { padding: '0.75rem', borderBottom: '1px solid #EEF2F7' };
const btnPrimary = { background: '#3B82F6', color: 'white', border: 0, borderRadius: 8, padding: '0.6rem 1rem', cursor: 'pointer' };
const btnDanger = { background: 'transparent', color: '#EF4444', border: '1px solid #FECACA', borderRadius: 8, padding: '0.4rem 0.75rem', cursor: 'pointer' };


