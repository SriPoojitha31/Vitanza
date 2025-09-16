import { authHeader } from "./auth";

const API_BASE = import.meta?.env?.VITE_API_BASE || ""; // use proxy when empty

export async function fetchHealthReports() {
  const res = await fetch(`${API_BASE}/api/health`, { headers: { ...authHeader() } });
  return res.json();
}

export async function fetchWaterReports() {
  const res = await fetch(`${API_BASE}/api/water`, { headers: { ...authHeader() } });
  return res.json();
}

export async function fetchHeatmap() {
  const res = await fetch(`${API_BASE}/api/gis/heatmap`, { headers: { ...authHeader() } });
  return res.json();
}

export async function fetchAlerts() {
  const res = await fetch(`${API_BASE}/api/alerts`, { headers: { ...authHeader() } });
  return res.json();
}

export async function sendSmsAlert({ message, severity, phone, lang }) {
  const params = new URLSearchParams({ message, severity, phone, lang });
  const res = await fetch(`${API_BASE}/api/alerts/sms?${params.toString()}`, { method: "POST", headers: { ...authHeader() } });
  return res.json();
}

// RBAC endpoints
export async function fetchAuthoritySummary() {
  const res = await fetch(`${API_BASE}/api/authority/summary`, { headers: { ...authHeader() } });
  return res.json();
}
export async function fetchWaterSummary() {
  const res = await fetch(`${API_BASE}/api/authority/water-summary`, { headers: { ...authHeader() } });
  return res.json();
}
export async function listCommunities() {
  const res = await fetch(`${API_BASE}/api/community`, { headers: { ...authHeader() } });
  return res.json();
}
export async function createCommunity(body) {
  const res = await fetch(`${API_BASE}/api/community`, { method: "POST", headers: { "Content-Type": "application/json", ...authHeader() }, body: JSON.stringify(body) });
  return res.json();
}
export async function deleteCommunity(id) {
  const res = await fetch(`${API_BASE}/api/community/${id}`, { method: "DELETE", headers: { ...authHeader() } });
  return res.json();
}