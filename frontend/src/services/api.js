import { authHeader } from "./auth";

const API_BASE = import.meta?.env?.VITE_API_BASE_URL || import.meta?.env?.VITE_API_BASE || "http://127.0.0.1:8000"; // support both names

export async function fetchHealthReports() {
  const res = await fetch(`${API_BASE}/api/health/`, { headers: { ...authHeader() } });
  return res.json();
}

export async function submitHealthReport(body) {
  const res = await fetch(`${API_BASE}/api/health/`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify(body)
  });
  return res.json();
}

export async function fetchWaterReports() {
  const res = await fetch(`${API_BASE}/api/water/`, { headers: { ...authHeader() } });
  return res.json();
}

export async function submitWaterReport(body) {
  const res = await fetch(`${API_BASE}/api/water/`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify(body)
  });
  return res.json();
}

export async function fetchHeatmap() {
  const res = await fetch(`${API_BASE}/api/gis/heatmap`, { headers: { ...authHeader() } });
  return res.json();
}

export async function fetchAlerts() {
  const res = await fetch(`${API_BASE}/api/alerts/`, { headers: { ...authHeader() } });
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
  const res = await fetch(`${API_BASE}/api/communities/community`, { headers: { ...authHeader() } });
  return res.json();
}
export async function createCommunity(body) {
  const res = await fetch(`${API_BASE}/api/communities/community`, { method: "POST", headers: { "Content-Type": "application/json", ...authHeader() }, body: JSON.stringify(body) });
  return res.json();
}
export async function deleteCommunity(id) {
  const res = await fetch(`${API_BASE}/api/communities/community/${id}`, { method: "DELETE", headers: { ...authHeader() } });
  return res.json();
}

// File upload functions
export async function uploadWaterData(formData) {
  const res = await fetch(`${API_BASE}/api/water/upload`, { 
    method: "POST", 
    headers: { ...authHeader() }, 
    body: formData 
  });
  return res.json();
}

export async function uploadHealthData(formData) {
  const res = await fetch(`${API_BASE}/api/health/upload`, { 
    method: "POST", 
    headers: { ...authHeader() }, 
    body: formData 
  });
  return res.json();
}

// Emergency alert functions
export async function sendEmergencyAlert(alertData) {
  const res = await fetch(`${API_BASE}/api/emergency`, { 
    method: "POST", 
    headers: { "Content-Type": "application/json", ...authHeader() }, 
    body: JSON.stringify(alertData) 
  });
  return res.json();
}

// Multilingual support
export async function sendMultilingualAlert(alertData) {
  const res = await fetch(`${API_BASE}/api/multilingual`, { 
    method: "POST", 
    headers: { "Content-Type": "application/json", ...authHeader() }, 
    body: JSON.stringify(alertData) 
  });
  return res.json();
}