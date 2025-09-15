export async function fetchHealthReports() {
  const res = await fetch("/api/health");
  return res.json();
}