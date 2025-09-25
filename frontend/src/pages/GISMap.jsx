import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, Circle, useMap } from "react-leaflet";
import { Icon, divIcon } from "leaflet";
import { 
  MapPin, 
  AlertTriangle, 
  Droplets, 
  Users, 
  Activity,
  Send,
  Phone,
  MessageSquare,
  Globe
} from "lucide-react";
import { fetchAlerts, fetchHeatmap, sendSmsAlert } from "../services/api";
import "leaflet/dist/leaflet.css";
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

// Fix for default markers in react-leaflet
delete Icon.Default.prototype._getIconUrl;
Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

// Custom marker icons (DivIcon avoids base64/encoding issues with emojis)
const createEmojiIcon = (backgroundColor, label) => divIcon({
  className: "",
  html: `
    <div style="
      display:inline-flex;
      align-items:center;
      justify-content:center;
      width:28px;height:28px;
      border-radius:50%;
      background:${backgroundColor};
      box-shadow:0 2px 6px rgba(0,0,0,0.2);
      border:2px solid rgba(255,255,255,0.9);
      font-size:16px;line-height:1;">${label}</div>
  `,
  iconSize: [28, 28],
  iconAnchor: [14, 14],
  popupAnchor: [0, -14]
});

const alertIcon = createEmojiIcon("#FEE2E2", "!");
const waterIcon = createEmojiIcon("#DBEAFE", "💧");
const communityIcon = createEmojiIcon("#D1FAE5", "👥");
const healthIcon = createEmojiIcon("#EDE9FE", "❤️");

// Map component with controls
function MapControls({ onToggleLayer, activeLayers }) {
  return (
    <div style={{
      position: "absolute",
      top: "10px",
      right: "10px",
      zIndex: 1000,
      background: "white",
      borderRadius: "8px",
      padding: "10px",
      boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
      display: "flex",
      flexDirection: "column",
      gap: "5px"
    }}>
      {[
        { key: "alerts", label: "Alerts", icon: <AlertTriangle size={16} />, color: "#EF4444" },
        { key: "water", label: "Water Quality", icon: <Droplets size={16} />, color: "#3B82F6" },
        { key: "communities", label: "Communities", icon: <Users size={16} />, color: "#10B981" },
        { key: "health", label: "Health Reports", icon: <Activity size={16} />, color: "#8B5CF6" }
      ].map(layer => (
        <button
          key={layer.key}
          onClick={() => onToggleLayer(layer.key)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            padding: "8px 12px",
            border: "none",
            borderRadius: "6px",
            background: activeLayers[layer.key] ? layer.color + "20" : "transparent",
            color: activeLayers[layer.key] ? layer.color : "#6B7280",
            cursor: "pointer",
            fontSize: "14px",
            fontWeight: "500",
            transition: "all 0.2s"
          }}
        >
          {layer.icon}
          {layer.label}
        </button>
      ))}
    </div>
  );
}

export default function GISMap() {
  const [points, setPoints] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [form, setForm] = useState({ message: "", severity: "medium", phone: "", lang: "en" });
  const [activeLayers, setActiveLayers] = useState({
    alerts: true,
    water: true,
    communities: true,
    health: true
  });
  const [mapCenter, setMapCenter] = useState([17.3850, 78.4867]); // Hyderabad, India
  const [mapZoom, setMapZoom] = useState(10);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [heatmapData, alertsData] = await Promise.all([
          fetchHeatmap().catch(() => []),
          fetchAlerts().catch(() => [])
        ]);
        setPoints(heatmapData);
        setAlerts(alertsData);
      } catch (error) {
        console.error("Error loading map data:", error);
      }
    };
    loadData();
  }, []);

  const toggleLayer = (layerKey) => {
    setActiveLayers(prev => ({
      ...prev,
      [layerKey]: !prev[layerKey]
    }));
  };

  const handleSendSMS = async () => {
    try {
      const result = await sendSmsAlert(form);
      if (result.sent) {
        alert("SMS sent successfully!");
        setForm({ message: "", severity: "medium", phone: "", lang: "en" });
      } else {
        alert("Failed to send SMS");
      }
    } catch (error) {
      alert("Error sending SMS");
    }
  };

  // Mock data for demonstration
  const mockData = {
    alerts: [
      { lat: 17.3850, lng: 78.4867, severity: "high", message: "Water contamination detected", timestamp: "2 hours ago" },
      { lat: 17.3950, lng: 78.4967, severity: "medium", message: "Health outbreak reported", timestamp: "4 hours ago" },
      { lat: 17.3750, lng: 78.4767, severity: "low", message: "Routine maintenance alert", timestamp: "6 hours ago" }
    ],
    water: [
      { lat: 17.3900, lng: 78.4900, quality: "safe", ph: 7.2, turbidity: 0.5 },
      { lat: 17.4000, lng: 78.5000, quality: "caution", ph: 6.8, turbidity: 2.1 },
      { lat: 17.3800, lng: 78.4800, quality: "unsafe", ph: 5.5, turbidity: 4.2 }
    ],
    communities: [
      { lat: 17.3850, lng: 78.4867, name: "Community A", population: 1200, status: "safe" },
      { lat: 17.3950, lng: 78.4967, name: "Community B", population: 800, status: "caution" },
      { lat: 17.3750, lng: 78.4767, name: "Community C", population: 1500, status: "safe" }
    ],
    health: [
      { lat: 17.3850, lng: 78.4867, reports: 15, symptoms: ["fever", "diarrhea"], severity: "medium" },
      { lat: 17.3950, lng: 78.4967, reports: 8, symptoms: ["cough", "fatigue"], severity: "low" },
      { lat: 17.3750, lng: 78.4767, reports: 23, symptoms: ["nausea", "headache"], severity: "high" }
    ]
  };

  return (
    <div style={{ padding: "1.5rem", maxWidth: "1400px", margin: "0 auto" }}>
      {/* Header */}
      <div style={{ marginBottom: "1.5rem" }}>
        <h1 style={{ 
          margin: 0, 
          fontSize: "2rem", 
          fontWeight: "700", 
          color: "#1F2937",
          display: "flex",
          alignItems: "center",
          gap: "0.5rem"
        }}>
          <Globe size={32} color="#3B82F6" />
          Interactive GIS Health Map
        </h1>
        <p style={{ 
          margin: "0.5rem 0 0 0", 
          color: "#6B7280",
          fontSize: "1.1rem"
        }}>
          Monitor health data, water quality, and community alerts across the region
        </p>
      </div>

      {/* Map Container */}
      <div style={{ 
        height: "500px", 
        borderRadius: "16px", 
        overflow: "hidden",
        boxShadow: "0 10px 25px rgba(0,0,0,0.1)",
        border: "1px solid #E5E7EB",
        position: "relative",
        marginBottom: "1.5rem"
      }}>
        <MapContainer
          center={mapCenter}
          zoom={mapZoom}
          style={{ height: "100%", width: "100%" }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          
          {/* Alerts Layer */}
          {activeLayers.alerts && mockData.alerts.map((alert, idx) => (
            <Marker
              key={`alert-${idx}`}
              position={[alert.lat, alert.lng]}
              icon={alertIcon}
            >
              <Popup>
                <div style={{ padding: "0.5rem" }}>
                  <h4 style={{ margin: "0 0 0.5rem 0", color: "#DC2626" }}>
                    {alert.severity.toUpperCase()} ALERT
                  </h4>
                  <p style={{ margin: "0 0 0.5rem 0" }}>{alert.message}</p>
                  <small style={{ color: "#6B7280" }}>{alert.timestamp}</small>
                </div>
              </Popup>
            </Marker>
          ))}

          {/* Water Quality Layer */}
          {activeLayers.water && mockData.water.map((water, idx) => (
            <Marker
              key={`water-${idx}`}
              position={[water.lat, water.lng]}
              icon={waterIcon}
            >
              <Popup>
                <div style={{ padding: "0.5rem" }}>
                  <h4 style={{ margin: "0 0 0.5rem 0", color: "#3B82F6" }}>
                    Water Quality Report
                  </h4>
                  <p style={{ margin: "0 0 0.25rem 0" }}>
                    Status: <span style={{ 
                      color: water.quality === "safe" ? "#10B981" : 
                            water.quality === "caution" ? "#F59E0B" : "#EF4444",
                      fontWeight: "600"
                    }}>
                      {water.quality.toUpperCase()}
                    </span>
                  </p>
                  <p style={{ margin: "0 0 0.25rem 0" }}>pH: {water.ph}</p>
                  <p style={{ margin: "0" }}>Turbidity: {water.turbidity} NTU</p>
                </div>
              </Popup>
            </Marker>
          ))}

          {/* Communities Layer */}
          {activeLayers.communities && mockData.communities.map((community, idx) => (
            <Marker
              key={`community-${idx}`}
              position={[community.lat, community.lng]}
              icon={communityIcon}
            >
              <Popup>
                <div style={{ padding: "0.5rem" }}>
                  <h4 style={{ margin: "0 0 0.5rem 0", color: "#10B981" }}>
                    {community.name}
                  </h4>
                  <p style={{ margin: "0 0 0.25rem 0" }}>Population: {community.population}</p>
                  <p style={{ margin: "0" }}>
                    Status: <span style={{ 
                      color: community.status === "safe" ? "#10B981" : "#F59E0B",
                      fontWeight: "600"
                    }}>
                      {community.status.toUpperCase()}
                    </span>
                  </p>
                </div>
              </Popup>
            </Marker>
          ))}

          {/* Health Reports Layer */}
          {activeLayers.health && mockData.health.map((health, idx) => (
            <Marker
              key={`health-${idx}`}
              position={[health.lat, health.lng]}
              icon={healthIcon}
            >
              <Popup>
                <div style={{ padding: "0.5rem" }}>
                  <h4 style={{ margin: "0 0 0.5rem 0", color: "#8B5CF6" }}>
                    Health Reports
                  </h4>
                  <p style={{ margin: "0 0 0.25rem 0" }}>Reports: {health.reports}</p>
                  <p style={{ margin: "0 0 0.25rem 0" }}>
                    Severity: <span style={{ 
                      color: health.severity === "high" ? "#EF4444" : 
                            health.severity === "medium" ? "#F59E0B" : "#10B981",
                      fontWeight: "600"
                    }}>
                      {health.severity.toUpperCase()}
                    </span>
                  </p>
                  <p style={{ margin: "0" }}>
                    Symptoms: {health.symptoms.join(", ")}
                  </p>
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
        
        <MapControls onToggleLayer={toggleLayer} activeLayers={activeLayers} />
      </div>

      {/* Controls and Information */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
        {/* SMS Alert Form */}
        <div style={{ 
          background: "white", 
          borderRadius: "16px", 
          padding: "1.5rem", 
          boxShadow: "0 4px 6px rgba(0,0,0,0.05)",
          border: "1px solid #E5E7EB"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
            <Send size={20} color="#3B82F6" />
            <h3 style={{ margin: 0, color: "#1F2937" }}>Send Emergency Alert</h3>
          </div>
          <div style={{ display: "grid", gap: "1rem" }}>
            <div>
              <label style={{ display: "block", marginBottom: "0.5rem", color: "#374151", fontWeight: "500" }}>
                Message
              </label>
              <textarea
                placeholder="Enter alert message..."
                value={form.message}
                onChange={(e) => setForm({ ...form, message: e.target.value })}
                style={{
                  width: "100%",
                  padding: "0.75rem",
                  borderRadius: "8px",
                  border: "1px solid #D1D5DB",
                  fontSize: "14px",
                  resize: "vertical",
                  minHeight: "80px"
                }}
              />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div>
                <label style={{ display: "block", marginBottom: "0.5rem", color: "#374151", fontWeight: "500" }}>
                  Severity
                </label>
                <select
                  value={form.severity}
                  onChange={(e) => setForm({ ...form, severity: e.target.value })}
                  style={{
                    width: "100%",
                    padding: "0.75rem",
                    borderRadius: "8px",
                    border: "1px solid #D1D5DB",
                    fontSize: "14px"
                  }}
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
              <div>
                <label style={{ display: "block", marginBottom: "0.5rem", color: "#374151", fontWeight: "500" }}>
                  Language
                </label>
                <select
                  value={form.lang}
                  onChange={(e) => setForm({ ...form, lang: e.target.value })}
                  style={{
                    width: "100%",
                    padding: "0.75rem",
                    borderRadius: "8px",
                    border: "1px solid #D1D5DB",
                    fontSize: "14px"
                  }}
                >
                  <option value="en">English</option>
                  <option value="hi">हिंदी</option>
                  <option value="te">తెలుగు</option>
                </select>
              </div>
            </div>
            <div>
              <label style={{ display: "block", marginBottom: "0.5rem", color: "#374151", fontWeight: "500" }}>
                Phone Number
              </label>
              <input
                type="tel"
                placeholder="Enter phone number"
                value={form.phone}
                onChange={(e) => setForm({ ...form, phone: e.target.value })}
                style={{
                  width: "100%",
                  padding: "0.75rem",
                  borderRadius: "8px",
                  border: "1px solid #D1D5DB",
                  fontSize: "14px"
                }}
              />
            </div>
            <button
              onClick={handleSendSMS}
              style={{
                background: "linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)",
                color: "white",
                border: "none",
                borderRadius: "8px",
                padding: "0.75rem 1rem",
                fontSize: "14px",
                fontWeight: "600",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "0.5rem",
                transition: "transform 0.2s"
              }}
              onMouseEnter={(e) => e.target.style.transform = "translateY(-1px)"}
              onMouseLeave={(e) => e.target.style.transform = "translateY(0)"}
            >
              <Send size={16} />
              Send Alert
            </button>
          </div>
        </div>

        {/* Recent Alerts */}
        <div style={{ 
          background: "white", 
          borderRadius: "16px", 
          padding: "1.5rem", 
          boxShadow: "0 4px 6px rgba(0,0,0,0.05)",
          border: "1px solid #E5E7EB"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
            <AlertTriangle size={20} color="#EF4444" />
            <h3 style={{ margin: 0, color: "#1F2937" }}>Recent Alerts</h3>
          </div>
          <div style={{ display: "grid", gap: "0.75rem", maxHeight: "300px", overflowY: "auto" }}>
            {alerts.length > 0 ? alerts.map((alert, i) => (
              <div key={i} style={{ 
                background: "#FEF2F2", 
                padding: "1rem", 
                borderRadius: "8px",
                border: "1px solid #FECACA"
              }}>
                <div style={{ 
                  fontWeight: "600", 
                  color: "#DC2626", 
                  fontSize: "0.875rem",
                  marginBottom: "0.25rem"
                }}>
                  {alert.severity?.toUpperCase() || "ALERT"}
                </div>
                <div style={{ color: "#374151", fontSize: "0.875rem", marginBottom: "0.25rem" }}>
                  {alert.message}
                </div>
                <div style={{ color: "#9CA3AF", fontSize: "0.75rem" }}>
                  {alert.timestamp}
                </div>
              </div>
            )) : (
              <div style={{ 
                textAlign: "center", 
                color: "#6B7280", 
                padding: "2rem",
                fontSize: "0.875rem"
              }}>
                No recent alerts
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

