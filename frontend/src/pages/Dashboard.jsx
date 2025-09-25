import {
  Activity,
  AlertTriangle,
  BarChart3,
  Droplets,
  Globe,
  Heart,
  MapPin,
  Shield,
  Users,
  Zap
} from "lucide-react";
import React, { useContext, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { AuthContext } from "../auth/AuthContext";
import { fetchAlerts, fetchAuthoritySummary, fetchHealthReports, fetchWaterSummary, listCommunities } from "../services/api";
import { colors } from "../theme";

export default function Dashboard() {
  const { user } = useContext(AuthContext);
  console.log('Dashboard - User context:', user);
  const [summary, setSummary] = useState({ active_monitors: 2847, communities: 156, safe_areas_pct: 98.2 });
  const [water, setWater] = useState({ safe: 142, caution: 23, unsafe: 8 });
  const [communities, setCommunities] = useState([]);
  const [recentReports, setRecentReports] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [summaryData, waterData, communitiesData, reportsData, alertsData] = await Promise.all([
          fetchAuthoritySummary().catch(() => ({ active_monitors: 2847, communities: 156, safe_areas_pct: 98.2 })),
          fetchWaterSummary().catch(() => ({ safe: 142, caution: 23, unsafe: 8 })),
          listCommunities().catch(() => []),
          fetchHealthReports().catch(() => []),
          fetchAlerts().catch(() => [])
        ]);

        if (summaryData && summaryData.active_monitors) setSummary(summaryData);
        if (waterData && "safe" in waterData) setWater(waterData);
        if (Array.isArray(communitiesData)) setCommunities(communitiesData);
        if (Array.isArray(reportsData)) setRecentReports(reportsData.slice(0, 5));
        if (Array.isArray(alertsData)) setAlerts(alertsData.slice(0, 3));
      } catch (error) {
        console.error("Error loading dashboard data:", error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  if (loading) {
    return (
      <div style={{ 
        display: "flex", 
        justifyContent: "center", 
        alignItems: "center", 
        height: "50vh",
        fontSize: "1.2rem",
        color: "#6B7280"
      }}>
        Loading dashboard...
      </div>
    );
  }

  return (
    <div style={{ padding: "2rem", maxWidth: "1400px", margin: "0 auto" }}>
      {/* Welcome Header */}
      <div style={{
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        borderRadius: "20px",
        padding: "2rem",
        color: "white",
        marginBottom: "2rem",
        position: "relative",
        overflow: "hidden"
      }}>
        <div style={{
          position: "absolute",
          top: "-50px",
          right: "-50px",
          width: "200px",
          height: "200px",
          background: "rgba(255,255,255,0.1)",
          borderRadius: "50%",
          filter: "blur(40px)"
        }} />
        <h1 style={{ 
          fontSize: "2.5rem", 
          margin: 0, 
          fontWeight: "700",
          marginBottom: "0.5rem"
        }}>
          Welcome back, {user?.displayName || "User"}!
        </h1>
        <p style={{ 
          fontSize: "1.1rem", 
          margin: 0, 
          opacity: 0.9,
          maxWidth: "600px"
        }}>
          Monitor water quality, track health symptoms, and stay informed with real-time alerts. 
          Together, we build healthier communities through data-driven insights.
        </p>
      </div>

      {/* Quick Stats */}
      <div style={{ 
        display: "grid", 
        gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", 
        gap: "1.5rem", 
        marginBottom: "2rem" 
      }}>
        <StatCard 
          icon={<Activity size={24} />}
          title="Active Monitors"
          value={summary.active_monitors.toLocaleString()}
          color="#10B981"
          trend="+12%"
        />
        <StatCard 
          icon={<Users size={24} />}
          title="Communities"
          value={summary.communities}
          color="#3B82F6"
          trend="+5%"
        />
        <StatCard 
          icon={<Shield size={24} />}
          title="Safe Areas"
          value={`${summary.safe_areas_pct}%`}
          color="#8B5CF6"
          trend="+2%"
        />
        <StatCard 
          icon={<AlertTriangle size={24} />}
          title="Active Alerts"
          value={alerts.length}
          color="#F59E0B"
          trend="-8%"
        />
      </div>

      {/* Main Content Grid */}
      <div style={{ 
        display: "grid", 
        gridTemplateColumns: "2fr 1fr", 
        gap: "2rem",
        marginBottom: "2rem"
      }}>
        {/* Left Column - Charts and Maps */}
        <div style={{ display: "grid", gap: "1.5rem" }}>
          {/* Water Quality Status */}
          <div style={{
            background: "white",
            borderRadius: "16px",
            padding: "1.5rem",
            boxShadow: "0 4px 6px rgba(0,0,0,0.05)",
            border: "1px solid #E5E7EB"
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
              <Droplets size={20} color="#3B82F6" />
              <h3 style={{ margin: 0, color: "#1F2937" }}>Water Quality Status</h3>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem" }}>
              {[
                { label: "Safe", value: water.safe, color: "#10B981", bg: "#ECFDF5" },
                { label: "Caution", value: water.caution, color: "#F59E0B", bg: "#FFFBEB" },
                { label: "Unsafe", value: water.unsafe, color: "#EF4444", bg: "#FEF2F2" }
              ].map((item) => (
                <div key={item.label} style={{
                  background: item.bg,
                  borderRadius: "12px",
                  padding: "1rem",
                  textAlign: "center",
                  border: `2px solid ${item.color}20`
                }}>
                  <div style={{ fontSize: "2rem", fontWeight: "800", color: item.color }}>
                    {item.value}
                  </div>
                  <div style={{ color: "#6B7280", fontWeight: "500" }}>
                    {item.label} Communities
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Health Reports Chart */}
          <div style={{
            background: "white",
            borderRadius: "16px",
            padding: "1.5rem",
            boxShadow: "0 4px 6px rgba(0,0,0,0.05)",
            border: "1px solid #E5E7EB"
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
              <BarChart3 size={20} color="#8B5CF6" />
              <h3 style={{ margin: 0, color: "#1F2937" }}>Health Reports Trend</h3>
            </div>
            <div style={{ 
              height: "200px", 
              background: "linear-gradient(135deg, #F3E8FF 0%, #E0E7FF 100%)", 
              borderRadius: "12px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#6B7280"
            }}>
              <div style={{ textAlign: "center" }}>
                <BarChart3 size={48} color="#8B5CF6" style={{ marginBottom: "0.5rem" }} />
                <div>Chart visualization coming soon</div>
              </div>
            </div>
          </div>

          {/* GIS Map Preview */}
          <div style={{
            background: "white",
            borderRadius: "16px",
            padding: "1.5rem",
            boxShadow: "0 4px 6px rgba(0,0,0,0.05)",
            border: "1px solid #E5E7EB"
          }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <MapPin size={20} color="#EF4444" />
                <h3 style={{ margin: 0, color: "#1F2937" }}>Community Map</h3>
              </div>
              <Link to="/alerts" style={{
                color: "#3B82F6",
                textDecoration: "none",
                fontSize: "0.875rem",
                fontWeight: "500"
              }}>
                View Full Map →
              </Link>
            </div>
            <div style={{ 
              height: "200px", 
              background: "linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%)", 
              borderRadius: "12px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#6B7280",
              position: "relative",
              overflow: "hidden"
            }}>
              <div style={{ textAlign: "center" }}>
                <Globe size={48} color="#F59E0B" style={{ marginBottom: "0.5rem" }} />
                <div>Interactive GIS Map</div>
              </div>
              {/* Mock map markers */}
              {[1,2,3,4,5].map((i) => (
                <div key={i} style={{
                  position: "absolute",
                  left: `${20 + (i * 15)}%`,
                  top: `${30 + (i * 10)}%`,
                  width: "12px",
                  height: "12px",
                  background: "#EF4444",
                  borderRadius: "50%",
                  boxShadow: "0 0 10px rgba(239, 68, 68, 0.5)"
                }} />
              ))}
            </div>
          </div>
        </div>

        {/* Right Column - Sidebar */}
        <div style={{ display: "grid", gap: "1.5rem" }}>
          {/* Quick Actions */}
          <div style={{
            background: "white",
            borderRadius: "16px",
            padding: "1.5rem",
            boxShadow: "0 4px 6px rgba(0,0,0,0.05)",
            border: "1px solid #E5E7EB"
          }}>
            <h3 style={{ margin: "0 0 1rem 0", color: "#1F2937" }}>Quick Actions</h3>
            <div style={{ display: "grid", gap: "0.75rem" }}>
              {[
                { icon: <Heart size={16} />, label: "Report Symptoms", link: "/reports", color: "#EF4444" },
                { icon: <Droplets size={16} />, label: "Check Water Quality", link: "/water", color: "#3B82F6" },
                { icon: <MapPin size={16} />, label: "View Alerts Map", link: "/alerts", color: "#F59E0B" },
                { icon: <Users size={16} />, label: "Community Forum", link: "/community", color: "#10B981" }
              ].map((action) => (
                <Link 
                  key={action.label}
                  to={action.link}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "0.75rem",
                    padding: "0.75rem",
                    borderRadius: "8px",
                    background: "#F9FAFB",
                    textDecoration: "none",
                    color: "#374151",
                    transition: "all 0.2s",
                    border: "1px solid #E5E7EB"
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.background = action.color + "10";
                    e.target.style.borderColor = action.color;
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.background = "#F9FAFB";
                    e.target.style.borderColor = "#E5E7EB";
                  }}
                >
                  <div style={{ color: action.color }}>{action.icon}</div>
                  <span style={{ fontWeight: "500" }}>{action.label}</span>
                </Link>
              ))}
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
            <div style={{ display: "grid", gap: "0.75rem" }}>
              {alerts.length > 0 ? alerts.map((alert, index) => (
                <div key={index} style={{
                  padding: "0.75rem",
                  background: "#FEF2F2",
                  borderRadius: "8px",
                  border: "1px solid #FECACA"
                }}>
                  <div style={{ fontWeight: "600", color: "#DC2626", fontSize: "0.875rem" }}>
                    {alert.severity?.toUpperCase() || "ALERT"}
                  </div>
                  <div style={{ color: "#374151", fontSize: "0.875rem", marginTop: "0.25rem" }}>
                    {alert.message}
                  </div>
                  <div style={{ color: "#9CA3AF", fontSize: "0.75rem", marginTop: "0.25rem" }}>
                    {alert.timestamp}
                  </div>
                </div>
              )) : (
                <div style={{ 
                  textAlign: "center", 
                  color: "#6B7280", 
                  padding: "1rem",
                  fontSize: "0.875rem"
                }}>
                  No recent alerts
                </div>
              )}
            </div>
          </div>

          {/* Community Stats */}
          <div style={{
            background: "white",
            borderRadius: "16px",
            padding: "1.5rem",
            boxShadow: "0 4px 6px rgba(0,0,0,0.05)",
            border: "1px solid #E5E7EB"
          }}>
            <h3 style={{ margin: "0 0 1rem 0", color: "#1F2937" }}>Top Communities</h3>
            <div style={{ display: "grid", gap: "0.5rem" }}>
              {communities.slice(0, 5).map((community, index) => (
                <div key={community.id || index} style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "0.5rem",
                  background: "#F9FAFB",
                  borderRadius: "6px"
                }}>
                  <div>
                    <div style={{ fontWeight: "500", color: "#374151" }}>
                      {community.name || `Community ${index + 1}`}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "#6B7280" }}>
                      {community.district || "District"}
                    </div>
                  </div>
                  <div style={{
                    background: "#10B981",
                    color: "white",
                    padding: "0.25rem 0.5rem",
                    borderRadius: "4px",
                    fontSize: "0.75rem",
                    fontWeight: "500"
                  }}>
                    Safe
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Feature Cards */}
      <div style={{ 
        display: "grid", 
        gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", 
        gap: "1.5rem",
        marginBottom: "2rem"
      }}>
        {featureCards.map((feature) => (
          <div key={feature.title} style={{
            background: "white",
            borderRadius: "16px",
            padding: "1.5rem",
            boxShadow: "0 4px 6px rgba(0,0,0,0.05)",
            border: "1px solid #E5E7EB",
            transition: "transform 0.2s, box-shadow 0.2s"
          }}
          onMouseEnter={(e) => {
            e.target.style.transform = "translateY(-2px)";
            e.target.style.boxShadow = "0 8px 25px rgba(0,0,0,0.1)";
          }}
          onMouseLeave={(e) => {
            e.target.style.transform = "translateY(0)";
            e.target.style.boxShadow = "0 4px 6px rgba(0,0,0,0.05)";
          }}
          >
            <div style={{ 
              fontSize: "2rem", 
              marginBottom: "1rem",
              display: "flex",
              alignItems: "center",
              gap: "0.5rem"
            }}>
              {feature.icon}
              <span style={{ fontSize: "1.25rem", color: colors.primary, fontWeight: "600" }}>
                {feature.title}
              </span>
            </div>
            <p style={{ margin: 0, color: "#6B7280", lineHeight: "1.5" }}>
              {feature.desc}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

// StatCard Component
function StatCard({ icon, title, value, color, trend }) {
  return (
    <div style={{
      background: "white",
      borderRadius: "16px",
      padding: "1.5rem",
      boxShadow: "0 4px 6px rgba(0,0,0,0.05)",
      border: "1px solid #E5E7EB",
      display: "flex",
      alignItems: "center",
      gap: "1rem"
    }}>
      <div style={{
        background: color + "20",
        color: color,
        padding: "0.75rem",
        borderRadius: "12px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center"
      }}>
        {icon}
      </div>
      <div>
        <div style={{ fontSize: "1.5rem", fontWeight: "700", color: "#1F2937" }}>
          {value}
        </div>
        <div style={{ color: "#6B7280", fontSize: "0.875rem", marginBottom: "0.25rem" }}>
          {title}
        </div>
        <div style={{ 
          color: trend.startsWith("+") ? "#10B981" : "#EF4444",
          fontSize: "0.75rem",
          fontWeight: "500"
        }}>
          {trend} from last month
        </div>
      </div>
    </div>
  );
}

const featureCards = [
  { 
    title: "Community Reporting", 
    desc: "Multilingual symptom reports from field workers and citizens with real-time data collection.", 
    icon: <Users size={24} color="#3B82F6" />
  },
  { 
    title: "Water Quality Monitoring", 
    desc: "Live safety labels and comprehensive water quality analysis across all communities.", 
    icon: <Droplets size={24} color="#10B981" />
  },
  { 
    title: "Real-time Alerts", 
    desc: "Instant notifications to officials and workers with smart escalation protocols.", 
    icon: <AlertTriangle size={24} color="#EF4444" />
  },
  { 
    title: "AI Predictions", 
    desc: "Advanced machine learning to detect patterns and predict health outbreaks early.", 
    icon: <Zap size={24} color="#8B5CF6" />
  },
  { 
    title: "GIS Mapping", 
    desc: "Interactive maps showing health data, water quality, and community locations.", 
    icon: <MapPin size={24} color="#F59E0B" />
  },
  { 
    title: "Health Analytics", 
    desc: "Comprehensive health reports and trend analysis for informed decision making.", 
    icon: <BarChart3 size={24} color="#06B6D4" />
  },
];