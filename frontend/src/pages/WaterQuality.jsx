import {
    Activity,
    AlertTriangle,
    CheckCircle,
    Clock,
    Droplets,
    Filter,
    MapPin,
    Plus,
    Search,
    TestTube,
    Thermometer,
    Upload
} from "lucide-react";
import React, { useContext, useEffect, useState } from "react";
import toast from "react-hot-toast";
import { AuthContext } from "../auth/AuthContext";
import { fetchWaterReports, sendEmergencyAlert, submitWaterReport, uploadWaterData } from "../services/api";

export default function WaterQuality() {
  const { user } = useContext(AuthContext);
  const [reports, setReports] = useState([]);
  const [filteredReports, setFilteredReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [showAddForm, setShowAddForm] = useState(false);
  const [showUploadForm, setShowUploadForm] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [newReport, setNewReport] = useState({
    location: "",
    ph: "",
    turbidity: "",
    temperature: "",
    chlorine: "",
    bacteria: "",
    notes: ""
  });

  useEffect(() => {
    loadReports();
  }, []);

  useEffect(() => {
    filterReports();
  }, [reports, searchTerm, statusFilter]);

  const loadReports = async () => {
    try {
      const data = await fetchWaterReports();
      const normalized = Array.isArray(data) ? data.map((r, idx) => ({
        id: r.id || r._id || idx,
        location: r.location || "Unknown",
        status: r.status || getWaterStatus(r.ph, r.turbidity),
        ph: r.ph,
        turbidity: r.turbidity,
        temperature: r.temp ?? r.temperature,
        chlorine: r.tds ? (Number(r.tds) / 100).toFixed(2) : r.chlorine,
        bacteria: r.bacteria || "",
        timestamp: r.timestamp || new Date().toISOString(),
        testedBy: r.testedBy || "System",
      })) : [];
      setReports(normalized);
    } catch (error) {
      console.error("Error loading water reports:", error);
      // Mock data for demonstration
      setReports([
        {
          id: 1,
          location: "Community A - Main Well",
          status: "safe",
          ph: 7.2,
          turbidity: 0.5,
          temperature: 22,
          chlorine: 0.8,
          bacteria: "negative",
          timestamp: "2024-01-15T10:30:00Z",
          testedBy: "Dr. Sarah Johnson",
          coordinates: { lat: 17.3850, lng: 78.4867 }
        },
        {
          id: 2,
          location: "Community B - Water Tank",
          status: "caution",
          ph: 6.8,
          turbidity: 2.1,
          temperature: 25,
          chlorine: 0.3,
          bacteria: "positive",
          timestamp: "2024-01-15T09:15:00Z",
          testedBy: "Health Worker Mike",
          coordinates: { lat: 17.3950, lng: 78.4967 }
        },
        {
          id: 3,
          location: "Community C - Bore Well",
          status: "unsafe",
          ph: 5.5,
          turbidity: 4.2,
          temperature: 28,
          chlorine: 0.1,
          bacteria: "positive",
          timestamp: "2024-01-15T08:45:00Z",
          testedBy: "Field Officer Lisa",
          coordinates: { lat: 17.3750, lng: 78.4767 }
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const filterReports = () => {
    let filtered = reports;

    if (searchTerm) {
      filtered = filtered.filter(report =>
        report.location.toLowerCase().includes(searchTerm.toLowerCase()) ||
        report.testedBy.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (statusFilter !== "all") {
      filtered = filtered.filter(report => report.status === statusFilter);
    }

    setFilteredReports(filtered);
  };

  const handleAddReport = async () => {
    if (!newReport.location || !newReport.ph || !newReport.turbidity) {
      toast.error("Please fill in all required fields");
      return;
    }

    const report = {
      id: Date.now(),
      ...newReport,
      status: getWaterStatus(newReport.ph, newReport.turbidity),
      timestamp: new Date().toISOString(),
      testedBy: user?.displayName || "Unknown",
      coordinates: { lat: 17.3850, lng: 78.4867 }
    };

    try {
      const payload = {
        sensor_id: `manual-${Date.now()}`,
        location: report.location,
        ph: parseFloat(report.ph),
        turbidity: parseFloat(report.turbidity),
        temp: report.temperature ? parseFloat(report.temperature) : undefined,
        tds: report.chlorine ? parseFloat(report.chlorine) * 100 : undefined
      };
      const saved = await submitWaterReport(payload);
      const mapped = {
        id: saved.id || saved._id || report.id,
        location: saved.location || report.location,
        status: saved.status || getWaterStatus(saved.ph, saved.turbidity),
        ph: saved.ph ?? report.ph,
        turbidity: saved.turbidity ?? report.turbidity,
        temperature: saved.temp ?? saved.temperature ?? report.temperature,
        chlorine: saved.tds ? (Number(saved.tds) / 100).toFixed(2) : saved.chlorine ?? report.chlorine,
        bacteria: saved.bacteria ?? report.bacteria,
        timestamp: saved.timestamp || report.timestamp,
        testedBy: report.testedBy,
      };
      setReports([mapped, ...reports]);
    } catch (e) {
      setReports([report, ...reports]);
    }
    
    // Check if this is a high alert and trigger emergency response
    if (report.status === "unsafe") {
      await triggerEmergencyAlert(report);
    }

    setNewReport({
      location: "",
      ph: "",
      turbidity: "",
      temperature: "",
      chlorine: "",
      bacteria: "",
      notes: ""
    });
    setShowAddForm(false);
    toast.success("Water quality report added successfully!");
  };

  const triggerEmergencyAlert = async (report) => {
    try {
      const alertData = {
        type: "water_quality_emergency",
        severity: "high",
        location: report.location,
        message: `URGENT: Unsafe water quality detected at ${report.location}. pH: ${report.ph}, Turbidity: ${report.turbidity} NTU. Immediate action required!`,
        coordinates: report.coordinates,
        testedBy: report.testedBy,
        timestamp: new Date().toISOString()
      };

      // Send to officials and citizens
      await sendEmergencyAlert(alertData);
      toast.error("HIGH ALERT: Emergency notifications sent to officials and citizens!");
    } catch (error) {
      console.error("Error sending emergency alert:", error);
    }
  };

  const handleFileUpload = async (e) => {
    e.preventDefault();
    if (!uploadFile) {
      toast.error("Please select a file to upload");
      return;
    }

    try {
      const formData = new FormData();
      formData.append('file', uploadFile);
      formData.append('type', 'water_quality');
      formData.append('uploadedBy', user?.displayName || 'Unknown');

      const result = await uploadWaterData(formData);
      
      if (result.success) {
        toast.success(`Successfully processed ${result.processedCount} water quality records`);
        setUploadFile(null);
        setShowUploadForm(false);
        loadReports(); // Reload data
      } else {
        toast.error("Error processing file: " + result.error);
      }
    } catch (error) {
      toast.error("Error uploading file: " + error.message);
    }
  };

  const getWaterStatus = (ph, turbidity) => {
    const phValue = parseFloat(ph);
    const turbidityValue = parseFloat(turbidity);
    
    if (phValue >= 6.5 && phValue <= 8.5 && turbidityValue <= 1.0) {
      return "safe";
    } else if (phValue >= 6.0 && phValue <= 9.0 && turbidityValue <= 4.0) {
      return "caution";
    } else {
      return "unsafe";
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "safe": return "#10B981";
      case "caution": return "#F59E0B";
      case "unsafe": return "#EF4444";
      default: return "#6B7280";
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "safe": return <CheckCircle size={20} color="#10B981" />;
      case "caution": return <AlertTriangle size={20} color="#F59E0B" />;
      case "unsafe": return <AlertTriangle size={20} color="#EF4444" />;
      default: return <Clock size={20} color="#6B7280" />;
    }
  };

  const canAccessFeature = (feature) => {
    const userRole = user?.role;
    const rolePermissions = {
      'admin': ['all'],
      'officer': ['view', 'add', 'upload', 'alert'],
      'worker': ['view', 'add', 'upload'],
      'community': ['view']
    };
    
    return rolePermissions[userRole]?.includes(feature) || rolePermissions[userRole]?.includes('all');
  };

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
        Loading water quality reports...
      </div>
    );
  }

  return (
    <div style={{ padding: "1.5rem", maxWidth: "1400px", margin: "0 auto" }}>
      {/* Header */}
      <div style={{ marginBottom: "2rem" }}>
        <h1 style={{ 
          margin: 0, 
          fontSize: "2rem", 
          fontWeight: "700", 
          color: "#1F2937",
          display: "flex",
          alignItems: "center",
          gap: "0.5rem"
        }}>
          <Droplets size={32} color="#3B82F6" />
          Water Quality Monitoring
        </h1>
        <p style={{ 
          margin: "0.5rem 0 0 0", 
          color: "#6B7280",
          fontSize: "1.1rem"
        }}>
          Monitor and track water quality across all communities with AI-powered alerts
        </p>
      </div>

      {/* Summary Cards */}
      <div style={{ 
        display: "grid", 
        gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", 
        gap: "1.5rem", 
        marginBottom: "2rem" 
      }}>
        {[
          { 
            label: "Total Reports", 
            value: reports.length, 
            icon: <TestTube size={24} color="#3B82F6" />,
            color: "#3B82F6"
          },
          { 
            label: "Safe Sources", 
            value: reports.filter(r => r.status === "safe").length, 
            icon: <CheckCircle size={24} color="#10B981" />,
            color: "#10B981"
          },
          { 
            label: "Caution Sources", 
            value: reports.filter(r => r.status === "caution").length, 
            icon: <AlertTriangle size={24} color="#F59E0B" />,
            color: "#F59E0B"
          },
          { 
            label: "Unsafe Sources", 
            value: reports.filter(r => r.status === "unsafe").length, 
            icon: <AlertTriangle size={24} color="#EF4444" />,
            color: "#EF4444"
          }
        ].map((stat) => (
          <div key={stat.label} style={{
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
              background: stat.color + "20",
              color: stat.color,
              padding: "0.75rem",
              borderRadius: "12px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center"
            }}>
              {stat.icon}
            </div>
            <div>
              <div style={{ fontSize: "1.5rem", fontWeight: "700", color: "#1F2937" }}>
                {stat.value}
              </div>
              <div style={{ color: "#6B7280", fontSize: "0.875rem" }}>
                {stat.label}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Controls */}
      <div style={{ 
        display: "flex", 
        justifyContent: "space-between", 
        alignItems: "center", 
        marginBottom: "1.5rem",
        flexWrap: "wrap",
        gap: "1rem"
      }}>
        <div style={{ display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ position: "relative" }}>
            <Search size={20} style={{ 
              position: "absolute", 
              left: "12px", 
              top: "50%", 
              transform: "translateY(-50%)",
              color: "#9CA3AF"
            }} />
            <input
              type="text"
              placeholder="Search locations..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                padding: "0.75rem 0.75rem 0.75rem 40px",
                borderRadius: "8px",
                border: "1px solid #D1D5DB",
                fontSize: "14px",
                width: "250px"
              }}
            />
          </div>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            style={{
              padding: "0.75rem",
              borderRadius: "8px",
              border: "1px solid #D1D5DB",
              fontSize: "14px",
              background: "white"
            }}
          >
            <option value="all">All Status</option>
            <option value="safe">Safe</option>
            <option value="caution">Caution</option>
            <option value="unsafe">Unsafe</option>
          </select>
        </div>
        <div style={{ display: "flex", gap: "1rem" }}>
          {canAccessFeature('upload') && (
            <button
              onClick={() => setShowUploadForm(!showUploadForm)}
              style={{
                background: "linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%)",
                color: "white",
                border: "none",
                borderRadius: "8px",
                padding: "0.75rem 1rem",
                fontSize: "14px",
                fontWeight: "600",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
                transition: "transform 0.2s"
              }}
              onMouseEnter={(e) => e.target.style.transform = "translateY(-1px)"}
              onMouseLeave={(e) => e.target.style.transform = "translateY(0)"}
            >
              <Upload size={16} />
              Upload CSV
            </button>
          )}
          {canAccessFeature('add') && (
            <button
              onClick={() => setShowAddForm(!showAddForm)}
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
                gap: "0.5rem",
                transition: "transform 0.2s"
              }}
              onMouseEnter={(e) => e.target.style.transform = "translateY(-1px)"}
              onMouseLeave={(e) => e.target.style.transform = "translateY(0)"}
            >
              <Plus size={16} />
              Add Report
            </button>
          )}
        </div>
      </div>

      {/* File Upload Form */}
      {showUploadForm && canAccessFeature('upload') && (
        <div style={{
          background: "white",
          borderRadius: "16px",
          padding: "1.5rem",
          boxShadow: "0 4px 6px rgba(0,0,0,0.05)",
          border: "1px solid #E5E7EB",
          marginBottom: "1.5rem"
        }}>
          <h3 style={{ margin: "0 0 1rem 0", color: "#1F2937", display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Upload size={20} color="#8B5CF6" />
            Upload Water Quality Data
          </h3>
          <p style={{ color: "#6B7280", margin: "0 0 1rem 0", fontSize: "0.875rem" }}>
            Upload CSV files with water quality data. The system will automatically process and generate alerts for unsafe conditions.
          </p>
          <form onSubmit={handleFileUpload}>
            <div style={{ marginBottom: "1rem" }}>
              <label style={{ display: "block", marginBottom: "0.5rem", color: "#374151", fontWeight: "500" }}>
                Select CSV File
              </label>
              <input
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={(e) => setUploadFile(e.target.files[0])}
                style={{
                  width: "100%",
                  padding: "0.75rem",
                  borderRadius: "8px",
                  border: "1px solid #D1D5DB",
                  fontSize: "14px"
                }}
              />
            </div>
            <div style={{ display: "flex", gap: "1rem" }}>
              <button
                type="submit"
                style={{
                  background: "linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%)",
                  color: "white",
                  border: "none",
                  borderRadius: "8px",
                  padding: "0.75rem 1.5rem",
                  fontSize: "14px",
                  fontWeight: "600",
                  cursor: "pointer"
                }}
              >
                Process File
              </button>
              <button
                type="button"
                onClick={() => setShowUploadForm(false)}
                style={{
                  background: "transparent",
                  color: "#6B7280",
                  border: "1px solid #D1D5DB",
                  borderRadius: "8px",
                  padding: "0.75rem 1.5rem",
                  fontSize: "14px",
                  fontWeight: "600",
                  cursor: "pointer"
                }}
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Add Report Form */}
      {showAddForm && canAccessFeature('add') && (
        <div style={{
          background: "white",
          borderRadius: "16px",
          padding: "1.5rem",
          boxShadow: "0 4px 6px rgba(0,0,0,0.05)",
          border: "1px solid #E5E7EB",
          marginBottom: "1.5rem"
        }}>
          <h3 style={{ margin: "0 0 1rem 0", color: "#1F2937" }}>Add New Water Quality Report</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem" }}>
            <div>
              <label style={{ display: "block", marginBottom: "0.5rem", color: "#374151", fontWeight: "500" }}>
                Location *
              </label>
              <input
                type="text"
                placeholder="Enter location"
                value={newReport.location}
                onChange={(e) => setNewReport({ ...newReport, location: e.target.value })}
                style={{
                  width: "100%",
                  padding: "0.75rem",
                  borderRadius: "8px",
                  border: "1px solid #D1D5DB",
                  fontSize: "14px"
                }}
              />
            </div>
            <div>
              <label style={{ display: "block", marginBottom: "0.5rem", color: "#374151", fontWeight: "500" }}>
                pH Level *
              </label>
              <input
                type="number"
                step="0.1"
                placeholder="7.0"
                value={newReport.ph}
                onChange={(e) => setNewReport({ ...newReport, ph: e.target.value })}
                style={{
                  width: "100%",
                  padding: "0.75rem",
                  borderRadius: "8px",
                  border: "1px solid #D1D5DB",
                  fontSize: "14px"
                }}
              />
            </div>
            <div>
              <label style={{ display: "block", marginBottom: "0.5rem", color: "#374151", fontWeight: "500" }}>
                Turbidity (NTU) *
              </label>
              <input
                type="number"
                step="0.1"
                placeholder="1.0"
                value={newReport.turbidity}
                onChange={(e) => setNewReport({ ...newReport, turbidity: e.target.value })}
                style={{
                  width: "100%",
                  padding: "0.75rem",
                  borderRadius: "8px",
                  border: "1px solid #D1D5DB",
                  fontSize: "14px"
                }}
              />
            </div>
            <div>
              <label style={{ display: "block", marginBottom: "0.5rem", color: "#374151", fontWeight: "500" }}>
                Temperature (°C)
              </label>
              <input
                type="number"
                step="0.1"
                placeholder="25"
                value={newReport.temperature}
                onChange={(e) => setNewReport({ ...newReport, temperature: e.target.value })}
                style={{
                  width: "100%",
                  padding: "0.75rem",
                  borderRadius: "8px",
                  border: "1px solid #D1D5DB",
                  fontSize: "14px"
                }}
              />
            </div>
            <div>
              <label style={{ display: "block", marginBottom: "0.5rem", color: "#374151", fontWeight: "500" }}>
                Chlorine (mg/L)
              </label>
              <input
                type="number"
                step="0.1"
                placeholder="0.5"
                value={newReport.chlorine}
                onChange={(e) => setNewReport({ ...newReport, chlorine: e.target.value })}
                style={{
                  width: "100%",
                  padding: "0.75rem",
                  borderRadius: "8px",
                  border: "1px solid #D1D5DB",
                  fontSize: "14px"
                }}
              />
            </div>
            <div>
              <label style={{ display: "block", marginBottom: "0.5rem", color: "#374151", fontWeight: "500" }}>
                Bacteria Test
              </label>
              <select
                value={newReport.bacteria}
                onChange={(e) => setNewReport({ ...newReport, bacteria: e.target.value })}
                style={{
                  width: "100%",
                  padding: "0.75rem",
                  borderRadius: "8px",
                  border: "1px solid #D1D5DB",
                  fontSize: "14px"
                }}
              >
                <option value="">Select result</option>
                <option value="negative">Negative</option>
                <option value="positive">Positive</option>
              </select>
            </div>
          </div>
          <div style={{ marginTop: "1rem" }}>
            <label style={{ display: "block", marginBottom: "0.5rem", color: "#374151", fontWeight: "500" }}>
              Notes
            </label>
            <textarea
              placeholder="Additional observations..."
              value={newReport.notes}
              onChange={(e) => setNewReport({ ...newReport, notes: e.target.value })}
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
          <div style={{ display: "flex", gap: "1rem", marginTop: "1rem" }}>
            <button
              onClick={handleAddReport}
              style={{
                background: "linear-gradient(135deg, #10B981 0%, #059669 100%)",
                color: "white",
                border: "none",
                borderRadius: "8px",
                padding: "0.75rem 1.5rem",
                fontSize: "14px",
                fontWeight: "600",
                cursor: "pointer"
              }}
            >
              Save Report
            </button>
            <button
              onClick={() => setShowAddForm(false)}
              style={{
                background: "transparent",
                color: "#6B7280",
                border: "1px solid #D1D5DB",
                borderRadius: "8px",
                padding: "0.75rem 1.5rem",
                fontSize: "14px",
                fontWeight: "600",
                cursor: "pointer"
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Reports List */}
      <div style={{ display: "grid", gap: "1rem" }}>
        {filteredReports.length === 0 ? (
          <div style={{
            background: "white",
            borderRadius: "16px",
            padding: "3rem",
            textAlign: "center",
            boxShadow: "0 4px 6px rgba(0,0,0,0.05)",
            border: "1px solid #E5E7EB"
          }}>
            <Droplets size={48} color="#9CA3AF" style={{ marginBottom: "1rem" }} />
            <h3 style={{ color: "#6B7280", margin: "0 0 0.5rem 0" }}>No water quality reports found</h3>
            <p style={{ color: "#9CA3AF", margin: 0 }}>Try adjusting your search or add a new report</p>
          </div>
        ) : (
          filteredReports.map((report) => (
            <div key={report.id} style={{
              background: "white",
              borderRadius: "16px",
              padding: "1.5rem",
              boxShadow: "0 4px 6px rgba(0,0,0,0.05)",
              border: "1px solid #E5E7EB",
              transition: "transform 0.2s, box-shadow 0.2s",
              borderLeft: report.status === "unsafe" ? "4px solid #EF4444" : 
                         report.status === "caution" ? "4px solid #F59E0B" : "4px solid #10B981"
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
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
                <div>
                  <h3 style={{ margin: "0 0 0.5rem 0", color: "#1F2937", fontSize: "1.25rem" }}>
                    {report.location}
                  </h3>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "#6B7280", fontSize: "0.875rem" }}>
                    <MapPin size={16} />
                    <span>Tested by {report.testedBy}</span>
                    <Clock size={16} style={{ marginLeft: "1rem" }} />
                    <span>{new Date(report.timestamp).toLocaleDateString()}</span>
                  </div>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  {getStatusIcon(report.status)}
                  <span style={{ 
                    color: getStatusColor(report.status), 
                    fontWeight: "600",
                    textTransform: "uppercase",
                    fontSize: "0.875rem"
                  }}>
                    {report.status}
                  </span>
                  {report.status === "unsafe" && (
                    <div style={{
                      background: "#FEF2F2",
                      color: "#DC2626",
                      padding: "0.25rem 0.5rem",
                      borderRadius: "4px",
                      fontSize: "0.75rem",
                      fontWeight: "600"
                    }}>
                      EMERGENCY
                    </div>
                  )}
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: "1rem" }}>
                <div style={{ textAlign: "center", padding: "0.75rem", background: "#F9FAFB", borderRadius: "8px" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.25rem", marginBottom: "0.25rem" }}>
                    <Activity size={16} color="#3B82F6" />
                    <span style={{ fontSize: "0.875rem", color: "#6B7280" }}>pH</span>
                  </div>
                  <div style={{ fontSize: "1.25rem", fontWeight: "700", color: "#1F2937" }}>
                    {report.ph}
                  </div>
                </div>
                <div style={{ textAlign: "center", padding: "0.75rem", background: "#F9FAFB", borderRadius: "8px" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.25rem", marginBottom: "0.25rem" }}>
                    <Filter size={16} color="#10B981" />
                    <span style={{ fontSize: "0.875rem", color: "#6B7280" }}>Turbidity</span>
                  </div>
                  <div style={{ fontSize: "1.25rem", fontWeight: "700", color: "#1F2937" }}>
                    {report.turbidity} NTU
                  </div>
                </div>
                <div style={{ textAlign: "center", padding: "0.75rem", background: "#F9FAFB", borderRadius: "8px" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.25rem", marginBottom: "0.25rem" }}>
                    <Thermometer size={16} color="#F59E0B" />
                    <span style={{ fontSize: "0.875rem", color: "#6B7280" }}>Temperature</span>
                  </div>
                  <div style={{ fontSize: "1.25rem", fontWeight: "700", color: "#1F2937" }}>
                    {report.temperature}°C
                  </div>
                </div>
                <div style={{ textAlign: "center", padding: "0.75rem", background: "#F9FAFB", borderRadius: "8px" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.25rem", marginBottom: "0.25rem" }}>
                    <TestTube size={16} color="#8B5CF6" />
                    <span style={{ fontSize: "0.875rem", color: "#6B7280" }}>Chlorine</span>
                  </div>
                  <div style={{ fontSize: "1.25rem", fontWeight: "700", color: "#1F2937" }}>
                    {report.chlorine} mg/L
                  </div>
                </div>
                <div style={{ textAlign: "center", padding: "0.75rem", background: "#F9FAFB", borderRadius: "8px" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.25rem", marginBottom: "0.25rem" }}>
                    <Activity size={16} color="#EF4444" />
                    <span style={{ fontSize: "0.875rem", color: "#6B7280" }}>Bacteria</span>
                  </div>
                  <div style={{ 
                    fontSize: "1.25rem", 
                    fontWeight: "700", 
                    color: report.bacteria === "negative" ? "#10B981" : "#EF4444"
                  }}>
                    {report.bacteria || "N/A"}
                  </div>
                </div>
              </div>

              {report.notes && (
                <div style={{ marginTop: "1rem", padding: "0.75rem", background: "#F0F9FF", borderRadius: "8px", border: "1px solid #BAE6FD" }}>
                  <div style={{ fontSize: "0.875rem", color: "#0369A1", fontWeight: "500", marginBottom: "0.25rem" }}>
                    Notes:
                  </div>
                  <div style={{ fontSize: "0.875rem", color: "#0C4A6E" }}>
                    {report.notes}
                  </div>
                </div>
              )}

              {report.status === "unsafe" && (
                <div style={{ 
                  marginTop: "1rem", 
                  padding: "1rem", 
                  background: "#FEF2F2", 
                  borderRadius: "8px", 
                  border: "1px solid #FECACA",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem"
                }}>
                  <AlertTriangle size={20} color="#DC2626" />
                  <div>
                    <div style={{ fontSize: "0.875rem", color: "#DC2626", fontWeight: "600", marginBottom: "0.25rem" }}>
                      Emergency Alert Sent
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "#991B1B" }}>
                      Officials and citizens have been notified via SMS and calls. Immediate action required!
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
