import React, { useEffect, useState } from 'react';
import Card from '../components/Card';

const HealthReports = () => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showSubmitForm, setShowSubmitForm] = useState(false);
  const [filters, setFilters] = useState({
    status: 'all',
    severity: 'all'
  });
  const [newReport, setNewReport] = useState({
    patientName: '',
    village: '',
    symptoms: '',
    severity: 'medium',
    description: '',
    age: '',
    gender: 'male'
  });

  // Sample data matching the reference design
  const sampleReports = [
    {
      id: 'RPT-001',
      patient: 'John Doe',
      village: 'Riverside Village',
      symptoms: ['Fever', 'Headache'],
      severity: 'medium',
      status: 'under review',
      date: '2024-01-15',
      actions: []
    },
    {
      id: 'RPT-002',
      patient: 'Maria Santos',
      village: 'Mountain View',
      symptoms: ['Diarrhea', 'Vomiting'],
      severity: 'high',
      status: 'confirmed',
      date: '2024-01-14',
      actions: []
    },
    {
      id: 'RPT-003',
      patient: 'Ahmed Hassan',
      village: 'Green Valley',
      symptoms: ['Stomach Pain', 'Fever'],
      severity: 'low',
      status: 'resolved',
      date: '2024-01-13',
      actions: []
    },
    {
      id: 'RPT-004',
      patient: 'Sarah Johnson',
      village: 'Sunset Hills',
      symptoms: ['Skin Rash', 'Itching'],
      severity: 'medium',
      status: 'investigating',
      date: '2024-01-12',
      actions: []
    }
  ];

  useEffect(() => {
    // Simulate API call
    setTimeout(() => {
      setReports(sampleReports);
      setLoading(false);
    }, 1000);
  }, []);

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return { backgroundColor: '#fee2e2', color: '#dc2626' };
      case 'medium': return { backgroundColor: '#fef3c7', color: '#d97706' };
      case 'low': return { backgroundColor: '#dcfce7', color: '#16a34a' };
      default: return { backgroundColor: '#f3f4f6', color: '#374151' };
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'confirmed': return { backgroundColor: '#dcfce7', color: '#16a34a' };
      case 'under review': return { backgroundColor: '#dbeafe', color: '#2563eb' };
      case 'investigating': return { backgroundColor: '#fef3c7', color: '#d97706' };
      case 'resolved': return { backgroundColor: '#f3f4f6', color: '#374151' };
      default: return { backgroundColor: '#f3f4f6', color: '#374151' };
    }
  };

  const handleSubmitReport = () => {
    const report = {
      id: `RPT-${String(reports.length + 1).padStart(3, '0')}`,
      patient: newReport.patientName,
      village: newReport.village,
      symptoms: newReport.symptoms.split(',').map(s => s.trim()),
      severity: newReport.severity,
      status: 'under review',
      date: new Date().toISOString().split('T')[0],
      actions: []
    };
    
    setReports([report, ...reports]);
    setShowSubmitForm(false);
    setNewReport({
      patientName: '',
      village: '',
      symptoms: '',
      severity: 'medium',
      description: '',
      age: '',
      gender: 'male'
    });
  };

  const filteredReports = reports.filter(report => {
    if (filters.status !== 'all' && report.status !== filters.status) return false;
    if (filters.severity !== 'all' && report.severity !== filters.severity) return false;
    return true;
  });

  const stats = {
    total: reports.length,
    underReview: reports.filter(r => r.status === 'under review').length,
    highSeverity: reports.filter(r => r.severity === 'high').length,
    resolved: reports.filter(r => r.status === 'resolved').length
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '16rem' }}>
        <div style={{ 
          width: '2rem', 
          height: '2rem', 
          border: '2px solid #e5e7eb', 
          borderTop: '2px solid #2563eb', 
          borderRadius: '50%', 
          animation: 'spin 1s linear infinite' 
        }}></div>
      </div>
    );
  }

  return (
    <div style={{ padding: "1.5rem" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
        <div>
          <h1 style={{ fontSize: "2rem", fontWeight: "bold", marginBottom: "0.5rem" }}>Health Reports</h1>
          <p style={{ color: "#666" }}>Manage and track health incident reports</p>
        </div>
        <div style={{ display: "flex", gap: "0.75rem" }}>
          <button style={{
            padding: "0.5rem 1rem",
            color: "#666",
            border: "1px solid #d1d5db",
            borderRadius: "0.5rem",
            backgroundColor: "white",
            cursor: "pointer"
          }}>
            Export Reports
          </button>
          <button
            onClick={() => setShowSubmitForm(true)}
            style={{
              backgroundColor: "#2563eb",
              color: "white",
              padding: "0.5rem 1rem",
              borderRadius: "0.5rem",
              border: "none",
              cursor: "pointer"
            }}
          >
            Submit Report
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1.5rem", marginBottom: "1.5rem" }}>
        <Card style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center" }}>
            <div style={{ padding: "0.75rem", backgroundColor: "#dbeafe", borderRadius: "50%", marginRight: "1rem" }}>
              <svg style={{ width: "1.5rem", height: "1.5rem", color: "#2563eb" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <div>
              <p style={{ fontSize: "0.875rem", fontWeight: "500", color: "#666", margin: 0 }}>Total Reports</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", margin: 0 }}>{stats.total}</p>
            </div>
          </div>
        </Card>

        <Card style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center" }}>
            <div style={{ padding: "0.75rem", backgroundColor: "#fef3c7", borderRadius: "50%", marginRight: "1rem" }}>
              <svg style={{ width: "1.5rem", height: "1.5rem", color: "#d97706" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <p style={{ fontSize: "0.875rem", fontWeight: "500", color: "#666", margin: 0 }}>Under Review</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", margin: 0 }}>{stats.underReview}</p>
            </div>
          </div>
        </Card>

        <Card style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center" }}>
            <div style={{ padding: "0.75rem", backgroundColor: "#fee2e2", borderRadius: "50%", marginRight: "1rem" }}>
              <svg style={{ width: "1.5rem", height: "1.5rem", color: "#dc2626" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <div>
              <p style={{ fontSize: "0.875rem", fontWeight: "500", color: "#666", margin: 0 }}>High Severity</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", margin: 0 }}>{stats.highSeverity}</p>
            </div>
          </div>
        </Card>

        <Card style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center" }}>
            <div style={{ padding: "0.75rem", backgroundColor: "#dcfce7", borderRadius: "50%", marginRight: "1rem" }}>
              <svg style={{ width: "1.5rem", height: "1.5rem", color: "#16a34a" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <p style={{ fontSize: "0.875rem", fontWeight: "500", color: "#666", margin: 0 }}>Resolved</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", margin: 0 }}>{stats.resolved}</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Filters */}
      <Card style={{ padding: "1.5rem", marginBottom: "1.5rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <span style={{ fontSize: "0.875rem", fontWeight: "500", color: "#374151" }}>Filter Reports</span>
          
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <label style={{ fontSize: "0.875rem", color: "#666" }}>Status:</label>
            <select
              value={filters.status}
              onChange={(e) => setFilters({...filters, status: e.target.value})}
              style={{
                padding: "0.25rem 0.5rem",
                border: "1px solid #d1d5db",
                borderRadius: "0.375rem",
                fontSize: "0.875rem"
              }}
            >
              <option value="all">All</option>
              <option value="under review">Under Review</option>
              <option value="confirmed">Confirmed</option>
              <option value="investigating">Investigating</option>
              <option value="resolved">Resolved</option>
            </select>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <label style={{ fontSize: "0.875rem", color: "#666" }}>Severity:</label>
            <select
              value={filters.severity}
              onChange={(e) => setFilters({...filters, severity: e.target.value})}
              style={{
                padding: "0.25rem 0.5rem",
                border: "1px solid #d1d5db",
                borderRadius: "0.375rem",
                fontSize: "0.875rem"
              }}
            >
              <option value="all">All</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>
        </div>
      </Card>

      {/* Reports Table */}
      <Card style={{ padding: "1.5rem" }}>
        <h3 style={{ fontSize: "1.125rem", fontWeight: "600", marginBottom: "1rem" }}>Recent Reports</h3>
        <p style={{ fontSize: "0.875rem", color: "#666", marginBottom: "1.5rem" }}>Latest health incident reports from communities</p>
        
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "separate", borderSpacing: 0 }}>
            <thead style={{ backgroundColor: "#f9fafb" }}>
              <tr>
                <th style={{ padding: "0.75rem", textAlign: "left", fontSize: "0.75rem", fontWeight: "500", color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.05em" }}>Report ID</th>
                <th style={{ padding: "0.75rem", textAlign: "left", fontSize: "0.75rem", fontWeight: "500", color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.05em" }}>Patient</th>
                <th style={{ padding: "0.75rem", textAlign: "left", fontSize: "0.75rem", fontWeight: "500", color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.05em" }}>Village</th>
                <th style={{ padding: "0.75rem", textAlign: "left", fontSize: "0.75rem", fontWeight: "500", color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.05em" }}>Symptoms</th>
                <th style={{ padding: "0.75rem", textAlign: "left", fontSize: "0.75rem", fontWeight: "500", color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.05em" }}>Severity</th>
                <th style={{ padding: "0.75rem", textAlign: "left", fontSize: "0.75rem", fontWeight: "500", color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.05em" }}>Status</th>
                <th style={{ padding: "0.75rem", textAlign: "left", fontSize: "0.75rem", fontWeight: "500", color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.05em" }}>Date</th>
                <th style={{ padding: "0.75rem", textAlign: "left", fontSize: "0.75rem", fontWeight: "500", color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.05em" }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredReports.map((report, index) => (
                <tr key={report.id} style={{ backgroundColor: index % 2 ? "#f9fafb" : "white" }}>
                  <td style={{ padding: "1rem 0.75rem", fontSize: "0.875rem", fontWeight: "500" }}>{report.id}</td>
                  <td style={{ padding: "1rem 0.75rem", fontSize: "0.875rem" }}>{report.patient}</td>
                  <td style={{ padding: "1rem 0.75rem", fontSize: "0.875rem" }}>{report.village}</td>
                  <td style={{ padding: "1rem 0.75rem", fontSize: "0.875rem" }}>{report.symptoms.join(', ')}</td>
                  <td style={{ padding: "1rem 0.75rem" }}>
                    <span style={{ 
                      padding: "0.25rem 0.5rem", 
                      fontSize: "0.75rem", 
                      fontWeight: "500", 
                      borderRadius: "9999px",
                      ...getSeverityColor(report.severity)
                    }}>
                      {report.severity}
                    </span>
                  </td>
                  <td style={{ padding: "1rem 0.75rem" }}>
                    <span style={{ 
                      padding: "0.25rem 0.5rem", 
                      fontSize: "0.75rem", 
                      fontWeight: "500", 
                      borderRadius: "9999px",
                      ...getStatusColor(report.status)
                    }}>
                      {report.status}
                    </span>
                  </td>
                  <td style={{ padding: "1rem 0.75rem", fontSize: "0.875rem" }}>{report.date}</td>
                  <td style={{ padding: "1rem 0.75rem", fontSize: "0.875rem", fontWeight: "500" }}>
                    <button style={{ color: "#2563eb", cursor: "pointer", marginRight: "0.75rem", background: "none", border: "none" }}>View</button>
                    <button style={{ color: "#16a34a", cursor: "pointer", background: "none", border: "none" }}>Update</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Submit Report Modal */}
      {showSubmitForm && (
        <div style={{ 
          position: "fixed", 
          top: 0, 
          left: 0, 
          right: 0, 
          bottom: 0, 
          backgroundColor: "rgba(0,0,0,0.5)", 
          display: "flex", 
          alignItems: "center", 
          justifyContent: "center", 
          zIndex: 50 
        }}>
          <div style={{ 
            backgroundColor: "white", 
            borderRadius: "0.5rem", 
            padding: "1.5rem", 
            width: "100%", 
            maxWidth: "32rem", 
            maxHeight: "90vh", 
            overflowY: "auto" 
          }}>
            <h3 style={{ fontSize: "1.125rem", fontWeight: "600", marginBottom: "1rem" }}>Submit Health Report</h3>
            
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1rem", marginBottom: "1rem" }}>
              <div>
                <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Patient Name</label>
                <input
                  type="text"
                  value={newReport.patientName}
                  onChange={(e) => setNewReport({...newReport, patientName: e.target.value})}
                  style={{
                    width: "100%",
                    padding: "0.5rem 0.75rem",
                    border: "1px solid #d1d5db",
                    borderRadius: "0.375rem",
                    fontSize: "0.875rem"
                  }}
                  placeholder="Enter patient name"
                />
              </div>
              
              <div>
                <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Age</label>
                <input
                  type="number"
                  value={newReport.age}
                  onChange={(e) => setNewReport({...newReport, age: e.target.value})}
                  style={{
                    width: "100%",
                    padding: "0.5rem 0.75rem",
                    border: "1px solid #d1d5db",
                    borderRadius: "0.375rem",
                    fontSize: "0.875rem"
                  }}
                  placeholder="Enter age"
                />
              </div>
              
              <div>
                <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Gender</label>
                <select
                  value={newReport.gender}
                  onChange={(e) => setNewReport({...newReport, gender: e.target.value})}
                  style={{
                    width: "100%",
                    padding: "0.5rem 0.75rem",
                    border: "1px solid #d1d5db",
                    borderRadius: "0.375rem",
                    fontSize: "0.875rem"
                  }}
                >
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="other">Other</option>
                </select>
              </div>
              
              <div>
                <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Village</label>
                <select
                  value={newReport.village}
                  onChange={(e) => setNewReport({...newReport, village: e.target.value})}
                  style={{
                    width: "100%",
                    padding: "0.5rem 0.75rem",
                    border: "1px solid #d1d5db",
                    borderRadius: "0.375rem",
                    fontSize: "0.875rem"
                  }}
                >
                  <option value="">Select Village</option>
                  <option value="Riverside Village">Riverside Village</option>
                  <option value="Mountain View">Mountain View</option>
                  <option value="Green Valley">Green Valley</option>
                  <option value="Sunset Hills">Sunset Hills</option>
                </select>
              </div>
              
              <div>
                <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Symptoms</label>
                <input
                  type="text"
                  value={newReport.symptoms}
                  onChange={(e) => setNewReport({...newReport, symptoms: e.target.value})}
                  style={{
                    width: "100%",
                    padding: "0.5rem 0.75rem",
                    border: "1px solid #d1d5db",
                    borderRadius: "0.375rem",
                    fontSize: "0.875rem"
                  }}
                  placeholder="Fever, Headache, Diarrhea (comma separated)"
                />
              </div>
              
              <div>
                <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Severity</label>
                <select
                  value={newReport.severity}
                  onChange={(e) => setNewReport({...newReport, severity: e.target.value})}
                  style={{
                    width: "100%",
                    padding: "0.5rem 0.75rem",
                    border: "1px solid #d1d5db",
                    borderRadius: "0.375rem",
                    fontSize: "0.875rem"
                  }}
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
            </div>
            
            <div style={{ marginBottom: "1.5rem" }}>
              <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Description</label>
              <textarea
                value={newReport.description}
                onChange={(e) => setNewReport({...newReport, description: e.target.value})}
                rows={4}
                style={{
                  width: "100%",
                  padding: "0.5rem 0.75rem",
                  border: "1px solid #d1d5db",
                  borderRadius: "0.375rem",
                  fontSize: "0.875rem"
                }}
                placeholder="Provide detailed description of the health incident..."
              />
            </div>
            
            <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.75rem" }}>
              <button
                onClick={() => setShowSubmitForm(false)}
                style={{
                  padding: "0.5rem 1rem",
                  color: "#666",
                  border: "1px solid #d1d5db",
                  borderRadius: "0.375rem",
                  backgroundColor: "white",
                  cursor: "pointer"
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleSubmitReport}
                style={{
                  padding: "0.5rem 1rem",
                  backgroundColor: "#2563eb",
                  color: "white",
                  borderRadius: "0.375rem",
                  border: "none",
                  cursor: "pointer"
                }}
              >
                Submit Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HealthReports;