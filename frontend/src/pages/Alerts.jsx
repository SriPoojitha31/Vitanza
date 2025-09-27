import React, { useEffect, useState } from 'react';
import Card from '../components/Card';

const Alerts = () => {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreateAlert, setShowCreateAlert] = useState(false);
  const [newAlert, setNewAlert] = useState({
    title: '',
    type: 'water_quality',
    severity: 'medium',
    community: '',
    description: '',
    peopleAffected: 0
  });

  // Sample data matching the reference design
  const sampleAlerts = [
    {
      id: 'alert-001',
      title: 'Water Contamination Detected',
      description: 'High bacteria levels found in Borehole B at Green Valley',
      type: 'water_quality',
      severity: 'critical',
      status: 'active',
      community: 'Green Valley',
      timestamp: '2024-01-15 14:30',
      peopleAffected: 2100,
      actions: [
        'Water source closed',
        'Alternative supply arranged',
        'Health screening initiated'
      ]
    },
    {
      id: 'alert-002',
      title: 'Disease Outbreak Alert',
      description: 'Unusual increase in diarrheal cases reported in Mountain View',
      type: 'disease_outbreak',
      severity: 'high',
      status: 'investigating',
      community: 'Mountain View',
      timestamp: '2024-01-14 09:15',
      peopleAffected: 890,
      actions: [
        'Contact tracing started',
        'Samples collected',
        'Community notified'
      ]
    },
    {
      id: 'alert-003',
      title: 'Equipment Maintenance Required',
      description: 'Water testing equipment needs calibration',
      type: 'maintenance',
      severity: 'medium',
      status: 'scheduled',
      community: 'All Communities',
      timestamp: '2024-01-13 16:45',
      peopleAffected: 0,
      actions: [
        'Maintenance scheduled',
        'Backup equipment prepared'
      ]
    },
    {
      id: 'alert-004',
      title: 'Health Worker Training',
      description: 'Mandatory training session for new protocols',
      type: 'training',
      severity: 'low',
      status: 'resolved',
      community: 'Riverside Village',
      timestamp: '2024-01-12 10:00',
      peopleAffected: 0,
      actions: [
        'Training completed',
        'Certificates issued'
      ]
    }
  ];

  const recentActivity = [
    {
      action: 'Critical alert created',
      alert: 'Water Contamination Detected',
      user: 'System',
      timestamp: '14:30'
    },
    {
      action: 'Alert status updated',
      alert: 'Disease Outbreak Alert',
      user: 'Dr. Sarah Johnson',
      timestamp: '13:45'
    },
    {
      action: 'Alert resolved',
      alert: 'Equipment Maintenance',
      user: 'Tech Team',
      timestamp: '12:20'
    },
    {
      action: 'Community notification sent',
      alert: 'Health Worker Training',
      user: 'Admin',
      timestamp: '11:15'
    }
  ];

  useEffect(() => {
    // Simulate API call
    setTimeout(() => {
      setAlerts(sampleAlerts);
      setLoading(false);
    }, 1000);
  }, []);

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return { backgroundColor: '#fee2e2', color: '#dc2626', border: '1px solid #fecaca' };
      case 'high': return { backgroundColor: '#fed7aa', color: '#ea580c', border: '1px solid #fed7aa' };
      case 'medium': return { backgroundColor: '#fef3c7', color: '#d97706', border: '1px solid #fef3c7' };
      case 'low': return { backgroundColor: '#dcfce7', color: '#16a34a', border: '1px solid #bbf7d0' };
      default: return { backgroundColor: '#f3f4f6', color: '#374151', border: '1px solid #e5e7eb' };
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return { backgroundColor: '#fee2e2', color: '#dc2626' };
      case 'investigating': return { backgroundColor: '#dbeafe', color: '#2563eb' };
      case 'scheduled': return { backgroundColor: '#fef3c7', color: '#d97706' };
      case 'resolved': return { backgroundColor: '#dcfce7', color: '#16a34a' };
      default: return { backgroundColor: '#f3f4f6', color: '#374151' };
    }
  };

  const handleCreateAlert = () => {
    const alert = {
      id: `alert-${Date.now()}`,
      ...newAlert,
      timestamp: new Date().toISOString(),
      status: 'active',
      actions: []
    };
    setAlerts([alert, ...alerts]);
    setShowCreateAlert(false);
    setNewAlert({
      title: '',
      type: 'water_quality',
      severity: 'medium',
      community: '',
      description: '',
      peopleAffected: 0
    });
  };

  const updateAlertStatus = (alertId, newStatus) => {
    setAlerts(alerts.map(alert => 
      alert.id === alertId ? { ...alert, status: newStatus } : alert
    ));
  };

  const stats = {
    critical: alerts.filter(a => a.severity === 'critical').length,
    active: alerts.filter(a => a.status === 'active').length,
    peopleAffected: alerts.reduce((sum, a) => sum + a.peopleAffected, 0),
    resolvedToday: alerts.filter(a => a.status === 'resolved' && 
      new Date(a.timestamp).toDateString() === new Date().toDateString()).length
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
          <h1 style={{ fontSize: "2rem", fontWeight: "bold", marginBottom: "0.5rem" }}>Alert Management</h1>
          <p style={{ color: "#666" }}>Monitor and manage community health alerts</p>
        </div>
        <button
          onClick={() => setShowCreateAlert(true)}
          style={{
            backgroundColor: "#2563eb",
            color: "white",
            padding: "0.5rem 1rem",
            borderRadius: "0.5rem",
            border: "none",
            cursor: "pointer"
          }}
        >
          Create Alert
        </button>
      </div>

      {/* Stats Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1.5rem", marginBottom: "1.5rem" }}>
        <Card style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center" }}>
            <div style={{ padding: "0.75rem", backgroundColor: "#fee2e2", borderRadius: "50%", marginRight: "1rem" }}>
              <svg style={{ width: "1.5rem", height: "1.5rem", color: "#dc2626" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <div>
              <p style={{ fontSize: "0.875rem", fontWeight: "500", color: "#666", margin: 0 }}>Critical Alerts</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", margin: 0 }}>{stats.critical}</p>
            </div>
          </div>
        </Card>

        <Card style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center" }}>
            <div style={{ padding: "0.75rem", backgroundColor: "#dbeafe", borderRadius: "50%", marginRight: "1rem" }}>
              <svg style={{ width: "1.5rem", height: "1.5rem", color: "#2563eb" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div>
              <p style={{ fontSize: "0.875rem", fontWeight: "500", color: "#666", margin: 0 }}>Active Alerts</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", margin: 0 }}>{stats.active}</p>
            </div>
          </div>
        </Card>

        <Card style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center" }}>
            <div style={{ padding: "0.75rem", backgroundColor: "#dcfce7", borderRadius: "50%", marginRight: "1rem" }}>
              <svg style={{ width: "1.5rem", height: "1.5rem", color: "#16a34a" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
            </div>
            <div>
              <p style={{ fontSize: "0.875rem", fontWeight: "500", color: "#666", margin: 0 }}>People Affected</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", margin: 0 }}>{stats.peopleAffected.toLocaleString()}</p>
            </div>
          </div>
        </Card>

        <Card style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center" }}>
            <div style={{ padding: "0.75rem", backgroundColor: "#e9d5ff", borderRadius: "50%", marginRight: "1rem" }}>
              <svg style={{ width: "1.5rem", height: "1.5rem", color: "#9333ea" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <p style={{ fontSize: "0.875rem", fontWeight: "500", color: "#666", margin: 0 }}>Resolved Today</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", margin: 0 }}>{stats.resolvedToday}</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Alerts List */}
      <div style={{ marginBottom: "1.5rem" }}>
        {alerts.map((alert) => (
          <Card key={alert.id} style={{ padding: "1.5rem", marginBottom: "1rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem" }}>
                  <h3 style={{ fontSize: "1.125rem", fontWeight: "600", margin: 0 }}>{alert.title}</h3>
                  <span style={{ 
                    padding: "0.25rem 0.5rem", 
                    borderRadius: "9999px", 
                    fontSize: "0.75rem", 
                    fontWeight: "500",
                    ...getSeverityColor(alert.severity)
                  }}>
                    {alert.severity}
                  </span>
                  <span style={{ 
                    padding: "0.25rem 0.5rem", 
                    borderRadius: "9999px", 
                    fontSize: "0.75rem", 
                    fontWeight: "500",
                    ...getStatusColor(alert.status)
                  }}>
                    {alert.status}
                  </span>
                </div>
                
                <p style={{ color: "#666", marginBottom: "0.75rem" }}>{alert.description}</p>
                
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem", fontSize: "0.875rem", color: "#666", marginBottom: "1rem" }}>
                  <div>
                    <span style={{ fontWeight: "500" }}>Type:</span> {alert.type.replace('_', ' ')}
                  </div>
                  <div>
                    <span style={{ fontWeight: "500" }}>Community:</span> {alert.community}
                  </div>
                  <div>
                    <span style={{ fontWeight: "500" }}>Timestamp:</span> {alert.timestamp}
                  </div>
                  <div>
                    <span style={{ fontWeight: "500" }}>People Affected:</span> {alert.peopleAffected.toLocaleString()}
                  </div>
                </div>

                {alert.actions.length > 0 && (
                  <div style={{ marginBottom: "1rem" }}>
                    <p style={{ fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.5rem" }}>Actions Taken:</p>
                    <ul style={{ listStyle: "disc", listStylePosition: "inside", fontSize: "0.875rem", color: "#666" }}>
                      {alert.actions.map((action, index) => (
                        <li key={index} style={{ marginBottom: "0.25rem" }}>{action}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              <div style={{ display: "flex", gap: "0.5rem", marginLeft: "1rem" }}>
                <button
                  onClick={() => updateAlertStatus(alert.id, 'investigating')}
                  style={{
                    padding: "0.25rem 0.75rem",
                    fontSize: "0.875rem",
                    backgroundColor: "#dbeafe",
                    color: "#2563eb",
                    borderRadius: "0.25rem",
                    border: "none",
                    cursor: "pointer"
                  }}
                >
                  View Details
                </button>
                <button
                  onClick={() => updateAlertStatus(alert.id, 'resolved')}
                  style={{
                    padding: "0.25rem 0.75rem",
                    fontSize: "0.875rem",
                    backgroundColor: "#dcfce7",
                    color: "#16a34a",
                    borderRadius: "0.25rem",
                    border: "none",
                    cursor: "pointer"
                  }}
                >
                  Update Status
                </button>
                <button style={{
                  padding: "0.25rem 0.75rem",
                  fontSize: "0.875rem",
                  backgroundColor: "#e9d5ff",
                  color: "#9333ea",
                  borderRadius: "0.25rem",
                  border: "none",
                  cursor: "pointer"
                }}>
                  Send Notification
                </button>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Recent Activity */}
      <Card style={{ padding: "1.5rem" }}>
        <h3 style={{ fontSize: "1.125rem", fontWeight: "600", marginBottom: "1rem" }}>Recent Alert Activity</h3>
        <p style={{ fontSize: "0.875rem", color: "#666", marginBottom: "1rem" }}>Timeline of recent alert updates</p>
        
        <div>
          {recentActivity.map((activity, index) => (
            <div key={index} style={{ display: "flex", alignItems: "center", gap: "0.75rem", padding: "0.75rem", backgroundColor: "#f9fafb", borderRadius: "0.5rem", marginBottom: "0.75rem" }}>
              <div style={{ width: "0.5rem", height: "0.5rem", backgroundColor: "#2563eb", borderRadius: "50%" }}></div>
              <div style={{ flex: 1 }}>
                <p style={{ fontSize: "0.875rem", fontWeight: "500", margin: 0 }}>{activity.action}</p>
                <p style={{ fontSize: "0.75rem", color: "#666", margin: 0 }}>{activity.alert} • {activity.user}</p>
              </div>
              <span style={{ fontSize: "0.75rem", color: "#9ca3af" }}>{activity.timestamp}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Create Alert Modal */}
      {showCreateAlert && (
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
            maxWidth: "28rem" 
          }}>
            <h3 style={{ fontSize: "1.125rem", fontWeight: "600", marginBottom: "1rem" }}>Create New Alert</h3>
            
            <div style={{ marginBottom: "1rem" }}>
              <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Title</label>
              <input
                type="text"
                value={newAlert.title}
                onChange={(e) => setNewAlert({...newAlert, title: e.target.value})}
                style={{
                  width: "100%",
                  padding: "0.5rem 0.75rem",
                  border: "1px solid #d1d5db",
                  borderRadius: "0.375rem",
                  fontSize: "0.875rem"
                }}
              />
            </div>
            
            <div style={{ marginBottom: "1rem" }}>
              <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Type</label>
              <select
                value={newAlert.type}
                onChange={(e) => setNewAlert({...newAlert, type: e.target.value})}
                style={{
                  width: "100%",
                  padding: "0.5rem 0.75rem",
                  border: "1px solid #d1d5db",
                  borderRadius: "0.375rem",
                  fontSize: "0.875rem"
                }}
              >
                <option value="water_quality">Water Quality</option>
                <option value="disease_outbreak">Disease Outbreak</option>
                <option value="maintenance">Maintenance</option>
                <option value="training">Training</option>
              </select>
            </div>
            
            <div style={{ marginBottom: "1rem" }}>
              <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Severity</label>
              <select
                value={newAlert.severity}
                onChange={(e) => setNewAlert({...newAlert, severity: e.target.value})}
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
                <option value="critical">Critical</option>
              </select>
            </div>
            
            <div style={{ marginBottom: "1rem" }}>
              <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Community</label>
              <input
                type="text"
                value={newAlert.community}
                onChange={(e) => setNewAlert({...newAlert, community: e.target.value})}
                style={{
                  width: "100%",
                  padding: "0.5rem 0.75rem",
                  border: "1px solid #d1d5db",
                  borderRadius: "0.375rem",
                  fontSize: "0.875rem"
                }}
              />
            </div>
            
            <div style={{ marginBottom: "1rem" }}>
              <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Description</label>
              <textarea
                value={newAlert.description}
                onChange={(e) => setNewAlert({...newAlert, description: e.target.value})}
                rows={3}
                style={{
                  width: "100%",
                  padding: "0.5rem 0.75rem",
                  border: "1px solid #d1d5db",
                  borderRadius: "0.375rem",
                  fontSize: "0.875rem"
                }}
              />
            </div>
            
            <div style={{ marginBottom: "1.5rem" }}>
              <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>People Affected</label>
              <input
                type="number"
                value={newAlert.peopleAffected}
                onChange={(e) => setNewAlert({...newAlert, peopleAffected: parseInt(e.target.value) || 0})}
                style={{
                  width: "100%",
                  padding: "0.5rem 0.75rem",
                  border: "1px solid #d1d5db",
                  borderRadius: "0.375rem",
                  fontSize: "0.875rem"
                }}
              />
            </div>
            
            <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.75rem" }}>
              <button
                onClick={() => setShowCreateAlert(false)}
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
                onClick={handleCreateAlert}
                style={{
                  padding: "0.5rem 1rem",
                  backgroundColor: "#2563eb",
                  color: "white",
                  borderRadius: "0.375rem",
                  border: "none",
                  cursor: "pointer"
                }}
              >
                Create Alert
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Alerts;