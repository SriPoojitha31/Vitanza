import React, { useEffect, useState } from 'react';
import Card from '../components/Card';

const Communities = () => {
  const [communities, setCommunities] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddCommunity, setShowAddCommunity] = useState(false);
  const [newCommunity, setNewCommunity] = useState({
    name: '',
    population: 0,
    coordinator: '',
    healthWorkers: 0,
    volunteers: 0,
    status: 'active'
  });

  // Sample data matching the reference design
  const sampleCommunities = [
    {
      id: 'comm-001',
      name: 'Riverside Village',
      population: 1250,
      status: 'active',
      healthScore: 92,
      coordinator: {
        name: 'Dr. Sarah Johnson',
        initials: 'DSJ',
        title: 'Community Health Coordinator'
      },
      healthWorkers: 3,
      volunteers: 8,
      lastUpdate: '2 hours ago',
      activities: [
        'Completed water quality testing',
        'Health screening program',
        'Community workshop conducted'
      ]
    },
    {
      id: 'comm-002',
      name: 'Mountain View',
      population: 890,
      status: 'active',
      healthScore: 78,
      coordinator: {
        name: 'Dr. Michael Chen',
        initials: 'DMC',
        title: 'Health Program Manager'
      },
      healthWorkers: 2,
      volunteers: 5,
      lastUpdate: '4 hours ago',
      activities: [
        'Submitted 3 new health reports',
        'Vaccination drive completed',
        'Health education session'
      ]
    },
    {
      id: 'comm-003',
      name: 'Green Valley',
      population: 2100,
      status: 'attention',
      healthScore: 65,
      coordinator: {
        name: 'Dr. Amara Okafor',
        initials: 'DAO',
        title: 'Senior Health Coordinator'
      },
      healthWorkers: 5,
      volunteers: 12,
      lastUpdate: '1 day ago',
      activities: [
        'Organized community health workshop',
        'Water quality issues identified',
        'Emergency response training'
      ]
    },
    {
      id: 'comm-004',
      name: 'Sunset Hills',
      population: 1680,
      status: 'active',
      healthScore: 88,
      coordinator: {
        name: 'Dr. Carlos Rodriguez',
        initials: 'DCR',
        title: 'Community Health Lead'
      },
      healthWorkers: 4,
      volunteers: 9,
      lastUpdate: '3 hours ago',
      activities: [
        'Updated vaccination records',
        'Health monitoring program',
        'Community outreach completed'
      ]
    }
  ];

  const recentActivities = [
    {
      activity: 'Completed water quality testing',
      community: 'Riverside Village',
      user: 'Dr. Sarah Johnson',
      time: '2 hours ago'
    },
    {
      activity: 'Submitted 3 new health reports',
      community: 'Mountain View',
      user: 'Health Volunteer Maria',
      time: '4 hours ago'
    },
    {
      activity: 'Organized community health workshop',
      community: 'Green Valley',
      user: 'Dr. Amara Okafor',
      time: '1 day ago'
    },
    {
      activity: 'Updated vaccination records',
      community: 'Sunset Hills',
      user: 'Nurse Carlos',
      time: '1 day ago'
    }
  ];

  useEffect(() => {
    // Simulate API call
    setTimeout(() => {
      setCommunities(sampleCommunities);
      setLoading(false);
    }, 1000);
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return { backgroundColor: '#dcfce7', color: '#16a34a' };
      case 'attention': return { backgroundColor: '#fef3c7', color: '#d97706' };
      case 'critical': return { backgroundColor: '#fee2e2', color: '#dc2626' };
      default: return { backgroundColor: '#f3f4f6', color: '#374151' };
    }
  };

  const getHealthScoreColor = (score) => {
    if (score >= 90) return '#16a34a';
    if (score >= 70) return '#d97706';
    return '#dc2626';
  };

  const handleAddCommunity = () => {
    const community = {
      id: `comm-${Date.now()}`,
      ...newCommunity,
      coordinator: {
        name: 'New Coordinator',
        initials: 'NC',
        title: 'Community Health Coordinator'
      },
      healthScore: 75,
      lastUpdate: 'Just now',
      activities: []
    };
    setCommunities([...communities, community]);
    setShowAddCommunity(false);
    setNewCommunity({
      name: '',
      population: 0,
      coordinator: '',
      healthWorkers: 0,
      volunteers: 0,
      status: 'active'
    });
  };

  const stats = {
    totalCommunities: communities.length,
    totalHealthWorkers: communities.reduce((sum, c) => sum + c.healthWorkers, 0),
    totalVolunteers: communities.reduce((sum, c) => sum + c.volunteers, 0),
    totalPopulation: communities.reduce((sum, c) => sum + c.population, 0)
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
          <h1 style={{ fontSize: "2rem", fontWeight: "bold", marginBottom: "0.5rem" }}>Community Management</h1>
          <p style={{ color: "#666" }}>Manage communities and health workers</p>
        </div>
        <button
          onClick={() => setShowAddCommunity(true)}
          style={{
            backgroundColor: "#2563eb",
            color: "white",
            padding: "0.5rem 1rem",
            borderRadius: "0.5rem",
            border: "none",
            cursor: "pointer"
          }}
        >
          Add Community
        </button>
      </div>

      {/* Stats Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1.5rem", marginBottom: "1.5rem" }}>
        <Card style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center" }}>
            <div style={{ padding: "0.75rem", backgroundColor: "#dbeafe", borderRadius: "50%", marginRight: "1rem" }}>
              <svg style={{ width: "1.5rem", height: "1.5rem", color: "#2563eb" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
              </svg>
            </div>
            <div>
              <p style={{ fontSize: "0.875rem", fontWeight: "500", color: "#666", margin: 0 }}>Communities</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", margin: 0 }}>{stats.totalCommunities}</p>
            </div>
          </div>
        </Card>

        <Card style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center" }}>
            <div style={{ padding: "0.75rem", backgroundColor: "#dcfce7", borderRadius: "50%", marginRight: "1rem" }}>
              <svg style={{ width: "1.5rem", height: "1.5rem", color: "#16a34a" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
            </div>
            <div>
              <p style={{ fontSize: "0.875rem", fontWeight: "500", color: "#666", margin: 0 }}>Health Workers</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", margin: 0 }}>{stats.totalHealthWorkers}</p>
            </div>
          </div>
        </Card>

        <Card style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center" }}>
            <div style={{ padding: "0.75rem", backgroundColor: "#e9d5ff", borderRadius: "50%", marginRight: "1rem" }}>
              <svg style={{ width: "1.5rem", height: "1.5rem", color: "#9333ea" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
            </div>
            <div>
              <p style={{ fontSize: "0.875rem", fontWeight: "500", color: "#666", margin: 0 }}>Volunteers</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", margin: 0 }}>{stats.totalVolunteers}</p>
            </div>
          </div>
        </Card>

        <Card style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center" }}>
            <div style={{ padding: "0.75rem", backgroundColor: "#fed7aa", borderRadius: "50%", marginRight: "1rem" }}>
              <svg style={{ width: "1.5rem", height: "1.5rem", color: "#ea580c" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
              </svg>
            </div>
            <div>
              <p style={{ fontSize: "0.875rem", fontWeight: "500", color: "#666", margin: 0 }}>Total Population</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", margin: 0 }}>{stats.totalPopulation.toLocaleString()}</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Communities Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: "1.5rem", marginBottom: "1.5rem" }}>
        {communities.map((community) => (
          <Card key={community.id} style={{ padding: "1.5rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
              <div>
                <h3 style={{ fontSize: "1.25rem", fontWeight: "600", margin: 0, marginBottom: "0.25rem" }}>{community.name}</h3>
                <p style={{ color: "#666", margin: 0 }}>{community.population.toLocaleString()} residents</p>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span style={{ 
                  padding: "0.25rem 0.5rem", 
                  borderRadius: "9999px", 
                  fontSize: "0.75rem", 
                  fontWeight: "500",
                  ...getStatusColor(community.status)
                }}>
                  {community.status}
                </span>
                <span style={{ 
                  fontSize: "0.875rem", 
                  fontWeight: "500", 
                  color: getHealthScoreColor(community.healthScore) 
                }}>
                  {community.healthScore}%
                </span>
              </div>
            </div>

            {/* Coordinator Info */}
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
              <div style={{ 
                width: "2.5rem", 
                height: "2.5rem", 
                backgroundColor: "#dbeafe", 
                borderRadius: "50%", 
                display: "flex", 
                alignItems: "center", 
                justifyContent: "center" 
              }}>
                <span style={{ fontSize: "0.875rem", fontWeight: "500", color: "#2563eb" }}>{community.coordinator.initials}</span>
              </div>
              <div>
                <p style={{ fontWeight: "500", margin: 0 }}>{community.coordinator.name}</p>
                <p style={{ fontSize: "0.875rem", color: "#666", margin: 0 }}>{community.coordinator.title}</p>
              </div>
            </div>

            {/* Stats */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1rem" }}>
              <div style={{ textAlign: "center", padding: "0.75rem", backgroundColor: "#f9fafb", borderRadius: "0.5rem" }}>
                <p style={{ fontSize: "1.5rem", fontWeight: "bold", margin: 0 }}>{community.healthWorkers}</p>
                <p style={{ fontSize: "0.875rem", color: "#666", margin: 0 }}>Health Workers</p>
              </div>
              <div style={{ textAlign: "center", padding: "0.75rem", backgroundColor: "#f9fafb", borderRadius: "0.5rem" }}>
                <p style={{ fontSize: "1.5rem", fontWeight: "bold", margin: 0 }}>{community.volunteers}</p>
                <p style={{ fontSize: "0.875rem", color: "#666", margin: 0 }}>Volunteers</p>
              </div>
            </div>

            {/* Last Update */}
            <div style={{ marginBottom: "1rem" }}>
              <p style={{ fontSize: "0.875rem", color: "#666", margin: 0 }}>Last update: {community.lastUpdate}</p>
            </div>

            {/* Recent Activities */}
            {community.activities.length > 0 && (
              <div style={{ marginBottom: "1rem" }}>
                <p style={{ fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.5rem" }}>Recent Activities:</p>
                <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
                  {community.activities.slice(0, 2).map((activity, index) => (
                    <li key={index} style={{ fontSize: "0.875rem", color: "#666", marginBottom: "0.25rem" }}>• {activity}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Actions */}
            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <button style={{
                padding: "0.5rem 1rem",
                backgroundColor: "#2563eb",
                color: "white",
                borderRadius: "0.5rem",
                border: "none",
                cursor: "pointer"
              }}>
                View Details
              </button>
            </div>
          </Card>
        ))}
      </div>

      {/* Recent Community Activities */}
      <Card style={{ padding: "1.5rem" }}>
        <h3 style={{ fontSize: "1.125rem", fontWeight: "600", marginBottom: "1rem" }}>Recent Community Activities</h3>
        <p style={{ fontSize: "0.875rem", color: "#666", marginBottom: "1rem" }}>Latest updates from community health workers</p>
        
        <div>
          {recentActivities.map((activity, index) => (
            <div key={index} style={{ display: "flex", alignItems: "center", gap: "0.75rem", padding: "0.75rem", backgroundColor: "#f9fafb", borderRadius: "0.5rem", marginBottom: "0.75rem" }}>
              <div style={{ width: "0.5rem", height: "0.5rem", backgroundColor: "#2563eb", borderRadius: "50%" }}></div>
              <div style={{ flex: 1 }}>
                <p style={{ fontSize: "0.875rem", fontWeight: "500", margin: 0 }}>{activity.activity}</p>
                <p style={{ fontSize: "0.75rem", color: "#666", margin: 0 }}>{activity.community} • {activity.user}</p>
              </div>
              <span style={{ fontSize: "0.75rem", color: "#9ca3af" }}>{activity.time}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Add Community Modal */}
      {showAddCommunity && (
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
            <h3 style={{ fontSize: "1.125rem", fontWeight: "600", marginBottom: "1rem" }}>Add New Community</h3>
            
            <div style={{ marginBottom: "1rem" }}>
              <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Community Name</label>
              <input
                type="text"
                value={newCommunity.name}
                onChange={(e) => setNewCommunity({...newCommunity, name: e.target.value})}
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
              <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Population</label>
              <input
                type="number"
                value={newCommunity.population}
                onChange={(e) => setNewCommunity({...newCommunity, population: parseInt(e.target.value) || 0})}
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
              <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Coordinator Name</label>
              <input
                type="text"
                value={newCommunity.coordinator}
                onChange={(e) => setNewCommunity({...newCommunity, coordinator: e.target.value})}
                style={{
                  width: "100%",
                  padding: "0.5rem 0.75rem",
                  border: "1px solid #d1d5db",
                  borderRadius: "0.375rem",
                  fontSize: "0.875rem"
                }}
              />
            </div>
            
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1rem" }}>
              <div>
                <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Health Workers</label>
                <input
                  type="number"
                  value={newCommunity.healthWorkers}
                  onChange={(e) => setNewCommunity({...newCommunity, healthWorkers: parseInt(e.target.value) || 0})}
                  style={{
                    width: "100%",
                    padding: "0.5rem 0.75rem",
                    border: "1px solid #d1d5db",
                    borderRadius: "0.375rem",
                    fontSize: "0.875rem"
                  }}
                />
              </div>
              <div>
                <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Volunteers</label>
                <input
                  type="number"
                  value={newCommunity.volunteers}
                  onChange={(e) => setNewCommunity({...newCommunity, volunteers: parseInt(e.target.value) || 0})}
                  style={{
                    width: "100%",
                    padding: "0.5rem 0.75rem",
                    border: "1px solid #d1d5db",
                    borderRadius: "0.375rem",
                    fontSize: "0.875rem"
                  }}
                />
              </div>
            </div>
            
            <div style={{ marginBottom: "1.5rem" }}>
              <label style={{ display: "block", fontSize: "0.875rem", fontWeight: "500", color: "#374151", marginBottom: "0.25rem" }}>Status</label>
              <select
                value={newCommunity.status}
                onChange={(e) => setNewCommunity({...newCommunity, status: e.target.value})}
                style={{
                  width: "100%",
                  padding: "0.5rem 0.75rem",
                  border: "1px solid #d1d5db",
                  borderRadius: "0.375rem",
                  fontSize: "0.875rem"
                }}
              >
                <option value="active">Active</option>
                <option value="attention">Attention</option>
                <option value="critical">Critical</option>
              </select>
            </div>
            
            <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.75rem" }}>
              <button
                onClick={() => setShowAddCommunity(false)}
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
                onClick={handleAddCommunity}
                style={{
                  padding: "0.5rem 1rem",
                  backgroundColor: "#2563eb",
                  color: "white",
                  borderRadius: "0.375rem",
                  border: "none",
                  cursor: "pointer"
                }}
              >
                Add Community
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Communities;