import { AlertTriangle, Bell, CheckCircle, Info, X } from "lucide-react";
import React, { useContext, useEffect, useState } from "react";
import { AuthContext } from "../auth/AuthContext";

export default function NotificationCenter() {
  const [isOpen, setIsOpen] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const { user } = useContext(AuthContext);

  useEffect(() => {
    if (user) {
      fetchNotifications();
      // Poll for new notifications every 30 seconds
      const interval = setInterval(fetchNotifications, 30000);
      return () => clearInterval(interval);
    }
  }, [user]);

  const fetchNotifications = async () => {
    try {
      const response = await fetch('/api/alerts/', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      if (response.ok) {
        const data = await response.json();
        setNotifications(Array.isArray(data) ? data : []);
      }
    } catch (error) {
      console.error('Failed to fetch notifications:', error);
    }
  };

  const getNotificationIcon = (severity) => {
    switch (severity) {
      case 'high':
        return <AlertTriangle size={16} color="#EF4444" />;
      case 'medium':
        return <AlertTriangle size={16} color="#F59E0B" />;
      case 'low':
        return <Info size={16} color="#3B82F6" />;
      default:
        return <CheckCircle size={16} color="#10B981" />;
    }
  };

  const getNotificationStyle = (severity) => {
    switch (severity) {
      case 'high':
        return {
          background: '#FEF2F2',
          border: '1px solid #FECACA',
          borderLeft: '4px solid #EF4444'
        };
      case 'medium':
        return {
          background: '#FFFBEB',
          border: '1px solid #FED7AA',
          borderLeft: '4px solid #F59E0B'
        };
      case 'low':
        return {
          background: '#EFF6FF',
          border: '1px solid #BFDBFE',
          borderLeft: '4px solid #3B82F6'
        };
      default:
        return {
          background: '#F0FDF4',
          border: '1px solid #BBF7D0',
          borderLeft: '4px solid #10B981'
        };
    }
  };

  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <div style={{ position: 'relative' }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          background: 'transparent',
          border: '1px solid #E5E7EB',
          borderRadius: '10px',
          padding: '0.5rem',
          cursor: 'pointer',
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        <Bell size={20} color="#4A5568" />
        {unreadCount > 0 && (
          <div
            style={{
              position: 'absolute',
              top: '-2px',
              right: '-2px',
              background: '#EF4444',
              color: 'white',
              borderRadius: '50%',
              width: '18px',
              height: '18px',
              fontSize: '10px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 'bold'
            }}
          >
            {unreadCount}
          </div>
        )}
      </button>

      {isOpen && (
        <div
          style={{
            position: 'absolute',
            top: '100%',
            right: 0,
            background: 'white',
            border: '1px solid #E5E7EB',
            borderRadius: '12px',
            boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
            width: '400px',
            maxHeight: '500px',
            overflowY: 'auto',
            zIndex: 1000
          }}
        >
          <div
            style={{
              padding: '1rem',
              borderBottom: '1px solid #F1F5F9',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <h3 style={{ margin: 0, color: '#1F2937' }}>Notifications</h3>
            <button
              onClick={() => setIsOpen(false)}
              style={{
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                padding: '0.25rem'
              }}
            >
              <X size={16} color="#6B7280" />
            </button>
          </div>

          <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
            {notifications.length === 0 ? (
              <div
                style={{
                  padding: '2rem',
                  textAlign: 'center',
                  color: '#6B7280'
                }}
              >
                No notifications yet
              </div>
            ) : (
              notifications.map((notification, index) => (
                <div
                  key={index}
                  style={{
                    padding: '1rem',
                    borderBottom: '1px solid #F1F5F9',
                    ...getNotificationStyle(notification.severity || 'info')
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '0.75rem'
                    }}
                  >
                    {getNotificationIcon(notification.severity || 'info')}
                    <div style={{ flex: 1 }}>
                      <div
                        style={{
                          fontWeight: '600',
                          color: '#1F2937',
                          marginBottom: '0.25rem'
                        }}
                      >
                        {notification.type || 'Alert'}
                      </div>
                      <div
                        style={{
                          color: '#4B5563',
                          fontSize: '0.875rem',
                          lineHeight: '1.4'
                        }}
                      >
                        {notification.message}
                      </div>
                      <div
                        style={{
                          color: '#9CA3AF',
                          fontSize: '0.75rem',
                          marginTop: '0.5rem'
                        }}
                      >
                        {new Date(notification.timestamp).toLocaleString()}
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
