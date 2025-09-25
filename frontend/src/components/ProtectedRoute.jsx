import React, { useContext } from "react";
import { Navigate } from "react-router-dom";
import { AuthContext } from "../auth/AuthContext";

export default function ProtectedRoute({ roles, children }) {
  const { user, loading } = useContext(AuthContext);
  console.log('ProtectedRoute - User:', user, 'Loading:', loading, 'Roles:', roles);
  
  if (loading) {
    return (
      <div style={{ 
        display: "flex", 
        justifyContent: "center", 
        alignItems: "center", 
        height: "100vh",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
      }}>
        <div style={{ color: "white", fontSize: "1.2rem" }}>Loading...</div>
      </div>
    );
  }
  
  if (!user) return <Navigate to="/login" replace />;
  if (roles && roles.length && !roles.includes(user.role)) return <Navigate to="/" replace />;
  return children;
}

export function GuestRoute({ children }) {
  const { user, loading } = useContext(AuthContext);
  if (loading) {
    return (
      <div style={{ 
        display: "flex", justifyContent: "center", alignItems: "center", height: "100vh",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)" }}>
        <div style={{ color: "white", fontSize: "1.2rem" }}>Loading...</div>
      </div>
    );
  }
  if (user) return <Navigate to="/dashboard" replace />;
  return children;
}


