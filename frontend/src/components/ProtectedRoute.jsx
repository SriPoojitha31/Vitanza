import React, { useContext } from "react";
import { Navigate } from "react-router-dom";
import { AuthContext } from "../auth/AuthContext";

export default function ProtectedRoute({ roles, children }) {
  const { token, role } = useContext(AuthContext);
  if (!token) return <Navigate to="/login" replace />;
  if (roles && roles.length && !roles.includes(role)) return <Navigate to="/" replace />;
  return children;
}


