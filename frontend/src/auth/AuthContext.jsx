import React, { createContext, useEffect, useMemo, useState } from "react";

function decodeRole(token) {
  try {
    const payload = JSON.parse(atob(token.split(".")[1] || ""));
    return payload?.role || null;
  } catch (_) {
    return null;
  }
}

export const AuthContext = createContext({ token: null, role: null, setToken: () => {}, logout: () => {} });

export function AuthProvider({ children }) {
  const [token, setTokenState] = useState(localStorage.getItem("token") || null);
  const [role, setRole] = useState(token ? decodeRole(token) : null);

  function setToken(t) {
    if (t) {
      localStorage.setItem("token", t);
    } else {
      localStorage.removeItem("token");
    }
    setTokenState(t);
    setRole(t ? decodeRole(t) : null);
  }

  function logout() {
    setToken(null);
  }

  useEffect(() => {
    if (!token) return;
    const r = decodeRole(token);
    if (r !== role) setRole(r);
  }, [token]);

  const value = useMemo(() => ({ token, role, setToken, logout }), [token, role]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}


