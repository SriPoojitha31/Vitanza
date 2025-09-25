import React, { createContext, useEffect, useMemo, useState } from "react";

const API_BASE = import.meta?.env?.VITE_API_BASE_URL || import.meta?.env?.VITE_API_BASE || "http://127.0.0.1:8000";

export const AuthContext = createContext({
  user: null,
  loading: true,
  signIn: () => {},
  signUp: () => {},
  logout: () => {},
  updateUserProfile: () => {},
  signInWithGoogle: async () => ({ success: false, error: 'Disabled' })
});

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Attempt to restore existing token from localStorage by pinging a lightweight endpoint if needed
    const token = localStorage.getItem('token');
    const profile = localStorage.getItem('profile');
    if (token && profile) {
      try {
        setUser(JSON.parse(profile));
      } catch {}
    }
    setLoading(false);
  }, []);

  const signIn = async (email, password) => {
    try {
      const res = await fetch(`${API_BASE}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      localStorage.setItem('token', data.access_token);
      
      // Fetch full user profile from /me endpoint
      try {
        const meRes = await fetch(`${API_BASE}/api/auth/me`, {
          headers: { 'Authorization': `Bearer ${data.access_token}` }
        });
        if (meRes.ok) {
          const userProfile = await meRes.json();
          console.log('User profile from /me:', userProfile);
          const profile = {
            email: userProfile.email,
            displayName: userProfile.displayName || userProfile.email,
            role: userProfile.role || 'community',
            id: userProfile.id
          };
          localStorage.setItem('profile', JSON.stringify(profile));
          setUser(profile);
          return { success: true, user: profile };
        } else {
          console.warn('Failed to fetch user profile, status:', meRes.status);
        }
      } catch (meError) {
        console.warn('Failed to fetch user profile:', meError);
      }
      
      // Fallback to basic profile if /me fails
      const profile = { email, displayName: email, role: 'community' };
      localStorage.setItem('profile', JSON.stringify(profile));
      setUser(profile);
      return { success: true, user: profile };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  const signUp = async (email, password, userData) => {
    try {
      const res = await fetch(`${API_BASE}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email,
          password,
          displayName: userData.displayName || userData.username,
          role: userData.role || 'community'
        })
      });
      if (!res.ok) throw new Error(await res.text());
      // Auto-login after signup
      const login = await fetch(`${API_BASE}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      if (!login.ok) throw new Error(await login.text());
      const data = await login.json();
      localStorage.setItem('token', data.access_token);
      const profile = { email, displayName: userData.displayName || userData.username, role: userData.role || 'community' };
      localStorage.setItem('profile', JSON.stringify(profile));
      setUser(profile);
      return { success: true, user: profile };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  const logout = async () => {
    localStorage.removeItem('token');
    localStorage.removeItem('profile');
    setUser(null);
    return { success: true };
  };

  const updateUserProfile = async (userData) => {
    if (!user) return { success: false, error: 'No user logged in' };
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`${API_BASE}/api/users/me`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(userData)
      });
      if (!res.ok) throw new Error(await res.text());
      const merged = { ...user, ...userData };
      setUser(merged);
      localStorage.setItem('profile', JSON.stringify(merged));
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  const signInWithGoogle = async () => ({ success: false, error: 'Disabled' });

  const value = useMemo(() => ({ 
    user, 
    loading, 
    signIn, 
    signUp, 
    logout, 
    updateUserProfile,
    signInWithGoogle
  }), [user, loading]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}


