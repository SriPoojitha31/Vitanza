import { LogOut, User } from "lucide-react";
import React, { useContext, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import logo from "../assets/Logo.png";
import { AuthContext } from "../auth/AuthContext";
import { useI18n } from "../i18n/I18nProvider";
import LanguageSwitcher from "./LanguageSwitcher";
import NotificationCenter from "./NotificationCenter";

export default function Navbar() {
  const { user, logout } = useContext(AuthContext);
  const { t } = useI18n();
  const navigate = useNavigate();
  const [showUserMenu, setShowUserMenu] = useState(false);

  const handleLogout = async () => {
    const result = await logout();
    if (result.success) {
      navigate("/");
    }
  };

  return (
    <nav style={{
      background: "white",
      padding: "0.75rem 1rem",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      borderBottom: "1px solid #EEF2F7",
      position: "sticky",
      top: 0,
      zIndex: 10,
      boxShadow: "0 1px 3px rgba(0,0,0,0.1)"
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", perspective: 800 }}>
        <div
          style={{
            height: 36,
            width: 36,
            display: "grid",
            placeItems: "center",
            transformStyle: "preserve-3d",
            animation: "vitanzaLogoSpin 6s linear infinite",
            filter: "drop-shadow(0 6px 12px rgba(45,156,219,0.35))",
          }}
        >
          <img
            src={logo}
            alt="Vitanza Logo"
            style={{
              height: "32px",
              width: "32px",
              transform: "rotateX(18deg) rotateY(24deg)",
              backfaceVisibility: "hidden",
            }}
          />
        </div>
        <span
          style={{
            fontWeight: 900,
            fontSize: "1.35rem",
            letterSpacing: 0.5,
            background: "conic-gradient(from 180deg at 50% 50%, #06b6d4, #3b82f6, #8b5cf6, #ec4899, #f59e0b, #06b6d4)",
            WebkitBackgroundClip: "text",
            backgroundClip: "text",
            color: "transparent",
            textShadow: "0 6px 18px rgba(59,130,246,0.25)",
            filter: "saturate(1.3)",
          }}
        >
          Vitanza
        </span>
      </div>
      
      <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
        <Link to="/" style={linkStyle}>{t('home')}</Link>
        {user && (
          <>
            <Link to="/dashboard" style={linkStyle}>{t('dashboard')}</Link>
            <Link to="/reports" style={linkStyle}>{t('health_reports')}</Link>
            <Link to="/community" style={linkStyle}>Community</Link>
            <Link to="/assistant" style={linkStyle}>AI Assistant</Link>
            <Link to="/water" style={linkStyle}>{t('water_quality')}</Link>
            <Link to="/alerts" style={linkStyle}>Alerts</Link>
          </>
        )}
      </div>
      
      <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
        <button aria-label="Toggle theme" style={iconBtn} onClick={() => {
          const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
          document.documentElement.setAttribute('data-theme', next);
          localStorage.setItem('theme', next);
        }}>🌓</button>
        {user && <NotificationCenter />}
        <LanguageSwitcher />
        
        {user ? (
          <div style={{ position: "relative" }}>
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
                background: "transparent",
                border: "1px solid #E5E7EB",
                borderRadius: "10px",
                padding: "0.5rem",
                cursor: "pointer",
                transition: "all 0.2s"
              }}
            >
              <User size={20} color="#4A5568" />
              <span style={{ color: "#4A5568", fontWeight: "500" }}>
                {user.displayName || user.email}
              </span>
            </button>
            
            {showUserMenu && (
              <div style={{
                position: "absolute",
                top: "100%",
                right: 0,
                background: "white",
                border: "1px solid #E5E7EB",
                borderRadius: "10px",
                boxShadow: "0 10px 25px rgba(0,0,0,0.1)",
                padding: "0.5rem",
                minWidth: "200px",
                zIndex: 1000
              }}>
                <div style={{ padding: "0.5rem", borderBottom: "1px solid #F1F5F9" }}>
                  <div style={{ fontWeight: "600", color: "#1F2937" }}>
                    {user.displayName || "User"}
                  </div>
                  <div style={{ fontSize: "0.875rem", color: "#6B7280" }}>
                    {user.email}
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "#9CA3AF", textTransform: "capitalize" }}>
                    {user.role}
                  </div>
                </div>
                <button
                  onClick={handleLogout}
                  style={{
                    width: "100%",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.5rem",
                    background: "transparent",
                    border: "none",
                    padding: "0.5rem",
                    cursor: "pointer",
                    color: "#EF4444",
                    fontWeight: "500",
                    borderRadius: "5px",
                    transition: "background 0.2s"
                  }}
                  onMouseEnter={(e) => e.target.style.background = "#FEF2F2"}
                  onMouseLeave={(e) => e.target.style.background = "transparent"}
                >
                  <LogOut size={16} />
                  {t('logout')}
                </button>
              </div>
            )}
          </div>
        ) : (
          <>
            <Link to="/login" style={{ ...ghostBtn, textDecoration: "none", color: "#111827" }}>{t('login')}</Link>
            <Link to="/signup" style={{ ...solidBtn, textDecoration: "none" }}>{t('signup')}</Link>
          </>
        )}
      </div>
    </nav>
  );
}

const linkStyle = {
  color: "#4A5568",
  textDecoration: "none",
  fontWeight: 600,
};

const iconBtn = {
  background: "transparent",
  border: "1px solid #E5E7EB",
  borderRadius: 10,
  padding: "0.35rem 0.5rem",
  cursor: "pointer",
};

const ghostBtn = {
  background: "transparent",
  border: "1px solid #CBD5E1",
  borderRadius: 12,
  padding: "0.5rem 0.9rem",
  fontWeight: 700,
  cursor: "pointer",
};

const solidBtn = {
  background: "#2D9CDB",
  color: "white",
  border: 0,
  borderRadius: 12,
  padding: "0.5rem 0.9rem",
  fontWeight: 700,
  cursor: "pointer",
  boxShadow: "0 8px 18px rgba(45,156,219,0.25)",
};