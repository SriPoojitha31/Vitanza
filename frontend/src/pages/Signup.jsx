import { Eye, EyeOff, Lock, Mail, Shield, User } from "lucide-react";
import React, { useContext, useState } from "react";
import toast from "react-hot-toast";
import { Link, useNavigate } from "react-router-dom";
import { AuthContext } from "../auth/AuthContext";

export default function Signup() {
  const { signUp, loading } = useContext(AuthContext);
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    confirmPassword: "",
    displayName: "",
    role: "community"
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (formData.password !== formData.confirmPassword) {
      toast.error("Passwords do not match");
      return;
    }

    if (formData.password.length < 6) {
      toast.error("Password must be at least 6 characters");
      return;
    }

    setIsLoading(true);
    
    const result = await signUp(formData.email, formData.password, {
      displayName: formData.displayName,
      username: formData.displayName,
      role: formData.role
    });
    
    if (result.success) {
      toast.success("Account created successfully!");
      navigate("/dashboard");
    } else {
      toast.error(result.error || "Sign up failed");
    }
    
    setIsLoading(false);
  };

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

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #2C3E50 0%, #27AE60 100%)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: "1rem"
    }}>
      <div style={{
        background: "white",
        padding: "2rem",
        borderRadius: "16px",
        boxShadow: "0 20px 40px rgba(0,0,0,0.1)",
        width: "100%",
        maxWidth: "520px",
      }}>
        <div style={{ textAlign: "center", marginBottom: "2rem" }}>
          <h1 style={{ 
            margin: 0, 
            color: "#2D3748", 
            fontSize: "2rem", 
            fontWeight: "700" 
          }}>
            Create Account
          </h1>
          <p style={{ 
            color: "#718096", 
            marginTop: "0.5rem",
            fontSize: "1rem"
          }}>
            Join Vitanza and help build healthier communities
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: "1rem" }}>
            <label style={{ 
              display: "block", 
              marginBottom: "0.5rem", 
              color: "#4A5568",
              fontWeight: "500"
            }}>
              Full Name
            </label>
            <div style={{ position: "relative" }}>
              <User style={{ 
                position: "absolute", 
                left: "12px", 
                top: "50%", 
                transform: "translateY(-50%)",
                color: "#A0AEC0",
                width: "20px",
                height: "20px"
              }} />
              <input
                type="text"
                name="displayName"
                value={formData.displayName}
                onChange={handleChange}
                placeholder="Enter your full name"
                required
                style={{
                  ...inputStyle,
                  paddingLeft: "45px"
                }}
              />
            </div>
          </div>

          <div style={{ marginBottom: "1rem" }}>
            <label style={{ 
              display: "block", 
              marginBottom: "0.5rem", 
              color: "#4A5568",
              fontWeight: "500"
            }}>
              Email Address
            </label>
            <div style={{ position: "relative" }}>
              <Mail style={{ 
                position: "absolute", 
                left: "12px", 
                top: "50%", 
                transform: "translateY(-50%)",
                color: "#A0AEC0",
                width: "20px",
                height: "20px"
              }} />
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                placeholder="Enter your email"
                required
                style={{
                  ...inputStyle,
                  paddingLeft: "45px"
                }}
              />
            </div>
          </div>

          <div style={{ marginBottom: "1rem" }}>
            <label style={{ 
              display: "block", 
              marginBottom: "0.5rem", 
              color: "#4A5568",
              fontWeight: "500"
            }}>
              Role
            </label>
            <div style={{ position: "relative" }}>
              <Shield style={{ 
                position: "absolute", 
                left: "12px", 
                top: "50%", 
                transform: "translateY(-50%)",
                color: "#A0AEC0",
                width: "20px",
                height: "20px"
              }} />
              <select
                name="role"
                value={formData.role}
                onChange={handleChange}
                required
                style={{
                  ...inputStyle,
                  paddingLeft: "45px",
                  appearance: "none",
                  backgroundImage: `url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='m6 8 4 4 4-4'/%3e%3c/svg%3e")`,
                  backgroundPosition: "right 12px center",
                  backgroundRepeat: "no-repeat",
                  backgroundSize: "16px"
                }}
              >
                <option value="community">Community User</option>
                <option value="worker">Health Worker</option>
                <option value="officer">Health Authority</option>
                <option value="ngo">NGO</option>
                <option value="government">Government</option>
                <option value="asha">ASHA Worker</option>
                <option value="admin">Admin</option>
              </select>
            </div>
          </div>

          <div style={{ marginBottom: "1rem" }}>
            <label style={{ 
              display: "block", 
              marginBottom: "0.5rem", 
              color: "#4A5568",
              fontWeight: "500"
            }}>
              Password
            </label>
            <div style={{ position: "relative" }}>
              <Lock style={{ 
                position: "absolute", 
                left: "12px", 
                top: "50%", 
                transform: "translateY(-50%)",
                color: "#A0AEC0",
                width: "20px",
                height: "20px"
              }} />
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Create a password"
                required
                style={{
                  ...inputStyle,
                  paddingLeft: "45px",
                  paddingRight: "45px"
                }}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                style={{
                  position: "absolute",
                  right: "12px",
                  top: "50%",
                  transform: "translateY(-50%)",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  color: "#A0AEC0"
                }}
              >
                {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>
          </div>

          <div style={{ marginBottom: "1.5rem" }}>
            <label style={{ 
              display: "block", 
              marginBottom: "0.5rem", 
              color: "#4A5568",
              fontWeight: "500"
            }}>
              Confirm Password
            </label>
            <div style={{ position: "relative" }}>
              <Lock style={{ 
                position: "absolute", 
                left: "12px", 
                top: "50%", 
                transform: "translateY(-50%)",
                color: "#A0AEC0",
                width: "20px",
                height: "20px"
              }} />
              <input
                type={showConfirmPassword ? "text" : "password"}
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                placeholder="Confirm your password"
                required
                style={{
                  ...inputStyle,
                  paddingLeft: "45px",
                  paddingRight: "45px"
                }}
              />
              <button
                type="button"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                style={{
                  position: "absolute",
                  right: "12px",
                  top: "50%",
                  transform: "translateY(-50%)",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  color: "#A0AEC0"
                }}
              >
                {showConfirmPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            style={{
              ...buttonStyle,
              opacity: isLoading ? 0.7 : 1,
              cursor: isLoading ? "not-allowed" : "pointer"
            }}
          >
            {isLoading ? "Creating Account..." : "Create Account"}
          </button>
        </form>

        <div style={{ display: "flex", alignItems: "center", gap: "12px", margin: "16px 0" }}>
          <div style={{ height: 1, background: "#E2E8F0", flex: 1 }} />
          <span style={{ color: "#A0AEC0", fontSize: 12 }}>OR</span>
          <div style={{ height: 1, background: "#E2E8F0", flex: 1 }} />
        </div>

        <GoogleSignupCTA />

        <div style={{ 
          textAlign: "center", 
          marginTop: "1.5rem",
          color: "#718096"
        }}>
          Already have an account?{" "}
          <Link 
            to="/login" 
            style={{ 
              color: "#667eea", 
              textDecoration: "none",
              fontWeight: "500"
            }}
          >
            Sign in here
          </Link>
        </div>
      </div>
    </div>
  );
}

const inputStyle = {
  width: "100%",
  padding: "12px",
  borderRadius: "10px",
  border: "2px solid #E2E8F0",
  fontSize: "1rem",
  transition: "border-color 0.2s",
  outline: "none"
};

const buttonStyle = {
  width: "100%",
  background: "linear-gradient(135deg, #2C3E50 0%, #27AE60 100%)",
  color: "white",
  border: "none",
  borderRadius: "10px",
  padding: "12px",
  fontSize: "1rem",
  fontWeight: "600",
  cursor: "pointer",
  transition: "transform 0.2s, box-shadow 0.2s"
};

function GoogleSignupCTA() {
  const { signInWithGoogle } = useContext(AuthContext);
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  return (
    <button
      type="button"
      onClick={async ()=>{
        setIsLoading(true);
        const result = await signInWithGoogle();
        if (result.success) {
          toast.success("Signed up with Google");
          navigate("/dashboard");
        } else {
          toast.error(result.error || "Google sign-in failed");
        }
        setIsLoading(false);
      }}
      style={{
        ...buttonStyle,
        background: "white",
        color: "#111827",
        border: "1px solid #E5E7EB",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: "8px"
      }}
    >
      <span style={{ width: 18, height: 18, display: "inline-block", background: "#EA4335", borderRadius: 4 }} />
      {isLoading ? "Connecting..." : "Continue with Google"}
    </button>
  );
}

