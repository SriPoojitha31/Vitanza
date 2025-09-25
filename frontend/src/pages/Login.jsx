import React, { useContext, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { AuthContext } from "../auth/AuthContext";
import googleLogo from "../assets/Logo.png";
import { Eye, EyeOff, Mail, Lock, User } from "lucide-react";
import toast from "react-hot-toast";

export default function Login() {
  const { signIn, signInWithGoogle, loading } = useContext(AuthContext);
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: "",
    password: ""
  });
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    
    const result = await signIn(formData.email, formData.password);
    
    if (result.success) {
      toast.success("Successfully signed in!");
      navigate("/dashboard");
    } else {
      toast.error(result.error || "Sign in failed");
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
      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: "1rem"
    }}>
      <div style={{
        background: "white",
        padding: "2.5rem",
        borderRadius: "20px",
        boxShadow: "0 20px 40px rgba(0,0,0,0.1)",
        width: "100%",
        maxWidth: "400px"
      }}>
        <div style={{ textAlign: "center", marginBottom: "2rem" }}>
          <h1 style={{ 
            margin: 0, 
            color: "#2D3748", 
            fontSize: "2rem", 
            fontWeight: "700" 
          }}>
            Welcome Back
          </h1>
          <p style={{ 
            color: "#718096", 
            marginTop: "0.5rem",
            fontSize: "1rem"
          }}>
            Sign in to your Vitanza account
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: "1.5rem" }}>
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

          <div style={{ marginBottom: "1.5rem" }}>
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
                placeholder="Enter your password"
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

          <button
            type="submit"
            disabled={isLoading}
            style={{
              ...buttonStyle,
              opacity: isLoading ? 0.7 : 1,
              cursor: isLoading ? "not-allowed" : "pointer"
            }}
          >
            {isLoading ? "Signing In..." : "Sign In"}
          </button>
        </form>

        <div style={{ display: "flex", alignItems: "center", gap: "12px", margin: "16px 0" }}>
          <div style={{ height: 1, background: "#E2E8F0", flex: 1 }} />
          <span style={{ color: "#A0AEC0", fontSize: 12 }}>OR</span>
          <div style={{ height: 1, background: "#E2E8F0", flex: 1 }} />
        </div>

        <button
          type="button"
          onClick={async ()=>{
            setIsLoading(true);
            const result = await signInWithGoogle();
            if (result.success) {
              toast.success("Signed in with Google");
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
          <img src={googleLogo} alt="g" style={{ width: 18, height: 18, borderRadius: 4 }} />
          Continue with Google
        </button>

        <div style={{ 
          textAlign: "center", 
          marginTop: "1.5rem",
          color: "#718096"
        }}>
          Don't have an account?{" "}
          <Link 
            to="/signup" 
            style={{ 
              color: "#667eea", 
              textDecoration: "none",
              fontWeight: "500"
            }}
          >
            Sign up here
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
  background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
  color: "white",
  border: "none",
  borderRadius: "10px",
  padding: "12px",
  fontSize: "1rem",
  fontWeight: "600",
  cursor: "pointer",
  transition: "transform 0.2s, box-shadow 0.2s"
};

