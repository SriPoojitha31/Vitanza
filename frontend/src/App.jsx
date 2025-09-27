import React from "react";
import { Toaster } from "react-hot-toast";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import ErrorBoundary from "./components/ErrorBoundary";
import Footer from "./components/Footer";
import Navbar from "./components/Navbar";
import ProtectedRoute, { GuestRoute } from "./components/ProtectedRoute";
import AdminDashboard from "./dashboards/AdminDashboard"; //dashboards for different user roles
import AuthorityDashboard from "./dashboards/AuthorityDashboard";
import WorkerDashboard from "./dashboards/WorkerDashboard";
import Alerts from "./pages/Alerts"; //alerts page
import Communities from "./pages/Communities";
import Dashboard from "./pages/Dashboard"; //dashboard page
import Feedback from "./pages/Feedback"; //feedback page
import GISMap from "./pages/GISMap"; //gis map page
import HealthReports from "./pages/HealthReports";
import Home from "./pages/Home"; //home page
import Login from "./pages/Login"; //login page
import Signup from "./pages/Signup"; //signup page
import WaterQuality from "./pages/WaterQuality"; //water quality page

function App() {
  const location = useLocation();//location of the page
  const showNavbar = location.pathname !== "/";
  return ( //routes for the pages
      <div style={{ background: "#F7F9FA", minHeight: "100vh" }}>
        {showNavbar && <Navbar />}
        <main>
          <ErrorBoundary>
            <Routes> 
              <Route path="/" element={<Home />} />
              <Route path="/dashboard" element={<ProtectedRoute roles={["admin","officer","worker","community"]}><Dashboard /></ProtectedRoute>} />
              <Route path="/dashboard/admin" element={<ProtectedRoute roles={["admin"]}><AdminDashboard /></ProtectedRoute>} />
              <Route path="/dashboard/authority" element={<ProtectedRoute roles={["admin","officer"]}><AuthorityDashboard /></ProtectedRoute>} />
              <Route path="/dashboard/worker" element={<ProtectedRoute roles={["admin","officer","worker"]}><WorkerDashboard /></ProtectedRoute>} />
              <Route path="/dashboard/community" element={<ProtectedRoute roles={["admin","officer","worker","community"]}><Communities /></ProtectedRoute>} />
              <Route path="/reports" element={<HealthReports />} />
              <Route path="/community" element={<Communities />} />
              <Route path="/alerts" element={<Alerts />} />
              <Route path="/assistant" element={<Feedback />} />
              <Route path="/water" element={<WaterQuality />} />
              <Route path="/gis" element={<GISMap />} />
              <Route path="/login" element={<GuestRoute><Login /></GuestRoute>} />
              <Route path="/signup" element={<GuestRoute><Signup /></GuestRoute>} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </ErrorBoundary>
        </main>
        <Footer />
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
            success: {
              duration: 3000,
              iconTheme: {
                primary: '#4ade80',
                secondary: '#fff',
              },
            },
            error: {
              duration: 5000,
              iconTheme: {
                primary: '#ef4444',
                secondary: '#fff',
              },
            },
          }}
        />
      </div>
  );
}

export default App;