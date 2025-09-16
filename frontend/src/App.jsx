import React from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import ErrorBoundary from "./components/ErrorBoundary";
import Footer from "./components/Footer";
import Navbar from "./components/Navbar";
import ProtectedRoute from "./components/ProtectedRoute";
import AdminDashboard from "./dashboards/AdminDashboard";
import AuthorityDashboard from "./dashboards/AuthorityDashboard";
import CommunityDashboard from "./dashboards/CommunityDashboard";
import WorkerDashboard from "./dashboards/WorkerDashboard";
import Awareness from "./pages/Awareness";
import Dashboard from "./pages/Dashboard";
import Feedback from "./pages/Feedback";
import GISMap from "./pages/GISMap";
import HealthReports from "./pages/HealthReports";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import WaterQuality from "./pages/WaterQuality";

function App() {
  return (
    <div style={{ background: "#F7F9FA", minHeight: "100vh" }}>
      <Navbar />
      <main>
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/dashboard/admin" element={<ProtectedRoute roles={["admin"]}><AdminDashboard /></ProtectedRoute>} />
            <Route path="/dashboard/authority" element={<ProtectedRoute roles={["admin","officer"]}><AuthorityDashboard /></ProtectedRoute>} />
            <Route path="/dashboard/worker" element={<ProtectedRoute roles={["admin","officer","worker"]}><WorkerDashboard /></ProtectedRoute>} />
            <Route path="/dashboard/community" element={<ProtectedRoute roles={["admin","officer","worker","community"]}><CommunityDashboard /></ProtectedRoute>} />
            <Route path="/reports" element={<HealthReports />} />
            <Route path="/community" element={<Awareness />} />
            <Route path="/assistant" element={<Feedback />} />
            <Route path="/water" element={<WaterQuality />} />
            <Route path="/alerts" element={<GISMap />} />
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </ErrorBoundary>
      </main>
      <Footer />
    </div>
  );
}

export default App;