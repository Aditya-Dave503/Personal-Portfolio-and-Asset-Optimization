import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import ProtectedRoute from "./components/ProtectedRoute.jsx";
import Landing from "./pages/Landing.jsx";
import Register from "./pages/Register.jsx";
import Login from "./pages/Login.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import RiskProfile from "./pages/RiskProfile.jsx";
import Goals from "./pages/Goals.jsx";
import Holdings from "./pages/Holdings.jsx";
import Portfolio from "./pages/Portfolio.jsx";
import { useAuth } from "./context/AuthContext.jsx";

function AppRoutes() {
  const { user } = useAuth();
  return <Layout><Routes>
    <Route path="/" element={<Landing />} />
    <Route path="/register" element={user ? <Navigate to="/dashboard" replace /> : <Register />} />
    <Route path="/login" element={user ? <Navigate to="/dashboard" replace /> : <Login />} />
    <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
    <Route path="/risk-profile" element={<ProtectedRoute><RiskProfile /></ProtectedRoute>} />
    <Route path="/goals" element={<ProtectedRoute><Goals /></ProtectedRoute>} />
    <Route path="/holdings" element={<ProtectedRoute><Holdings /></ProtectedRoute>} />
    <Route path="/portfolio" element={<ProtectedRoute><Portfolio /></ProtectedRoute>} />
    <Route path="*" element={<Navigate to="/" replace />} />
  </Routes></Layout>;
}

export default function App() {
  return <AppRoutes />;
}