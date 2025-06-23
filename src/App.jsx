import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './auth/AuthContext';
import BehaviorTracker from './components/BehaviorTracker';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';

function PrivateRoute({ children }) {
    const { currentUser } = useAuth();
    return currentUser ? children : <Navigate to="/" />;
}

export default function App() {
    return (
        <AuthProvider>
            <Router>
                <BehaviorTracker />
                <Routes>
                    <Route path="/" element={<LoginPage />} />
                    <Route
                        path="/dashboard"
                        element={
                            <PrivateRoute>
                                <DashboardPage />
                            </PrivateRoute>
                        }
                    />
                </Routes>
            </Router>
        </AuthProvider>
    );
}