import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import BehaviorTracker from './components/BehaviorTracker';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';

function App() {
    return (
        <Router>
            <BehaviorTracker />
            <Routes>
                <Route path="/" element={<LoginPage />} />
                <Route path="/dashboard" element={<DashboardPage />} />
            </Routes>
        </Router>
    );
}

export default App;