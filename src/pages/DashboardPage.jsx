import { useAuth } from '../auth/AuthContext';
import { auth } from '../firebase';
import { signOut } from 'firebase/auth';
import { useNavigate } from 'react-router-dom';
import BehaviorTracker from '../components/BehaviorTracker';
import '../styles/Dashboard.css';

export default function DashboardPage() {
  const { currentUser } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await signOut(auth);
      navigate('/login');
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div className="user-info">
          <h2>Welcome, {currentUser?.email}</h2>
          {currentUser && !currentUser.emailVerified && (
            <div className="verification-warning">
              Email not verified. Please check your inbox.
            </div>
          )}
        </div>
        <button onClick={handleLogout} className="logout-button">
          Logout
        </button>
      </header>
      <BehaviorTracker />
    </div>
  );
}