import { useEffect } from 'react';
import { useAuth } from '../auth/AuthContext';
import { useNavigate } from 'react-router-dom';
import './Dashboard.css'; // Create this file for styles

export default function DashboardPage() {
  const { currentUser, logout } = useAuth();
  const navigate = useNavigate();

  // Redirect if not logged in (redundant safety check)
  useEffect(() => {
    if (!currentUser) {
      navigate('/');
    }
  }, [currentUser, navigate]);

  return (
    <div className="dashboard-container">
      {/* Security Status Header */}
      <header className="security-header">
        <div className="user-info">
          <span className="user-email">{currentUser?.email}</span>
          <span className="trust-badge">Verified User</span>
        </div>
        <button 
          onClick={logout}
          className="logout-button"
          aria-label="Logout"
        >
          🔒 Logout
        </button>
      </header>

      {/* Behavioral Security Dashboard */}
      <section className="security-overview">
        <h2>Security Overview</h2>
        <div className="metrics-grid">
          <div className="metric-card">
            <h3>Session Trust Score</h3>
            <div className="score-display">87%</div>
            <p>Based on your current behavior patterns</p>
          </div>
          
          <div className="metric-card">
            <h3>Typing Profile</h3>
            <div className="confidence-level">High Confidence</div>
            <progress value="85" max="100"></progress>
          </div>
        </div>
      </section>

      {/* Quick Actions */}
      <section className="quick-actions">
        <h2>Security Controls</h2>
        <div className="action-buttons">
          <button className="action-button">
            🛡️ Lock Session
          </button>
          <button className="action-button">
            🔄 Refresh Behavior Profile
          </button>
          <button className="action-button">
            📊 View Security History
          </button>
        </div>
      </section>

      {/* Recent Activity */}
      <section className="recent-activity">
        <h2>Authentication Events</h2>
        <ul className="activity-list">
          <li>
            <span className="activity-time">Today, 10:30 AM</span>
            <span className="activity-detail">Behavior anomaly detected (mouse movement)</span>
          </li>
          <li>
            <span className="activity-time">Today, 9:15 AM</span>
            <span className="activity-detail">Successful login from Chrome on Windows</span>
          </li>
        </ul>
      </section>
    </div>
  );
}