import { useState, useEffect, useRef } from 'react';
import { useAuth } from '../auth/AuthContext';
import { auth } from '../firebase';
import { signOut } from 'firebase/auth';
import { useNavigate } from 'react-router-dom';
import BehaviorTracker from '../components/BehaviorTracker';
import '../styles/Dashboard.css';

export default function DashboardPage() {
  const { currentUser } = useAuth();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('account');
  const [trustScore, setTrustScore] = useState(100);
  const [trustStatus, setTrustStatus] = useState('normal');
  const [safeMode, setSafeMode] = useState(false);
  const [location, setLocation] = useState(null);
  const [locationError, setLocationError] = useState(null);
  const behaviorData = useRef({
    clickEvents: [],
    typingPatterns: [],
    mouseMovements: [],
    deviceInfo: {},
    initialized: true
  });

  // Initialize device info with geolocation
  useEffect(() => {
    const initDeviceInfo = () => {
      behaviorData.current.deviceInfo = {
        userAgent: navigator.userAgent,
        screen: `${window.screen.width}x${window.screen.height}`,
        viewportSize: `${window.innerWidth}x${window.innerHeight}`,
        browser: navigator.userAgentData?.brands?.[0]?.brand || 'unknown',
        os: navigator.userAgentData?.platform || 'unknown',
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        userId: currentUser?.uid || 'anonymous',
        safeMode,
        location: location ? { 
          latitude: location.latitude, 
          longitude: location.longitude 
        } : null
      };
    };

    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLocation({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            accuracy: position.coords.accuracy
          });
          initDeviceInfo();
        },
        (error) => {
          setLocationError(error.message);
          console.error('Geolocation error:', error);
          initDeviceInfo();
        }
      );
    } else {
      setLocationError('Geolocation is not supported by this browser.');
      initDeviceInfo();
    }
  }, [currentUser]);

  // Track mouse movements (reduced intensity in safe mode)
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (safeMode && Math.random() > 0.7) return; // Sample less in safe mode
      behaviorData.current.mouseMovements.push({
        x: e.clientX,
        y: e.clientY,
        timestamp: Date.now()
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [safeMode]);

  // Track clicks (reduced intensity in safe mode)
  useEffect(() => {
    const handleClick = (e) => {
      if (safeMode && Math.random() > 0.5) return; // Sample less in safe mode
      behaviorData.current.clickEvents.push({
        x: e.clientX,
        y: e.clientY,
        target: e.target.tagName,
        timestamp: Date.now(),
        hoverTime: 0
      });
    };

    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [safeMode]);

  // Track typing (reduced intensity in safe mode)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (safeMode && Math.random() > 0.6) return; // Sample less in safe mode
      behaviorData.current.typingPatterns.push({
        key: e.key,
        code: e.code,
        timestamp: Date.now(),
        dwellTime: null
      });
    };

    const handleKeyUp = (e) => {
      const lastKeyPress = behaviorData.current.typingPatterns.findLast(
        kp => kp.key === e.key && kp.dwellTime === null
      );
      if (lastKeyPress) {
        lastKeyPress.dwellTime = Date.now() - lastKeyPress.timestamp;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [safeMode]);

  // Send data to backend periodically with context info
  const sendBehaviorData = async () => {
    if (safeMode && Math.random() > 0.3) return; // Send less frequently in safe mode
    
    if (!behaviorData.current.typingPatterns.length && 
        !behaviorData.current.mouseMovements.length && 
        !behaviorData.current.clickEvents.length) {
      return;
    }

    const payload = {
      timestamp: new Date().toISOString(),
      user_email: currentUser?.email || 'anonymous',
      context: {
        safeMode,
        location,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
      },
      data: {
        typingPatterns: behaviorData.current.typingPatterns,
        mouseMovements: behaviorData.current.mouseMovements,
        clickEvents: behaviorData.current.clickEvents,
        deviceInfo: behaviorData.current.deviceInfo,
        fingerprint: `user-${currentUser?.uid || 'anonymous'}`
      }
    };

    try {
      const response = await fetch('http://localhost:8000/api/behavior', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      
      const rawScore = data.anomaly_score;
      const normalizedScore = 50 + (rawScore * 100);
      const clampedScore = Math.min(100, Math.max(0, Math.round(normalizedScore)));
      
      setTrustScore(clampedScore);
      setTrustStatus(data.is_anomalous ? 'anomalous' : 'normal');

      behaviorData.current.typingPatterns = [];
      behaviorData.current.mouseMovements = [];
      behaviorData.current.clickEvents = [];

    } catch (error) {
      console.error('Error sending behavior data:', error);
    }
  };

  useEffect(() => {
    const interval = setInterval(sendBehaviorData, safeMode ? 30000 : 15000);
    return () => clearInterval(interval);
  }, [currentUser, safeMode]);

  const handleLogout = async () => {
    try {
      await signOut(auth);
      navigate('/login');
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'account':
        return (
          <div className="tab-content">
            <h3>Account Information</h3>
            <div className="account-details">
              <div className="detail-row">
                <span className="detail-label">Account Holder:</span>
                <span className="detail-value">{currentUser?.email}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Account Number:</span>
                <span className="detail-value">XXXX-XXXX-7890</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Account Type:</span>
                <span className="detail-value">Premium Savings</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Balance:</span>
                <span className="detail-value">$12,456.78</span>
              </div>
            </div>
          </div>
        );
      case 'transfer':
        return (
          <div className="tab-content">
            <h3>Transfer Money</h3>
            <div className="transfer-form">
              <div className="form-group">
                <label>Recipient Account</label>
                <input type="text" placeholder="Enter account number" />
              </div>
              <div className="form-group">
                <label>Amount ($)</label>
                <input type="number" placeholder="0.00" />
              </div>
              <div className="form-group">
                <label>Description</label>
                <input type="text" placeholder="Optional" />
              </div>
              <button className="transfer-button">Transfer Now</button>
            </div>
          </div>
        );
      case 'funds':
        return (
          <div className="tab-content">
            <h3>Mutual Funds</h3>
            <div className="funds-grid">
              <div className="fund-card">
                <h4>Global Equity Fund</h4>
                <p>1Y Return: +12.5%</p>
                <button className="invest-button">Invest</button>
              </div>
              <div className="fund-card">
                <h4>Bond Income Fund</h4>
                <p>1Y Return: +5.2%</p>
                <button className="invest-button">Invest</button>
              </div>
              <div className="fund-card">
                <h4>Tech Growth Fund</h4>
                <p>1Y Return: +18.7%</p>
                <button className="invest-button">Invest</button>
              </div>
            </div>
          </div>
        );
      case 'trust':
        return (
          <div className="tab-content">
            <h3>Behavioral Trust Score</h3>
            <div className="trust-score-info">
              <p>Your trust score is continuously updated based on your interaction patterns.</p>
              <BehaviorTracker 
                trustScore={trustScore}
                trustStatus={trustStatus}
              />
              
              <div className="context-controls">
                <h4>Context Awareness Settings</h4>
                <div className="toggle-group">
                  <label>
                    <input 
                      type="checkbox" 
                      checked={safeMode}
                      onChange={() => setSafeMode(!safeMode)}
                    />
                    <span>Safe Mode (reduces tracking intensity)</span>
                  </label>
                </div>
                {location ? (
                  <div className="location-info">
                    <p>Current location: {location.latitude.toFixed(4)}, {location.longitude.toFixed(4)}</p>
                    <p>Accuracy: ±{Math.round(location.accuracy)} meters</p>
                  </div>
                ) : (
                  <div className="location-error">
                    {locationError || 'Location data not available'}
                  </div>
                )}
              </div>

              <div className="trust-tips">
                <h4>Tips to maintain high trust score:</h4>
                <ul>
                  <li>Use consistent typing patterns</li>
                  <li>Avoid rapid, erratic mouse movements</li>
                  <li>Use the same device when possible</li>
                  <li>Keep location services enabled</li>
                </ul>
              </div>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div className="bank-logo">
          <h1>SecureBank</h1>
        </div>
        <div className="user-info">
          <span className="welcome-message">Welcome, {currentUser?.email}</span>
          {currentUser && !currentUser.emailVerified && (
            <div className="verification-warning">
              Email not verified. Please check your inbox.
            </div>
          )}
          <button onClick={handleLogout} className="logout-button">
            Logout
          </button>
        </div>
      </header>

      <div className="dashboard-container">
        <nav className="dashboard-nav">
          <button
            className={`nav-button ${activeTab === 'account' ? 'active' : ''}`}
            onClick={() => setActiveTab('account')}
          >
            <i className="fas fa-user"></i> Account Info
          </button>
          <button
            className={`nav-button ${activeTab === 'transfer' ? 'active' : ''}`}
            onClick={() => setActiveTab('transfer')}
          >
            <i className="fas fa-exchange-alt"></i> Transfer Money
          </button>
          <button
            className={`nav-button ${activeTab === 'funds' ? 'active' : ''}`}
            onClick={() => setActiveTab('funds')}
          >
            <i className="fas fa-chart-line"></i> Mutual Funds
          </button>
          <button
            className={`nav-button ${activeTab === 'trust' ? 'active' : ''}`}
            onClick={() => setActiveTab('trust')}
          >
            <i className="fas fa-shield-alt"></i> Trust Score
          </button>
        </nav>

        <main className="dashboard-content">
          {renderTabContent()}
        </main>
      </div>
    </div>
  );
}