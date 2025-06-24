import { useEffect, useRef } from 'react';
import axios from 'axios';
import { useAuth } from '../auth/AuthContext';

const BehaviorTracker = () => {
  const { currentUser } = useAuth();
  const behaviorData = useRef({
    typingPatterns: [],
    mouseMovements: [],
    clickEvents: [],
    deviceInfo: {},
    fingerprint: null
  });

  // Collect device info on mount (unchanged)
  useEffect(() => {
    behaviorData.current.deviceInfo = {
      browser: navigator.userAgent,
      screenResolution: `${window.screen.width}x${window.screen.height}`,
      viewportSize: `${window.innerWidth}x${window.innerHeight}`,
      os: navigator.platform,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
    };

    (async () => {
      const fp = await import('@fingerprintjs/fingerprintjs');
      const result = await (await fp.load()).get();
      behaviorData.current.fingerprint = result.visitorId;
    })();
  }, []);

  // Tracking handlers (unchanged)
  useEffect(() => {
    const handleKeyDown = (e) => {
      behaviorData.current.typingPatterns.push({
        key: e.key,
        code: e.code,
        timestamp: Date.now(),
        dwellTime: null
      });
    };

    const handleKeyUp = (e) => {
      const lastKey = behaviorData.current.typingPatterns.findLast(
        k => k.key === e.key && !k.dwellTime
      );
      if (lastKey) lastKey.dwellTime = Date.now() - lastKey.timestamp;
    };

    const handleMouseMove = (e) => {
      behaviorData.current.mouseMovements.push({
        x: e.clientX,
        y: e.clientY,
        timestamp: Date.now()
      });
    };

    const handleClick = (e) => {
      behaviorData.current.clickEvents.push({
        x: e.clientX,
        y: e.clientY,
        target: e.target.tagName,
        timestamp: Date.now()
      });
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('click', handleClick);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('click', handleClick);
    };
  }, []);

  // Fixed: Explicitly set API URL if .env fails
  const sendBehaviorData = async () => {
    if (!behaviorData.current.typingPatterns.length && 
        !behaviorData.current.mouseMovements.length) return;

    try {
<<<<<<< HEAD
      await axios.post(
        import.meta.env.VITE_API_URL ? 
          `${import.meta.env.VITE_API_URL}/api/behavior` : 
          'http://localhost:8000/api/behavior',
        {
          ...behaviorData.current,
          currentUser: currentUser ? { email: currentUser.email } : null
        }
      );
      behaviorData.current.typingPatterns = [];
      behaviorData.current.mouseMovements = [];
=======
      // We'll implement the actual API call once Faham sets up the backend
      console.log('Behavior data:', behaviorData.current);
      await axios.post('/api/behavior', behaviorData.current); // Don't forget to use npm install axios purr-a-xanos
>>>>>>> 2578055a518a045c5b89e7d593e131543ea85f4c
    } catch (error) {
      console.error('Error sending behavior data:', error);
    }
  };

  // Send data every 30 seconds (unchanged)
  useEffect(() => {
    const interval = setInterval(sendBehaviorData, 30000);
    return () => clearInterval(interval);
  }, [currentUser]);

  return null;
};

export default BehaviorTracker;
