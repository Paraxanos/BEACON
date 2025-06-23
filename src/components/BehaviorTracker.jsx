import { useEffect, useRef } from 'react';
import FingerprintJS from '@fingerprintjs/fingerprintjs';

const BehaviorTracker = () => {
  const behaviorData = useRef({
    typingPatterns: [],
    mouseMovements: [],
    clickEvents: [],
    deviceInfo: {},
    fingerprint: null
  });

  useEffect(() => {
    // Initialize fingerprinting
    const getFingerprint = async () => {
      const fp = await FingerprintJS.load();
      const result = await fp.get();
      behaviorData.current.fingerprint = result.visitorId;
    };
    getFingerprint();

    // Collect device info
    behaviorData.current.deviceInfo = {
      browser: navigator.userAgent,
      screenResolution: `${window.screen.width}x${window.screen.height}`,
      viewportSize: `${window.innerWidth}x${window.innerHeight}`,
      os: navigator.platform,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
    };

    // Typing pattern tracking
    const handleKeyDown = (e) => {
      const timestamp = Date.now();
      behaviorData.current.typingPatterns.push({
        key: e.key,
        code: e.code,
        timestamp,
        dwellTime: null // will be updated on keyup
      });
    };

    const handleKeyUp = (e) => {
      const timestamp = Date.now();
      const lastKeyDown = behaviorData.current.typingPatterns.findLast(
        entry => entry.key === e.key && entry.dwellTime === null
      );
      if (lastKeyDown) {
        lastKeyDown.dwellTime = timestamp - lastKeyDown.timestamp;
      }
    };

    // Mouse movement tracking
    const handleMouseMove = (e) => {
      behaviorData.current.mouseMovements.push({
        x: e.clientX,
        y: e.clientY,
        timestamp: Date.now()
      });
    };

    // Click tracking
    const handleClick = (e) => {
      behaviorData.current.clickEvents.push({
        x: e.clientX,
        y: e.clientY,
        target: e.target.tagName,
        timestamp: Date.now(),
        hoverTime: null // will be calculated based on mouseenter/leave
      });
    };

    // Add event listeners
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

  // Function to send data to backend
  const sendBehaviorData = async () => {
    try {
      // We'll implement the actual API call once Faham sets up the backend
      console.log('Behavior data:', behaviorData.current);
      await axios.post('/api/behavior', behaviorData.current); // Don't forget to use npm install axios purr-a-xanos
    } catch (error) {
      console.error('Error sending behavior data:', error);
    }
  };

  // Send data periodically (every 30 seconds)
  useEffect(() => {
    const interval = setInterval(sendBehaviorData, 30000);
    return () => clearInterval(interval);
  }, []);

  return null; // This is a non-visual component
};

export default BehaviorTracker;
