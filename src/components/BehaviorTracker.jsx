import '../styles/TrustScore.css';

const BehaviorTracker = ({ 
  trustScore, 
  trustStatus
}) => {
  return (
    <div className="trust-score-container">
      <div className="trust-score-header">
        <h3 className="trust-score-title">Current Trust Level</h3>
        <span className={`status-indicator status-${trustStatus}`}>
          {trustStatus === 'normal' ? 'Normal' : 'Anomalous'}
        </span>
      </div>

      <div className="score-display">
        <div className="score-value">{trustScore}%</div>
        <div className="score-bar">
          <div 
            className={`score-fill ${trustStatus === 'anomalous' ? 'score-fill-anomalous' : ''}`}
            style={{ width: `${trustScore}%` }}
          />
        </div>
      </div>

      <div className="trust-score-footer">
        Last updated: {new Date().toLocaleTimeString()}
      </div>
    </div>
  );
};

export default BehaviorTracker;