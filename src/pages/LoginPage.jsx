import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isSignUp, setIsSignUp] = useState(false);
  const { login, signUp } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      if (isSignUp) {
        await signUp(email, password);
      } else {
        await login(email, password);
      }
      navigate('/dashboard');
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="auth-container">
      <h1>{isSignUp ? 'Sign Up' : 'Login'}</h1>
      {error && <div className="error">{error}</div>}
      <form onSubmit={handleSubmit}>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          minLength="6"
        />
        <button type="submit">{isSignUp ? 'Sign Up' : 'Login'}</button>
      </form>
      <button 
        onClick={() => setIsSignUp(!isSignUp)}
        className="toggle-auth"
      >
        {isSignUp ? 'Already have an account? Login' : 'Need an account? Sign Up'}
      </button>
    </div>
  );
}