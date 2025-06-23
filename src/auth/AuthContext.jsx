import { createContext, useContext, useState, useEffect } from 'react';
import bcrypt from 'bcryptjs';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check session on startup
    const user = sessionStorage.getItem('user');
    if (user) setCurrentUser(JSON.parse(user));
    setLoading(false);
  }, []);

  const signUp = async (email, password) => {
    const hashedPassword = await bcrypt.hash(password, 10);
    const users = JSON.parse(localStorage.getItem('users')) || {};
    
    if (users[email]) {
      throw new Error('User already exists');
    }

    users[email] = { email, password: hashedPassword };
    localStorage.setItem('users', JSON.stringify(users));
    sessionStorage.setItem('user', JSON.stringify({ email }));
    setCurrentUser({ email });
  };

  const login = async (email, password) => {
    const users = JSON.parse(localStorage.getItem('users')) || {};
    const user = users[email];

    if (!user || !(await bcrypt.compare(password, user.password))) {
      throw new Error('Invalid credentials');
    }

    sessionStorage.setItem('user', JSON.stringify({ email }));
    setCurrentUser({ email });
  };

  const logout = () => {
    sessionStorage.removeItem('user');
    setCurrentUser(null);
  };

  return (
    <AuthContext.Provider value={{ currentUser, signUp, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}