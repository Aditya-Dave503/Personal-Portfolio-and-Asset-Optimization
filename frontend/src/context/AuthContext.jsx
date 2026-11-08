import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { authApi } from "../services/api.js";

const AuthContext = createContext(null);
const TOKEN_KEY = "optiwealth_access_token";

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [user, setUser] = useState(null);
  const [checking, setChecking] = useState(Boolean(token));

  useEffect(() => {
    let active = true;
    if (!token) {
      setUser(null);
      setChecking(false);
      return () => {
        active = false;
      };
    }

    setChecking(true);
    authApi
      .me(token)
      .then((data) => {
        if (active) setUser(data);
      })
      .catch((err) => {
        if (active) {
          if (err.status === 401) {
            localStorage.removeItem(TOKEN_KEY);
            setToken(null);
            setUser(null);
          }
        }
      })
      .finally(() => {
        if (active) setChecking(false);
      });

    return () => {
      active = false;
    };
  }, [token]);

  const value = useMemo(
    () => ({
      token,
      user,
      checking,
      async login(email, password) {
        const result = await authApi.login(email, password);
        localStorage.setItem(TOKEN_KEY, result.access_token);
        setToken(result.access_token);
        const currentUser = await authApi.me(result.access_token);
        setUser(currentUser);
        return currentUser;
      },
      async register(email, password) {
        return authApi.register(email, password);
      },
      logout() {
        localStorage.removeItem(TOKEN_KEY);
        setToken(null);
        setUser(null);
      },
    }),
    [token, user, checking]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used inside AuthProvider");
  return value;
}
