import React, { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { decodeJwt } from '../utils/jwt'

export type User = {
  id?: number
  email?: string
  rol: 'admin' | 'docente' | 'estudiante'
}

type AuthContextType = {
  user: User | null
  token: string | null
  login: (token: string) => void
  logout: () => void
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  token: null,
  login: () => {},
  logout: () => {},
})

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem('token'))
  const [user, setUser] = useState<User | null>(null)

  useEffect(() => {
    if (!token) {
      setUser(null)
      return
    }
    try {
      const payload = decodeJwt(token)
      const role = (payload?.role || payload?.rol) as User['rol'] | undefined
      const uid = payload?.sub ? Number(payload.sub) : undefined
      setUser(role ? { id: uid, rol: role } : null)
    } catch {
      setUser(null)
    }
  }, [token])

  const login = (tk: string) => {
    localStorage.setItem('token', tk)
    setToken(tk)
  }

  const logout = () => {
    localStorage.removeItem('token')
    setToken(null)
    setUser(null)
  }

  const value = useMemo(() => ({ user, token, login, logout }), [user, token])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  return useContext(AuthContext)
}
