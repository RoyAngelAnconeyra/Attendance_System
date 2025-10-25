import React from 'react'
import { Routes, Route, Navigate, Link } from 'react-router-dom'
import Landing from './pages/Landing'
import Login from './pages/Login'
import Register from './pages/Register'
import ProfessorDashboard from './pages/ProfessorDashboard'
import StudentDashboard from './pages/StudentDashboard'
import { AuthProvider, useAuth } from './context/AuthContext'

function Protected({ children, allowed }: { children: React.ReactNode; allowed?: ('docente'|'estudiante'|'admin')[] }) {
  const { user } = useAuth()
  if (!user) return <Navigate to="/login" replace />
  if (allowed && !allowed.includes(user.rol)) return <Navigate to="/" replace />
  return <>{children}</>
}

function Shell({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth()
  return (
    <div style={{ fontFamily: 'Inter, system-ui, Arial', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{ borderBottom: '1px solid #eee', padding: '12px 20px', display: 'flex', justifyContent: 'space-between' }}>
        <Link to="/" style={{ textDecoration: 'none', color: '#111', fontWeight: 700 }}>Asistencia Universitaria</Link>
        <nav style={{ display: 'flex', gap: 12 }}>
          {!user && (<>
            <Link to="/login">Iniciar sesión</Link>
            <Link to="/register">Registrarse</Link>
          </>)}
          {user && (<>
            {user.rol === 'docente' && <Link to="/prof">Profesor</Link>}
            {user.rol === 'estudiante' && <Link to="/student">Estudiante</Link>}
            <button onClick={logout} style={{ border: '1px solid #ddd', padding: '6px 10px', borderRadius: 6, background: 'white' }}>Salir</button>
          </>)}
        </nav>
      </header>
      <main style={{ flex: 1 }}>{children}</main>
      <footer style={{ borderTop: '1px solid #eee', padding: 16, textAlign: 'center', color: '#666' }}>
        Reconocimiento facial (MTCNN + FaceNet + SVM)
      </footer>
    </div>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <Shell>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          {/*<Route path="/prof" element={<Protected allowed={['docente','admin']}><ProfessorDashboard /></Protected>} />*/}
          <Route path="/prof" element={<ProfessorDashboard />} />
          {/*<Route path="/student" element={<Protected allowed={['estudiante']}><StudentDashboard /></Protected>} />*/}
          <Route path="/student" element={<StudentDashboard />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Shell>
    </AuthProvider>
  )
}
