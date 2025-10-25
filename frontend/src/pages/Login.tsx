import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../api/client'
import { useAuth } from '../context/AuthContext'
import { Button, TextField, Paper, Typography, Box } from '@mui/material'

export default function Login() {
  const { login } = useAuth()
  const nav = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const res = await api.post('/login', { email, password })
      const token = res.data?.access_token
      if (token) {
        login(token)
        nav('/')
      } else {
        setError('Respuesta inválida del servidor')
      }
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Error de autenticación')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box sx={{ display: 'grid', placeItems: 'center', minHeight: '70vh', p: 2 }}>
      <Paper sx={{ p: 3, width: 360 }} elevation={2}>
        <Typography variant="h6" mb={2}>Iniciar sesión</Typography>
        <form onSubmit={onSubmit}>
          <TextField label="Correo" type="email" value={email} onChange={e => setEmail(e.target.value)} fullWidth margin="normal" required />
          <TextField label="Contraseña" type="password" value={password} onChange={e => setPassword(e.target.value)} fullWidth margin="normal" required />
          {error && <Typography color="error" variant="body2" mt={1}>{error}</Typography>}
          <Button type="submit" variant="contained" fullWidth sx={{ mt: 2 }} disabled={loading}>
            {loading ? 'Ingresando...' : 'Ingresar'}
          </Button>
        </form>
      </Paper>
    </Box>
  )
}
