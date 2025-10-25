import React, { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../api/client'
import { Box, Paper, Typography, TextField, Button, MenuItem, Grid, Alert } from '@mui/material'

type Role = 'docente' | 'estudiante' | 'admin'

export default function Register() {
  const nav = useNavigate()
  const [role, setRole] = useState<Role>('docente')
  const [form, setForm] = useState({
    email: '', password: '', nombres: '', apellidos: '',
    codigo: '', carrera: '', uses_glasses: false, uses_cap: false, uses_mask: false,
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [ok, setOk] = useState<string | null>(null)

  const canSubmit = useMemo(() => {
    if (!form.email || !form.password || !form.nombres || !form.apellidos) return false
    if (role === 'estudiante' && !form.codigo) return false
    return true
  }, [form, role])

  const handle = (k: string, v: any) => setForm(s => ({ ...s, [k]: v }))

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setOk(null)
    try {
      const payload: any = {
        email: form.email,
        password: form.password,
        nombres: form.nombres,
        apellidos: form.apellidos,
        rol: role,
      }
      if (role === 'estudiante') {
        payload.codigo = form.codigo
        payload.carrera = form.carrera || null
        payload.uses_glasses = !!form.uses_glasses
        payload.uses_cap = !!form.uses_cap
        payload.uses_mask = !!form.uses_mask
      }
      await api.post('/register', payload)
      setOk('Registro completado. Ahora puedes iniciar sesión.')
      setTimeout(() => nav('/login'), 800)
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Error al registrar')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box sx={{ display: 'grid', placeItems: 'center', minHeight: '70vh', p: 2 }}>
      <Paper sx={{ p: 3, width: 720, maxWidth: '90vw' }} elevation={2}>
        <Typography variant="h6" mb={2}>Registro</Typography>
        <Grid container spacing={2} component="form" onSubmit={onSubmit}>
          <Grid item xs={12} md={6}>
            <TextField label="Correo" type="email" fullWidth required value={form.email} onChange={e=>handle('email', e.target.value)} />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField label="Contraseña" type="password" fullWidth required value={form.password} onChange={e=>handle('password', e.target.value)} />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField label="Nombres" fullWidth required value={form.nombres} onChange={e=>handle('nombres', e.target.value)} />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField label="Apellidos" fullWidth required value={form.apellidos} onChange={e=>handle('apellidos', e.target.value)} />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField select label="Rol" fullWidth value={role} onChange={e=>setRole(e.target.value as Role)}>
              <MenuItem value="docente">Docente</MenuItem>
              <MenuItem value="estudiante">Estudiante</MenuItem>
            </TextField>
          </Grid>

          {role === 'estudiante' && (<>
            <Grid item xs={12} md={6}>
              <TextField label="CUI (código)" fullWidth required value={form.codigo} onChange={e=>handle('codigo', e.target.value)} />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField label="Carrera" fullWidth value={form.carrera} onChange={e=>handle('carrera', e.target.value)} />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField select label="Usa lentes frecuentemente?" fullWidth value={form.uses_glasses ? 'si':'no'} onChange={e=>handle('uses_glasses', e.target.value==='si')}>
                <MenuItem value="no">No</MenuItem>
                <MenuItem value="si">Sí</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField select label="Usa gorra frecuentemente?" fullWidth value={form.uses_cap ? 'si':'no'} onChange={e=>handle('uses_cap', e.target.value==='si')}>
                <MenuItem value="no">No</MenuItem>
                <MenuItem value="si">Sí</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField select label="Usa mascarilla frecuentemente?" fullWidth value={form.uses_mask ? 'si':'no'} onChange={e=>handle('uses_mask', e.target.value==='si')}>
                <MenuItem value="no">No</MenuItem>
                <MenuItem value="si">Sí</MenuItem>
              </TextField>
            </Grid>
          </>)}

          <Grid item xs={12}>
            {error && <Alert severity="error" sx={{ mb: 1 }}>{error}</Alert>}
            {ok && <Alert severity="success" sx={{ mb: 1 }}>{ok}</Alert>}
            <Button type="submit" variant="contained" disabled={!canSubmit || loading}>
              {loading ? 'Creando...' : 'Crear cuenta'}
            </Button>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  )
}
