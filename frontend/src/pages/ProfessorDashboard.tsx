import React, { useEffect, useMemo, useRef, useState } from 'react'
import api from '../api/client'
import { Course, Student, Attendance } from '../types'
import { Box, Paper, Typography, List, ListItem, ListItemButton, ListItemText, Divider, Button, Alert, TextField } from '@mui/material'

export default function ProfessorDashboard() {
  const [courses, setCourses] = useState<Course[]>([])
  const [selected, setSelected] = useState<Course | null>(null)
  const [students, setStudents] = useState<Student[]>([])
  const [loading, setLoading] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [newName, setNewName] = useState('')
  const [newCode, setNewCode] = useState('')
  const [newHorario, setNewHorario] = useState('')
  const [enrollCui, setEnrollCui] = useState('')
  const [trainLoading, setTrainLoading] = useState(false)
  const [trainMsg, setTrainMsg] = useState<string | null>(null)
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [wsConnected, setWsConnected] = useState(false)
  const [live, setLive] = useState<Array<{ codigo?: string, confidence?: number, event?: string, ts: number }>>([])
  const wsTick = useRef<number | null>(null)
  const [attendance, setAttendance] = useState<Attendance[]>([])

  useEffect(() => {
    const run = async () => {
      try {
        const res = await api.get('/courses')
        setCourses(res.data || [])
      } catch (e: any) {
        setErr(e?.response?.data?.detail || 'No se pudieron cargar cursos')
      }
    }
    run()
  }, [])

  const trainModel = async () => {
    setTrainMsg(null)
    setErr(null)
    setTrainLoading(true)
    try {
      const res = await api.post('/train')
      const classes = (res.data?.classes || []).join(', ')
      setTrainMsg(`Modelo entrenado. Clases: ${classes} · Muestras: ${res.data?.samples}`)
    } catch (e: any) {
      setErr(e?.response?.data?.detail || 'No se pudo entrenar el modelo')
    } finally {
      setTrainLoading(false)
    }
  }

  const wsUrlForCourse = (courseId: number) => {
    const base = (import.meta.env.VITE_API_URL || 'http://localhost:8000/api') as string
    const httpBase = base.endsWith('/api') ? base.slice(0, -4) : base
    const wsBase = httpBase.replace('http://', 'ws://').replace('https://', 'wss://')
    return `${wsBase}/api/recognize/ws?course_id=${courseId}`
  }

  const connectWS = () => {
    if (!selected || wsConnected) return
    const url = wsUrlForCourse(selected.id)
    const sock = new WebSocket(url)
    sock.onopen = () => {
      setWsConnected(true)
      setWs(sock)
      if (wsTick.current) window.clearInterval(wsTick.current)
      wsTick.current = window.setInterval(() => { try { sock.send('tick') } catch {} }, 1000)
    }
    sock.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        const ts = Date.now()
        setLive(prev => [{ ...data, ts }, ...prev].slice(0, 20))
      } catch {}
    }
    sock.onclose = () => {
      setWsConnected(false)
      setWs(null)
      if (wsTick.current) { window.clearInterval(wsTick.current); wsTick.current = null }
    }
    sock.onerror = () => {
      setErr('Error en WebSocket')
    }
  }

  const disconnectWS = () => {
    if (ws) {
      try { ws.close() } catch {}
    }
  }

  useEffect(() => {
    const run = async () => {
      if (!selected) return
      try {
        const res = await api.get('/students', { params: { course_id: selected.id } })
        setStudents(res.data || [])
      } catch (e: any) {
        setErr(e?.response?.data?.detail || 'No se pudieron cargar estudiantes')
      }
    }
    run()
  }, [selected])

  useEffect(() => {
    const run = async () => {
      if (!selected) { setAttendance([]); return }
      const today = new Date()
      const on_date = today.toISOString().slice(0,10) // YYYY-MM-DD
      try {
        const res = await api.get('/attendance', { params: { course_id: selected.id, on_date } })
        setAttendance(res.data || [])
      } catch (e: any) {
        // no bloquear la UI del profesor si falla asistencia
      }
    }
    run()
  }, [selected, msg])

  const startRecognition = async () => {
    if (!selected) return
    setLoading(true)
    setMsg(null)
    setErr(null)
    try {
      const res = await api.get('/recognize/start', { params: { course_id: selected.id, duration_minutes: 2 } })
      setMsg(res.data?.message || 'Sesión iniciada (demo)')
    } catch (e: any) {
      setErr(e?.response?.data?.detail || 'No se pudo iniciar reconocimiento')
    } finally {
      setLoading(false)
    }
  }

  const createCourse = async () => {
    setErr(null); setMsg(null)
    try {
      const res = await api.post('/courses', { nombre: newName, codigo: newCode, horario: newHorario })
      setMsg(`Curso creado: ${res.data?.nombre}`)
      setNewName(''); setNewCode(''); setNewHorario('')
      const cur = await api.get('/courses')
      setCourses(cur.data || [])
    } catch (e: any) {
      setErr(e?.response?.data?.detail || 'No se pudo crear el curso')
    }
  }

  const enrollByCui = async () => {
    if (!selected || !enrollCui) return
    setErr(null); setMsg(null)
    try {
      const stu = await api.get('/students/by_code', { params: { codigo: enrollCui } })
      const estudiante_id = stu.data?.id
      await api.post('/enroll', { estudiante_id, curso_id: selected.id })
      setMsg(`Estudiante ${enrollCui} matriculado`)
      setEnrollCui('')
      // refresh students list
      const res = await api.get('/students', { params: { course_id: selected.id } })
      setStudents(res.data || [])
    } catch (e: any) {
      setErr(e?.response?.data?.detail || 'No se pudo matricular')
    }
  }

  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 2, p: 2 }}>
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" mb={1}>Mis cursos</Typography>
        <List>
          {courses.map(c => (
            <ListItemButton key={c.id} selected={selected?.id === c.id} onClick={() => setSelected(c)}>
              <ListItemText primary={c.nombre} secondary={`${c.codigo}${c.horario ? ' · '+c.horario: ''}`} />
            </ListItemButton>
          ))}
          {courses.length === 0 && <Typography variant="body2" color="text.secondary">No hay cursos</Typography>}
        </List>
      </Paper>

      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" mb={1}>Reconocimiento en vivo (stream)</Typography>
        <List dense>
          {live.map(item => (
            <ListItemText key={item.ts} primary={item.codigo ? `${item.codigo} (${(item.confidence ?? 0).toFixed(2)})` : (item.event || 'evento')} secondary={new Date(item.ts).toLocaleTimeString()} />
          ))}
          {live.length === 0 && <Typography variant="body2" color="text.secondary">Sin eventos</Typography>}
        </List>
      </Paper>

      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" mb={1}>{selected ? selected.nombre : 'Selecciona un curso'}</Typography>
        <Divider sx={{ mb: 2 }} />
        {err && <Alert severity="error" sx={{ mb: 1 }}>{err}</Alert>}
        {msg && <Alert severity="success" sx={{ mb: 1 }}>{msg}</Alert>}
        {trainMsg && <Alert severity="info" sx={{ mb: 1 }}>{trainMsg}</Alert>}
        <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
          <Button variant="outlined" onClick={trainModel} disabled={trainLoading}>{trainLoading ? 'Entrenando...' : 'Entrenar modelo'}</Button>
          <Button variant="outlined" onClick={connectWS} disabled={!selected || wsConnected}>Conectar WS</Button>
          <Button variant="outlined" onClick={disconnectWS} disabled={!wsConnected}>Desconectar WS</Button>
          <Typography variant="body2" color="text.secondary">{wsConnected ? 'WS conectado' : 'WS desconectado'}</Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
          <Button variant="contained" disabled={!selected || loading} onClick={startRecognition}>
            {loading ? 'Iniciando...' : 'Iniciar reconocimiento (demo)'}
          </Button>
        </Box>
        {selected && (
          <Box sx={{ display: 'flex', gap: 1, mb: 2, alignItems: 'center' }}>
            <TextField size="small" label="CUI estudiante" value={enrollCui} onChange={e=>setEnrollCui(e.target.value)} />
            <Button variant="outlined" onClick={enrollByCui} disabled={!enrollCui}>Matricular</Button>
          </Box>
        )}
        <Typography variant="subtitle1" gutterBottom>Estudiantes</Typography>
        <List dense>
          {students.map(s => {
            const att = attendance.filter(a => a.estudiante_id === s.id).sort((a,b)=> (a.fecha_hora||'').localeCompare(b.fecha_hora||''))[0]
            const present = att && att.estado === 'presente'
            const color = present ? 'success.main' : 'error.main'
            const time = present && att?.fecha_hora ? new Date(att.fecha_hora).toLocaleTimeString() : undefined
            return (
              <ListItem key={s.id} sx={{
                borderLeft: theme => `4px solid ${theme.palette[present ? 'success' : 'error'].main}`,
                bgcolor: theme => theme.palette.action.hover,
                mb: 0.5,
                borderRadius: 1,
              }}>
                <ListItemText
                  primaryTypographyProps={{ sx: { color, fontWeight: 700 } }}
                  secondaryTypographyProps={{ sx: { fontWeight: 500 } }}
                  primary={`${s.codigo}${present && time ? ` · ${time}` : ''}`}
                  secondary={s.carrera || (present ? 'Presente' : 'Ausente')}
                />
              </ListItem>
            )
          })}
          {selected && students.length === 0 && <Typography variant="body2" color="text.secondary">Sin estudiantes</Typography>}
        </List>
      </Paper>

      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" mb={1}>Crear curso</Typography>
        <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
          <TextField size="small" label="Nombre" value={newName} onChange={e=>setNewName(e.target.value)} />
          <TextField size="small" label="Código" value={newCode} onChange={e=>setNewCode(e.target.value)} />
          <TextField size="small" label="Horario" value={newHorario} onChange={e=>setNewHorario(e.target.value)} />
          <Button variant="contained" onClick={createCourse} disabled={!newName || !newCode}>Crear</Button>
        </Box>
      </Paper>
    </Box>
  )
}
