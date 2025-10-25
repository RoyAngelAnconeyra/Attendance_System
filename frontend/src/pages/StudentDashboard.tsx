import React, { useEffect, useMemo, useState } from 'react'
import api from '../api/client'
import { Attendance, Course, Student } from '../types'
import { Box, Paper, Typography, List, ListItemText, Divider, Button, Alert, Dialog, DialogTitle, DialogContent, DialogActions, TextField, MenuItem, Select, FormControl, InputLabel } from '@mui/material'

export default function StudentDashboard() {
  const [courses, setCourses] = useState<Course[]>([])
  const [selected, setSelected] = useState<Course | null>(null)
  const [attendance, setAttendance] = useState<Attendance[]>([])
  const [err, setErr] = useState<string | null>(null)
  const [me, setMe] = useState<Student | null>(null)
  const [openCap, setOpenCap] = useState(false)
  const [photos, setPhotos] = useState<number>(30)
  const [capMsg, setCapMsg] = useState<string | null>(null)
  const [capLoading, setCapLoading] = useState(false)
  const [capCategory, setCapCategory] = useState<'normal'|'gorro'|'lentes'|'mascarilla'>('normal')
  const [joinCode, setJoinCode] = useState('')
  const [browse, setBrowse] = useState<Course[]>([])
  const [browseSelected, setBrowseSelected] = useState<number | ''>('')
  const [joinMsg, setJoinMsg] = useState<string | null>(null)

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

  useEffect(() => {
    const run = async () => {
      try {
        const res = await api.get('/students')
        const s: Student | undefined = (res.data || [])[0]
        setMe(s || null)
      } catch (e: any) {
        // ignore
      }
    }
    run()
  }, [])

  const loadAttendance = async () => {
    if (!selected) return
    try {
      const res = await api.get('/attendance', { params: { course_id: selected.id } })
      setAttendance(res.data || [])
    } catch (e: any) {
      setErr(e?.response?.data?.detail || 'No se pudo cargar asistencia')
    }
  }

  useEffect(() => { loadAttendance() }, [selected])

  const canCapture = useMemo(() => !!me?.codigo, [me])

  const toMsg = (err: any): string => {
    if (typeof err === 'string') return err
    const detail = err?.response?.data?.detail ?? err?.message ?? err
    if (Array.isArray(detail)) {
      return detail.map((d: any) => d?.msg || JSON.stringify(d)).join('; ')
    }
    if (typeof detail === 'object') return JSON.stringify(detail)
    return String(detail || 'Ocurrió un error')
  }

  const onCapture = async () => {
    if (!me?.codigo) return
    setCapLoading(true)
    setCapMsg(null)
    try {
      const res = await api.post(`/capture/${encodeURIComponent(me.codigo)}`, null, { params: { photos, categoria: capCategory } })
      const saved = res.data?.saved
      const requested = res.data?.requested
      setCapMsg(`Captura finalizada: ${saved}/${requested} fotos guardadas en dataset/${me.codigo}/${capCategory}/`)
    } catch (e: any) {
      setCapMsg(e?.response?.data?.detail || 'Error en captura')
    } finally {
      setCapLoading(false)
    }
  }

  const refreshMyCourses = async () => {
    try {
      const res = await api.get('/courses')
      setCourses(res.data || [])
    } catch {}
  }

  const loadBrowse = async () => {
    try {
      const res = await api.get('/courses/browse')
      setBrowse(res.data || [])
    } catch {}
  }

  const joinByCode = async () => {
    if (!joinCode) return
    setJoinMsg(null)
    try {
      const res = await api.post('/self-enroll/by_code', { codigo: joinCode })
      setJoinMsg(`Te uniste al curso: ${res.data?.nombre}`)
      setJoinCode('')
      await refreshMyCourses()
    } catch (e: any) {
      setJoinMsg(toMsg(e) || 'No se pudo unir al curso')
    }
  }

  const joinBySelect = async () => {
    if (!browseSelected) return
    setJoinMsg(null)
    try {
      const res = await api.post('/self-enroll', { curso_id: browseSelected })
      setJoinMsg(`Te uniste al curso: ${res.data?.nombre}`)
      setBrowseSelected('')
      await refreshMyCourses()
    } catch (e: any) {
      setJoinMsg(toMsg(e) || 'No se pudo unir al curso')
    }
  }

  useEffect(() => { loadBrowse() }, [])

  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 2, p: 2 }}>
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" mb={1}>Mis cursos</Typography>
        <List>
          {courses.map(c => (
            <Button key={c.id} variant={selected?.id===c.id? 'contained':'outlined'} onClick={() => setSelected(c)} sx={{ mr: 1, mb: 1 }}>
              {c.nombre}
            </Button>
          ))}
          {courses.length === 0 && <Typography variant="body2" color="text.secondary">No hay cursos</Typography>}
        </List>
      </Paper>

      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" mb={1}>Unirme a un curso</Typography>
        <Box sx={{ display: 'flex', gap: 1, mb: 2, alignItems: 'center', flexWrap: 'wrap' }}>
          <TextField size="small" label="Código de curso" value={joinCode} onChange={(e)=>setJoinCode(e.target.value)} />
          <Button variant="contained" onClick={joinByCode} disabled={!joinCode}>Unirme</Button>
        </Box>
        <Box sx={{ display: 'flex', gap: 1, mb: 1, alignItems: 'center', flexWrap: 'wrap' }}>
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel id="browse-label">Seleccionar curso</InputLabel>
            <Select labelId="browse-label" label="Seleccionar curso" value={browseSelected} onChange={(e)=>setBrowseSelected(e.target.value as any)}>
              {browse.map(c => (
                <MenuItem key={c.id} value={c.id}>{c.nombre} · {c.codigo}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <Button variant="outlined" onClick={joinBySelect} disabled={!browseSelected}>Unirme</Button>
          <Button onClick={loadBrowse}>Actualizar lista</Button>
        </Box>
      </Paper>

      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" mb={1}>{selected ? selected.nombre : 'Selecciona un curso'}</Typography>
        <Divider sx={{ mb: 2 }} />
        {err && <Alert severity="error" sx={{ mb: 1 }}>{err}</Alert>}
        {joinMsg && <Alert severity="info" sx={{ mb: 1 }}>{joinMsg}</Alert>}
        <Typography variant="subtitle1" gutterBottom>Mi asistencia</Typography>
        <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
          <Button variant="outlined" onClick={() => setOpenCap(true)} disabled={!canCapture}>Capturar fotos (webcam)</Button>
        </Box>
        <List dense>
          {attendance.map(a => (
            <ListItemText key={a.id} primary={`${a.estado.toUpperCase()}`} secondary={`Registro #${a.id}`} />
          ))}
          {selected && attendance.length === 0 && <Typography variant="body2" color="text.secondary">Sin registros</Typography>}
        </List>
      </Paper>

      <Dialog open={openCap} onClose={() => setOpenCap(false)} fullWidth maxWidth="sm">
        <DialogTitle>Capturar fotos para mi perfil</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Se tomarán recortes de rostro mediante MTCNN y se guardarán en <code>dataset/{me?.codigo}/{capCategory}/</code> en el servidor.
          </Typography>
          <FormControl fullWidth size="small" sx={{ mb: 2 }}>
            <InputLabel id="cap-cat">Categoría</InputLabel>
            <Select labelId="cap-cat" label="Categoría" value={capCategory} onChange={(e)=>setCapCategory(e.target.value as any)}>
              <MenuItem value="normal">Normal</MenuItem>
              <MenuItem value="gorro">Gorro</MenuItem>
              <MenuItem value="lentes">Lentes</MenuItem>
              <MenuItem value="mascarilla">Mascarilla</MenuItem>
            </Select>
          </FormControl>
          <TextField
            type="number"
            label="Cantidad de fotos"
            value={photos}
            onChange={(e) => setPhotos(Math.max(1, Number(e.target.value) || 1))}
            fullWidth
            inputProps={{ min: 1, max: 200 }}
          />
          {capMsg && <Alert severity="info" sx={{ mt: 2 }}>{capMsg}</Alert>}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenCap(false)}>Cerrar</Button>
          <Button onClick={onCapture} variant="contained" disabled={capLoading || !canCapture}>
            {capLoading ? 'Capturando...' : 'Iniciar captura'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}
