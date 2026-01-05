import React, { useEffect, useMemo, useRef, useState } from 'react'
import api from '../api/client'
import DeleteIcon from '@mui/icons-material/Delete';
import { Course, Student, Attendance, User } from '../types'
import { Box, Paper, Typography, List, ListItem, ListItemButton, ListItemText, Divider, Button, Alert, TextField, IconButton, Grid } from '@mui/material'

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
  const [refreshTick, setRefreshTick] = useState(0)

  const bumpRefresh = () => setRefreshTick((v) => v + 1)

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

  const trainModel = async (courseId: number, courseName: string) => {
    if (!courseId || !courseName) {
      console.error("Falta courseId o courseName");
      return;
    }

    setTrainLoading(true);
    setTrainMsg(null);
    setErr(null);

    try {
      const response = await api.post('/train', null, {
        params: { 
          course_id: courseId, 
          course_name: courseName 
        }
      });
      setTrainMsg(response.data?.message || 'Modelo entrenado exitosamente');
      return response.data;
    } catch (error) {
      console.error('Error al entrenar el modelo:', error);
      setErr(error?.response?.data?.detail || 'Error al entrenar el modelo');
      throw error;
    } finally {
      setTrainLoading(false);
    }
  };

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
        const data = JSON.parse(ev.data);
        const ts = Date.now();

        const today = new Date().toISOString().slice(0, 10);
        const refreshAttendance = () => {
          if (!selected) return;
          api.get('/attendance', { 
            params: { 
              course_id: selected.id, 
              on_date: today 
            } 
          }).then(res => setAttendance(res.data || []))
            .catch(() => {});
        };

        if (data.type === 'recognition_completed' || data.event === 'attendance_marked') {
          console.log('[WS] Evento asistencia', data);
          refreshAttendance();
          bumpRefresh();
        }

        // Mantener la lógica existente para actualizaciones en tiempo real
        setLive(prev => {
          const newLive = [{ ...data, ts }, ...prev].slice(0, 20);
          return newLive;
        });
      } catch (e) {
        console.error('Error procesando mensaje WebSocket:', e);
      }
    };
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
    // Limpiar el estado de reconocimiento en vivo al desconectar
    setLive([])
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
  }, [selected, msg, refreshTick])

  const startRecognition = async () => {
    if (!selected) return;
    setLoading(true);
    setMsg(null);
    setErr(null);
    // Limpiar resultados previos para evitar mostrar asistencias antiguas
    setLive([]);
    setAttendance([]);
    
    try {
      // 1. Iniciar el reconocimiento oficial (marca asistencia)
      const res = await api.get('/recognize/start', { 
        params: { 
          course_id: selected.id, 
          duration_minutes: 2 
        } 
      });
      
      // El backend devuelve información detallada del reconocimiento
      const recognized = res.data?.recognized || [];
      const summary = res.data?.session_summary || {};
      console.log('[Reconocimiento] Resumen', { recognized, summary });
      
      
      // 2. Actualizar inmediatamente la asistencia y estudiantes
      const today = new Date();
      const on_date = today.toISOString().slice(0, 10);
      
      try {
        // Obtener la asistencia actualizada del día
        const attendanceRes = await api.get('/attendance', { 
          params: { 
            course_id: selected.id, 
            on_date 
          } 
        });
        
        // Obtener la lista de estudiantes
        const studentsRes = await api.get('/students', { 
          params: { course_id: selected.id } 
        });
        
        // Actualizar el estado de asistencia
        setAttendance(attendanceRes.data || []);
        
        // Actualizar estudiantes con información de asistencia
        if (studentsRes.data?.length) {
          setStudents(studentsRes.data.map((student: Student) => {
            const studentAttendance = attendanceRes.data?.find(
              (a: Attendance) => a.estudiante_id === student.id
            );
            
            // NO usar lastRecognition aquí, solo usar attendance
            // lastRecognition es solo para el WebSocket (que ya no lo usa)
            return {
              ...student,
              // Limpiar lastRecognition si existe (solo debe venir de asistencia oficial)
              lastRecognition: undefined
            };
          }));
        }
        bumpRefresh();
      } catch (e) {
        console.error('Error al actualizar la asistencia:', e);
      }
    } catch (e: any) {
      setErr(e?.response?.data?.detail || 'No se pudo iniciar reconocimiento');
    } finally {
      setLoading(false);
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

  const removeStudent = async (studentId: number) => {
    if (!selected) return;
    
    if (!window.confirm('¿Estás seguro de eliminar este estudiante del curso?')) {
      return;
    }

    try {
      await api.delete(`/courses/${selected.id}/students/${studentId}`);
      
      // Actualizar la lista de estudiantes
      const res = await api.get('/students', { 
        params: { course_id: selected.id } 
      });
      setStudents(res.data || []);
      
      setMsg('Estudiante eliminado correctamente');
    } catch (e: any) {
      setErr(e?.response?.data?.detail || 'Error al eliminar el estudiante');
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      {/* Fila superior: Mis cursos y Reconocimiento en vivo */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: '100%' }}>
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
        </Grid>

        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" mb={1}>Reconocimiento en vivo</Typography>
            <List dense sx={{ maxHeight: 300, overflow: 'auto' }}>
              {live.map(item => {
                const confidence = item.confidence ?? 0;
                const hasCode = item.codigo && item.codigo !== 'Desconocido';
                
                return (
                  <ListItem 
                    key={item.ts} 
                    sx={{ 
                      bgcolor: hasCode ? 'info.light' : 'background.paper',
                      mb: 0.5,
                      borderRadius: 1,
                      borderLeft: hasCode ? '3px solid' : 'none',
                      borderColor: hasCode ? 'info.main' : 'transparent'
                    }}
                  >
                    <ListItemText 
                      primary={
                        <Typography 
                          variant="body2" 
                          color={hasCode ? 'info.dark' : 'text.secondary'}
                          fontWeight={hasCode ? 'medium' : 'normal'}
                        >
                          {item.codigo || 'Desconocido'}
                        </Typography>
                      }
                      secondary={
                        <Typography variant="caption" color="text.secondary">
                          {`${(confidence * 100).toFixed(1)}% confianza · ${new Date(item.ts).toLocaleTimeString()}`}
                        </Typography>
                      }
                    />
                  </ListItem>
                );
              })}
              {live.length === 0 && (
                <Typography variant="body2" color="text.secondary" sx={{ p: 2, textAlign: 'center' }}>
                  {wsConnected ? 'Esperando detecciones...' : 'Conecte el WebSocket para ver reconocimiento en tiempo real'}
                </Typography>
              )}
            </List>
          </Paper>
        </Grid>
      </Grid>

      {/* Sección principal: Detalles del curso seleccionado */}
      <Paper sx={{ p: 3, mb: 2 }}>
        <Typography variant="h5" mb={2}>{selected ? selected.nombre : 'Selecciona un curso'}</Typography>
        <Divider sx={{ mb: 2 }} />
        
        {err && <Alert severity="error" sx={{ mb: 2 }}>{err}</Alert>}
        {msg && <Alert severity="success" sx={{ mb: 2 }}>{msg}</Alert>}
        {trainMsg && <Alert severity="info" sx={{ mb: 2 }}>{trainMsg}</Alert>}
        
        {/* Controles principales */}
        <Box sx={{ display: 'flex', gap: 1, mb: 3, flexWrap: 'wrap', alignItems: 'center' }}>
          <Button 
            variant="outlined" 
            onClick={() => {
              if (selected) {
                trainModel(selected.id, selected.nombre);
              } else {
                setErr('Por favor selecciona un curso primero');
              }
            }} 
            disabled={trainLoading || !selected}
          >
            {trainLoading ? 'Entrenando...' : 'Entrenar modelo'}
          </Button>
          <Button variant="outlined" onClick={connectWS} disabled={!selected || wsConnected}>
            Conectar WS
          </Button>
          <Button variant="outlined" onClick={disconnectWS} disabled={!wsConnected}>
            Desconectar WS
          </Button>
          <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
            {wsConnected ? '● WS conectado' : '○ WS desconectado'}
          </Typography>
        </Box>

        <Box sx={{ mb: 3 }}>
          <Button 
            variant="contained" 
            size="large"
            disabled={!selected || loading} 
            onClick={startRecognition}
          >
            {loading ? 'Iniciando...' : 'Iniciar Reconocimiento'}
          </Button>
        </Box>

        {/* Matricular estudiante */}
        {selected && (
          <Box sx={{ display: 'flex', gap: 1, mb: 3, alignItems: 'center' }}>
            <TextField 
              size="small" 
              label="CUI estudiante" 
              value={enrollCui} 
              onChange={e => setEnrollCui(e.target.value)} 
            />
            <Button variant="outlined" onClick={enrollByCui} disabled={!enrollCui}>
              Matricular
            </Button>
          </Box>
        )}

        {/* Lista de estudiantes en filas */}
        <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
          Estudiantes {students.length > 0 && `(${students.length})`}
        </Typography>
        
        <List dense>
          {students.map(student => {
            const studentAttendance = attendance.find(a => a.estudiante_id === student.id);
            const isPresent = studentAttendance?.estado === 'presente';
            const studentData = studentAttendance?.student || student;
            const userName = studentData.user 
              ? `${studentData.user.nombres} ${studentData.user.apellidos}`.trim()
              : 'Estudiante sin nombre';
            const fechaLabel = studentAttendance?.fecha_hora 
              ? new Date(studentAttendance.fecha_hora).toLocaleString('es-PE', { timeZone: 'America/Lima', day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })
              : 'Sin marca';

            return (
              <ListItem
                key={student.id}
                sx={{
                  border: `2px solid ${isPresent ? '#2e7d32' : '#d32f2f'}`,
                  backgroundColor: isPresent ? 'rgba(46, 125, 50, 0.06)' : 'rgba(211, 47, 47, 0.06)',
                  borderRadius: 1.5,
                  mb: 1,
                  alignItems: 'center'
                }}
                secondaryAction={
                  <IconButton edge="end" aria-label="delete" onClick={() => removeStudent(student.id)}>
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                }
              >
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
                      <Typography variant="subtitle2" sx={{ color: isPresent ? '#2e7d32' : '#d32f2f', fontWeight: 700 }}>
                        {isPresent ? 'Presente' : 'Ausente'}
                      </Typography>
                      <Typography variant="body1" fontWeight="bold">
                        {student.codigo}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {userName} · {student.carrera}
                      </Typography>
                    </Box>
                  }
                  secondary={
                    <Typography variant="caption" color="text.secondary">
                      {fechaLabel}
                    </Typography>
                  }
                />
              </ListItem>
            );
          })}
          {selected && students.length === 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
              Sin estudiantes matriculados en este curso
            </Typography>
          )}
        </List>
      </Paper>

      {/* Sección crear curso - Compacta en la parte inferior */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="subtitle2" mb={1} color="text.secondary">Crear nuevo curso</Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
          <TextField 
            size="small" 
            label="Nombre" 
            value={newName} 
            onChange={e => setNewName(e.target.value)}
            sx={{ minWidth: 200 }}
          />
          <TextField 
            size="small" 
            label="Código" 
            value={newCode} 
            onChange={e => setNewCode(e.target.value)}
            sx={{ minWidth: 120 }}
          />
          <TextField 
            size="small" 
            label="Horario" 
            value={newHorario} 
            onChange={e => setNewHorario(e.target.value)}
            sx={{ minWidth: 150 }}
          />
          <Button 
            variant="contained" 
            size="small"
            onClick={createCourse} 
            disabled={!newName || !newCode}
          >
            Crear
          </Button>
        </Box>
      </Paper>
    </Box>
  )
}
