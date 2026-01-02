import React, { useEffect, useMemo, useRef, useState } from 'react'
import api from '../api/client'
import DeleteIcon from '@mui/icons-material/Delete';
import { Course, Student, Attendance } from '../types'
import { Box, Paper, Typography, List, ListItem, ListItemButton, ListItemText, Divider, Button, Alert, TextField, IconButton } from '@mui/material'

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
        const data = JSON.parse(ev.data)
        const ts = Date.now()
        
        // SOLO actualizar la lista de reconocimientos en tiempo real
        // NO actualizar lastRecognition de estudiantes (eso solo lo hace Iniciar Reconocimiento)
        setLive(prev => {
          const newLive = [{ ...data, ts }, ...prev].slice(0, 20)
          return newLive
        })
      } catch (e) {
        console.error('Error procesando mensaje WebSocket:', e)
      }
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
  }, [selected, msg])

  const startRecognition = async () => {
    if (!selected) return;
    setLoading(true);
    setMsg(null);
    setErr(null);
    
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
      
      setMsg(
        `Reconocimiento completado. ` +
        `Presentes: ${summary.present || 0}, ` +
        `Ausentes: ${summary.absent || 0}`
      );
      
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
        <Typography variant="h6" mb={1}>Reconocimiento en vivo (stream) - Solo visualización</Typography>
        <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
          Este panel muestra reconocimiento en tiempo real sin marcar asistencia. 
          Use "Iniciar Reconocimiento" para el proceso oficial.
        </Typography>
        <List dense>
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

      <Paper sx={{ p: 2, flex: 1 }}>
        <Typography variant="h6" mb={1}>{selected ? selected.nombre : 'Selecciona un curso'}</Typography>
        <Divider sx={{ mb: 2 }} />
        {err && <Alert severity="error" sx={{ mb: 1 }}>{err}</Alert>}
        {msg && <Alert severity="success" sx={{ mb: 1 }}>{msg}</Alert>}
        {trainMsg && <Alert severity="info" sx={{ mb: 1 }}>{trainMsg}</Alert>}
        <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
          <Button variant="outlined" onClick={() => {
            if (selected) {
              trainModel(selected.id, selected.nombre);
            } else {
              setErr('Por favor selecciona un curso primero');
            }
          }} disabled={trainLoading || !selected}>{trainLoading ? 'Entrenando...' : 'Entrenar modelo'}</Button>
          <Button variant="outlined" onClick={connectWS} disabled={!selected || wsConnected}>Conectar WS</Button>
          <Button variant="outlined" onClick={disconnectWS} disabled={!wsConnected}>Desconectar WS</Button>
          <Typography variant="body2" color="text.secondary">{wsConnected ? 'WS conectado' : 'WS desconectado'}</Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
          <Button variant="contained" disabled={!selected || loading} onClick={startRecognition}>
            {loading ? 'Iniciando...' : 'Iniciar Reconocimiento'}
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
          {students.map(student => {
            // Buscar asistencia del estudiante para hoy (SOLO de Iniciar Reconocimiento)
            const studentAttendance = attendance.find(a => a.estudiante_id === student.id);
            const confidence = studentAttendance?.confidence || 0; // Porcentaje de coincidencia
            const meetsThreshold = confidence >= 70; // Umbral del 70%
            
            // Determinar estado basado en el umbral
            const isPresent = studentAttendance && meetsThreshold;
            const isAbsent = studentAttendance && !meetsThreshold;
            
            // Los colores verde/rojo SOLO se muestran si hay registro de asistencia oficial
            // (no del WebSocket de prueba)

            return (
              <ListItem 
                key={student.id}
                sx={{
                  bgcolor: isPresent 
                    ? 'success.light' 
                    : (isAbsent 
                        ? 'error.light' 
                        : 'background.paper'),
                  mb: 0.5,
                  borderRadius: 1,
                  transition: 'background-color 0.3s ease',
                  borderLeft: theme => `4px solid ${
                    isPresent 
                      ? theme.palette.success.main 
                      : (isAbsent
                          ? theme.palette.error.main
                          : theme.palette.divider)
                  }`
                }}
              >
                <Box sx={{ 
                  width: '100%', 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center' 
                }}>
                  <Box sx={{ flex: 1 }}>
                    <Typography 
                      variant="body2" 
                      color={isPresent 
                        ? 'success.dark' 
                        : (isAbsent
                            ? 'error.dark'
                            : 'text.primary')}
                      fontWeight={(isPresent || isAbsent) ? 'bold' : 'normal'}
                      component="div"
                      sx={{ mb: 0.5 }}
                    >
                      {student.nombre || student.codigo}
                    </Typography>
                    <Box 
                      component="div" 
                      sx={{ 
                        display: 'flex', 
                        flexDirection: 'column',
                        gap: 0.5,
                        typography: 'caption',
                        color: 'text.secondary'
                      }}
                    >
                      <span>{student.codigo} {student.carrera ? `· ${student.carrera}` : ''}</span>
                      {studentAttendance ? (
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
                          <span style={{ 
                            fontWeight: 'bold',
                            color: isPresent ? '#2e7d32' : '#d32f2f'
                          }}>
                            {isPresent 
                              ? `🟩 Presente (${Math.round(confidence)}%)` 
                              : `🟥 Ausente (${Math.round(confidence)}%)`}
                          </span>
                          {studentAttendance.fecha_hora && isPresent && (
                            <span>
                              Fecha: {new Date(studentAttendance.fecha_hora).toLocaleDateString('es-ES', {
                                year: 'numeric',
                                month: '2-digit',
                                day: '2-digit'
                              })}
                              {' · '}
                              Hora: {new Date(studentAttendance.fecha_hora).toLocaleTimeString('es-ES', {
                                hour: '2-digit',
                                minute: '2-digit',
                                second: '2-digit'
                              })}
                            </span>
                          )}
                          {isAbsent && (
                            <span style={{ fontStyle: 'italic' }}>
                              No asistió
                            </span>
                          )}
                        </Box>
                      ) : (
                        <span style={{ fontStyle: 'italic', color: 'text.disabled' }}>
                          Sin registro de asistencia
                        </span>
                      )}
                    </Box>
                  </Box>
                  <IconButton 
                    color="error" 
                    onClick={(e) => {
                      e.stopPropagation();
                      removeStudent(student.id);
                    }}
                    title="Eliminar del curso"
                    size="small"
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </Box>
              </ListItem>
            );
          })}
          {selected && students.length === 0 && (
            <Typography variant="body2" color="text.secondary">
              Sin estudiantes matriculados en este curso
            </Typography>
          )}
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
