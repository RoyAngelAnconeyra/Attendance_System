export type Role = 'admin' | 'docente' | 'estudiante'

export type Course = {
  id: number
  docente_id: number
  nombre: string
  codigo: string
  horario?: string
}

export interface User {
  id: number;
  nombres: string;
  apellidos: string;
  email: string;
  rol: string;
}

export type Student = {
  id: number
  user_id: number
  codigo: string
  nombre?: string
  carrera?: string
  user?: {
    nombres: string;
    apellidos: string;
    email: string;
  };
  uses_glasses: boolean
  uses_cap: boolean
  uses_mask: boolean
  lastRecognition?: {
    ts: number
    confidence: number
    recognized: boolean
  }
}

export type Attendance = {
  id: number
  estudiante_id: number
  curso_id: number
  estado: 'presente' | 'ausente'
  fecha_hora?: string
  student?: Student
}
