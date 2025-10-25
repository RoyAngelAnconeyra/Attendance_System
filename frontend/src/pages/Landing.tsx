import React from 'react'
import { Link } from 'react-router-dom'

export default function Landing() {
  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: '48px 20px' }}>
      <h1 style={{ fontSize: 32, marginBottom: 8 }}>Sistema de Asistencia Universitaria</h1>
      <p style={{ color: '#555', marginBottom: 24 }}>Reconocimiento facial en tiempo real con MTCNN + FaceNet + SVM</p>

      <div style={{ display: 'flex', gap: 16, marginBottom: 40 }}>
        <Link to="/student" style={{ padding: '14px 18px', border: '1px solid #ddd', borderRadius: 8, textDecoration: 'none' }}>Soy estudiante</Link>
        <Link to="/prof" style={{ padding: '14px 18px', border: '1px solid #ddd', borderRadius: 8, textDecoration: 'none' }}>Soy profesor</Link>
      </div>

      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
        <Card title="Inicio de sesión" desc="Accede como docente o estudiante" />
        <Card title="Gestión de cursos" desc="Docentes ven y administran sus cursos" />
        <Card title="Asistencia en vivo" desc="Reconocimiento facial en tiempo real" />
        <Card title="Perfil estudiante" desc="Actualizar fotos con lentes/gorra/mascarilla" />
      </section>
    </div>
  )
}

function Card({ title, desc }: { title: string; desc: string }) {
  return (
    <div style={{ border: '1px solid #eee', borderRadius: 8, padding: 16 }}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      <p style={{ color: '#666' }}>{desc}</p>
    </div>
  )
}
