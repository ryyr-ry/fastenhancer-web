import { Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { FramePage } from './pages/FramePage'
import { HomePage } from './pages/HomePage'
import { ReactPage } from './pages/ReactPage'
import { VanillaPage } from './pages/VanillaPage'

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="/react" element={<ReactPage />} />
        <Route path="/vanilla" element={<VanillaPage />} />
        <Route path="/frame" element={<FramePage />} />
      </Route>
      <Route path="*" element={<Navigate replace to="/" />} />
    </Routes>
  )
}

export default App
