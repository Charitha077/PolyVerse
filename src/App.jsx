import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import HomePage from './pages/HomePage'; // Correct path
import PredictPage from './pages/PredictPage'; // Correct path
import Navbar from './components/layout/Navbar'; // Correct path
import AnalysisPage from './pages/AnalysisPage'; // Correct path
import Login from './pages/Login'; // Correct path
import About from './pages/About'; // Correct path
import Team from './pages/Team'; // Correct path
import Contact from './pages/Contact'; // Correct path
import PropertyDisplay from './components/PropertyDisplay'; // Correct path
import './index.css'; // Ensure Tailwind CSS is applied properly

function Footer() {
  return (
    <footer className="w-full py-4 sm:py-6 bg-gray-800 text-white text-center text-sm sm:text-base mt-auto">
      <p>&copy; 2024 PolyVerse. All rights reserved.</p>
    </footer>
  );
}

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(localStorage.getItem('authToken') ? true : false);

  useEffect(() => {
    setIsLoggedIn(!!localStorage.getItem('authToken'));
  }, []);

  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col">
        <Navbar isLoggedIn={isLoggedIn} setIsLoggedIn={setIsLoggedIn} /> {/* Pass props to Navbar */}
        <div className="flex-grow">
          <Routes>
            <Route path="/login" element={<Login setIsLoggedIn={setIsLoggedIn} />} /> {/* Pass setter to Login */}
            <Route path="/predict" element={isLoggedIn ? <PredictPage /> : <Navigate to="/login" />} />
            <Route path="/polymer-display" element={<PropertyDisplay />} />
            <Route path="/analysis" element={<AnalysisPage />} />
            <Route path="/about" element={<About />} />
            <Route path="/team" element={<Team />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/" element={<HomePage />} />
          </Routes>
        </div>
        <Footer /> {/* Persistent Footer */}
      </div>
    </BrowserRouter>
  );
}

export default App;
