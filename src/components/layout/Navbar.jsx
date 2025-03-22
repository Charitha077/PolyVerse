import React, { useState } from 'react';
import { Beaker, Home, Database, LineChart, TestTube, Menu, X } from 'lucide-react';
import { Link, useLocation, useNavigate } from 'react-router-dom';

const navItems = [
  { path: '/', label: 'Home', icon: Home },
  { path: '/about', label: 'About', icon: TestTube },
  { path: '/contact', label: 'Contact', icon: Database },
  { path: '/team', label: 'Team', icon: LineChart },
];

export default function Navbar({ isLoggedIn, setIsLoggedIn }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem('authToken');
    setIsLoggedIn(false);
    navigate('/login');
  };

  return (
    <nav className="fixed top-0 left-0 w-full bg-blue-300 shadow-md z-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 sm:h-20">
          <Link to="/" className="flex-shrink-0 flex items-center gap-4">
            <Beaker className="w-8 h-8 sm:w-10 sm:h-10 text-blue-600" />
            <span className="text-xl sm:text-3xl font-bold text-gray-900">PolyVerse</span>
          </Link>

          {!menuOpen && (
            <button
              className="sm:hidden p-2 rounded-md text-gray-500 hover:text-gray-700 hover:bg-gray-100"
              onClick={() => setMenuOpen(true)}
            >
              <Menu className="w-6 h-6" />
            </button>
          )}

          <div className={`${menuOpen ? 'block' : 'hidden'} sm:flex sm:space-x-8 text-lg`}>
            {menuOpen && (
              <div className="absolute top-16 right-4 w-64 bg-white shadow-lg rounded-lg sm:hidden z-20">
                <button
                  className="absolute top-2 right-2 p-1 text-gray-600"
                  onClick={() => setMenuOpen(false)}
                >
                  <X className="w-6 h-6" />
                </button>
                <div className="pt-8">
                  {navItems.map(({ path, label, icon: Icon }) => (
                    <Link
                      key={path}
                      to={path}
                      className={`block px-4 py-2 border-b text-gray-700 font-medium ${
                        location.pathname === path ? 'text-blue-700 border-b-2 border-blue-700' : ''
                      }`}
                      onClick={() => setMenuOpen(false)}
                    >
                      <Icon className="w-5 h-5 inline-block mr-2" />
                      {label}
                    </Link>
                  ))}
                  {isLoggedIn ? (
                    <button
                      className="block px-4 py-2 bg-red-600 text-white font-semibold rounded"
                      onClick={() => {
                        handleLogout();
                        setMenuOpen(false);
                      }}
                    >
                      Logout
                    </button>
                  ) : (
                    <Link
                      to="/login"
                      className="block px-4 py-2 bg-blue-400 text-white font-semibold rounded"
                      onClick={() => setMenuOpen(false)}
                    >
                      Get Started
                    </Link>
                  )}
                </div>
              </div>
            )}

            <div className="hidden sm:flex sm:space-x-8">
              {navItems.map(({ path, label, icon: Icon }) => (
                <Link
                  key={path}
                  to={path}
                  className={`block sm:inline-flex items-center text-gray-700 font-medium ${
                    location.pathname === path ? 'text-blue-700 border-b-2 border-blue-700' : ''
                  }`}
                >
                  <Icon className="w-5 h-5 sm:w-6 sm:h-6 mr-2" />
                  {label}
                </Link>
              ))}
              {isLoggedIn ? (
                <button
                  className="block sm:inline-flex items-center px-6 py-2 bg-blue-600 text-white rounded-full"
                  onClick={handleLogout}
                >
                  Logout
                </button>
              ) : (
                <Link
                  to="/login"
                  className="block sm:inline-flex items-center px-6 py-2 bg-blue-600 text-white rounded-full"
                >
                  Get Started
                </Link>
              )}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}
