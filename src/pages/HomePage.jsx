import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { TestTube, LineChart } from 'lucide-react';
import backgroundImage from './b.png';

const features = (isLoggedIn) => [
  {
    icon: TestTube,
    title: 'Property Prediction',
    description: 'Predict polymer properties using advanced ML models with chemical awareness.',
    link: isLoggedIn ? '/predict' : '/login',
  },
  {
    icon: LineChart,
    title: 'SMILES Converter and Applications',
    description: 'Transform molecular structures with SMILES and explore its use.',
    link:'/analysis',
  },
];

export default function HomePage() {
  const navigate = useNavigate();
  const [isLoggedIn, setIsLoggedIn] = useState(!!localStorage.getItem('authToken'));

  useEffect(() => {
    setIsLoggedIn(!!localStorage.getItem('authToken'));
  }, []);

  const handleHomeClick = () => {
    navigate(0); // This will force a re-render by reloading the page
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Main content section */}
      <div
        className="flex-grow py-12 px-6 flex flex-col items-center"
        style={{
          backgroundImage: `url('${backgroundImage}')`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundAttachment: 'fixed',
        }}
      >
        <header className="text-center mb-16 max-w-2xl" style={{ marginTop: '80px' }}>
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            <button onClick={handleHomeClick} className="text-4xl font-bold text-gray-800 mb-4">
              Welcome to PolyVerse
            </button>
          </h1>
          <p className="text-xl text-gray-600">
            Advanced polymer property prediction and analysis using machine learning
          </p>
        </header>

        <div className="flex flex-wrap justify-center gap-8 max-w-screen-lg">
          {features(isLoggedIn).map(({ icon: Icon, title, description, link }) => (
            <Link
              key={title}
              to={link}
              className="bg-transparent border border-gray-300 rounded-lg shadow-lg hover:shadow-xl transition-all p-6 text-center flex flex-col items-center w-full sm:w-1/2 lg:w-1/3 hover:scale-105 hover:bg-gradient-to-r hover:from-blue-50 hover:to-blue-300"
            >
              <Icon className="w-12 h-12 text-blue-600 mb-4" />
              <h3 className="text-xl font-semibold text-gray-800 mb-2">
                {title}
              </h3>
              <p className="text-gray-600">{description}</p>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
