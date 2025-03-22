import React from 'react';
import bg from './a.jpg';

export default function About() {
  const aboutCards = [
    {
      title: 'The Challenge in Polymer Research',
      description: 'Polymer materials exhibit diverse properties that vary with processing and environmental conditions, making accurate prediction challenging.',
    },
    {
      title: 'Polymer Tokenizer with Chemical Awareness',
      description: 'This project develops a chemical-aware polymer tokenizer to improve polymer property prediction models and material design.',
    },
    {
      title: 'Transformers in Polymer Research',
      description: 'Using transformer models, originally designed for NLP, we capture complex relationships between polymer structures and properties to make accurate predictions.',
    },
    {
      title: 'Structure-Property Relationship Analysis',
      description: 'With transformer models, we analyze how polymer structures influence their properties, allowing faster material discovery and prediction.',
    },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-cover bg-center"
     style={{
              backgroundImage: `url('${bg}')`,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
              backgroundAttachment: 'fixed',
            }}>
      <header className="text-center mb-8 pt-14  mt-14"> {/* Add mt-24 for margin-top */}
        <h1 className="text-4xl font-bold text-gray-800 mb-4">About PolyVerse</h1>
        <p className="text-xl text-gray-800">
          Discover how PolyVerse revolutionizes polymer research with cutting-edge machine learning solutions.
        </p>
      </header>

      <div className="flex-grow flex justify-center items-start px-6 pd-10 mb-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-8">
          {aboutCards.map((card, index) => (
            <div key={index} className="hover:shadow-xl transition-all  text-center flex flex-col items-center  hover:scale-105 hover:bg-gradient-to-r hover:from-blue-50 hover:to-blue-300 p-6  shadow-lg">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">{card.title}</h3>
              <p className="text-gray-600">{card.description}</p>
            </div>
          ))}
        </div>
      </div>

      
    </div>
  );
}