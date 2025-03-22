import React from 'react';
import { Activity } from 'lucide-react';

export default function PropertyDisplay({ predictions }) {
  if (!predictions || Object.keys(predictions).length === 0) {
    return (
      <div className="w-full max-w-2xl bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center gap-2 mb-6">
          <Activity className="w-5 h-5 text-gray-600" />
          <h2 className="text-xl font-semibold text-gray-800">No Predictions Available</h2>
        </div>
        <p className="text-gray-500">There are no predicted properties to display. Please ensure data is provided and try again.</p>
      </div>
    );
  }

  return (
    <div className="w-full max-w-2xl bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center gap-2 mb-6">
        <Activity className="w-5 h-5 text-green-600" />
        <h2 className="text-xl font-semibold text-gray-800">Predicted Properties</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {Object.entries(predictions).map(([property, value], index) => (
          <div
            key={index}
            className="p-4 border border-gray-200 rounded-lg hover:shadow-md transition-shadow"
          >
            <h3 className="text-sm font-medium text-gray-500">{property}</h3>
            <div className="mt-1 flex items-end gap-2">
              <span className="text-2xl font-bold">{value}</span>
              <span className="text-sm text-gray-500">
                {getUnitForProperty(property)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

const getUnitForProperty = (property) => {
  const propertyUnits = {
    'Tensile_Strength(Mpa)': 'MPa',
    'Ionisation_Energy(eV)': 'eV',
    'Electron_Affinity(eV)': 'eV',
    'LogP': '',
    'Refractive_Index': '',
    'Molecular_Weight(g/mol)': 'g/mol',
  };

  return propertyUnits[property] || '';
};
