import React, { useState } from 'react';
import { FileText, RefreshCw } from 'lucide-react';

const INPUT_TYPES = {
  NAME: 'name',
  SMILES: 'smiles',
};

export default function PolymerInput({ onSubmit, isLoading }) {
  const [inputType, setInputType] = useState(INPUT_TYPES.NAME);
  const [inputValue, setInputValue] = useState('');

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim()) {
      const data = inputType === INPUT_TYPES.NAME ? { name: inputValue } : { smiles: inputValue };
      onSubmit(data);
    }
  };

  return (
    <div className="w-full max-w-2xl bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center gap-2 mb-4">
        <FileText className="w-5 h-5 text-blue-600" />
        <h2 className="text-xl font-semibold">Polymer Input</h2>
      </div>

      <div className="flex items-center mb-4">
        <label htmlFor="inputType" className="block text-sm font-medium text-gray-700 mr-2">
          Input Type:
        </label>
        <div className="flex items-center">
          <input
            type="radio"
            id="nameRadio"
            name="inputType"
            value={INPUT_TYPES.NAME}
            checked={inputType === INPUT_TYPES.NAME}
            onChange={() => setInputType(INPUT_TYPES.NAME)}
            className="w-4 h-4 border-gray-300 rounded-sm focus:ring-0 focus:ring-offset-0 focus:outline-none focus:ring-blue-500"
          />
          <label htmlFor="nameRadio" className="ml-2">
            Polymer Name
          </label>
          <input
            type="radio"
            id="smilesRadio"
            name="inputType"
            value={INPUT_TYPES.SMILES}
            checked={inputType === INPUT_TYPES.SMILES}
            onChange={() => setInputType(INPUT_TYPES.SMILES)}
             className="w-4 h-4 border-gray-300 rounded-sm focus:ring-0 focus:ring-offset-0 focus:outline-none focus:ring-blue-500 ml-4"
          />
          <label htmlFor="smilesRadio" className="ml-2 text-gray-700">
            SMILES
          </label>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {inputType === INPUT_TYPES.NAME ? 'Polymer Name' : 'SMILES String'}
          </label>
          <input
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            className="w-full px-4 py-2 border rounded-md"
            placeholder={
              inputType === INPUT_TYPES.NAME
                ? 'Enter polymer name (e.g., Polyethylene)'
                : 'Enter SMILES string (e.g., O=C[C@H](F)C=O)'
            }
            disabled={isLoading}
          />
        </div>

        <button
          type="submit"
          disabled={isLoading || !inputValue.trim()}
          className="w-full flex items-center justify-center gap-2 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <RefreshCw className="w-5 h-5 animate-spin" />
          ) : (
            'Predict Properties'
          )}
        </button>
      </form>
    </div>
  );
}