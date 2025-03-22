// PolymerData object structure
const PolymerData = {
  smiles: '', // string representing the polymer's structure
  properties: {
    mechanical: {
      youngsModulus: undefined, // number (optional)
      tensileStrength: undefined, // number (optional)
      elongation: undefined // number (optional)
    },
    thermal: {
      glassTempC: undefined, // number (optional)
      meltTempC: undefined, // number (optional)
      thermalConductivity: undefined // number (optional)
    },
    optical: {
      refractiveIndex: undefined, // number (optional)
      transparency: undefined // number (optional)
    }
  }
};

// PredictionResult object structure
const PredictionResult = {
  property: '', // string representing the name of the property
  value: 0, // number representing the predicted value
  confidence: 0, // number representing the confidence in the prediction
  unit: '' // string representing the unit of the prediction value
};
