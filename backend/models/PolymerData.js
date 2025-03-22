const mongoose = require('mongoose');

const PolymerDataSchema = new mongoose.Schema({
    smiles: { type: String, required: true },
    properties: {
        mechanical: {
            youngsModulus: { type: Number, default: null },
            tensileStrength: { type: Number, default: null },
            elongation: { type: Number, default: null }
        },
        thermal: {
            glassTempC: { type: Number, default: null },
            meltTempC: { type: Number, default: null },
            thermalConductivity: { type: Number, default: null }
        },
        optical: {
            refractiveIndex: { type: Number, default: null },
            transparency: { type: Number, default: null }
        }
    }
});

const PolymerData = mongoose.model('PolymerData', PolymerDataSchema);
module.exports = PolymerData;
