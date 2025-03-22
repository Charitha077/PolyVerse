const express = require('express');
const mongoose = require('mongoose');
const FormData = require('../models/FormData');
const PolymerData = require('../models/PolymerData');
// const { transformInput } = require('../transformer'); // Placeholder for your transformer model
const router = express.Router();

// Register route
router.post('/register', async (req, res) => {
    const { email, password } = req.body;
    try {
        const user = await FormData.findOne({ email });
        if (user) {
            return res.status(400).json({ message: 'User already registered' });
        }
        const newUser = new FormData({ email, password });
        await newUser.save();
        res.status(201).json({ message: 'User registered successfully' });
    } catch (error) {
        res.status(500).json({ message: 'Error registering user', error });
    }
});

// Login route
router.post('/login', async (req, res) => {
    const { email, password } = req.body;
    try {
        const user = await FormData.findOne({ email });
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }
        if (user.password === password) {
            return res.status(200).json({ message: 'Login successful' });
        }
        res.status(400).json({ message: 'Incorrect password' });
    } catch (error) {
        res.status(500).json({ message: 'Error logging in', error });
    }
});

// Submit polymer structure (SMILES) and store it in the database
router.post('/submit-polymer', async (req, res) => {
    const { smiles, properties } = req.body;
    try {
        const newPolymerData = new PolymerData({ smiles, properties });
        await newPolymerData.save();
        res.status(201).json({ message: 'Polymer data saved successfully' });
    } catch (error) {
        res.status(500).json({ message: 'Error saving polymer data', error });
    }
});

// Handle prediction request based on SMILES notation
router.post('/predict', async (req, res) => {
    const { smiles } = req.body;
    try {
        // Use your transformer model to predict properties from the SMILES string
        const modelOutput = await transformInput(smiles);
        res.status(200).json(modelOutput); // Return model predictions
    } catch (error) {
        res.status(500).json({ message: 'Error predicting properties', error });
    }
});

module.exports = router;
