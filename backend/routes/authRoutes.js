const express = require('express');
const router = express.Router();
const FormDataModel = require('../models/FormData');

// Register a new user
router.post('/register', (req, res) => {
  const { name, email, password } = req.body;
  FormDataModel.findOne({ email })
    .then(user => {
      if (user) {
        res.json("User already registered");
      } else {
        const newUser = new FormDataModel({
          name,
          email,
          password
        });

        newUser.save()
          .then((user) => res.json(user))
          .catch((err) => res.status(400).json({ message: err.message }));
      }
    })
    .catch(err => res.status(500).json({ message: err.message }));
});

// Login a user
router.post('/login', (req, res) => {
  const { email, password } = req.body;
  FormDataModel.findOne({ email })
    .then(user => {
      if (user) {
        if (user.password === password) {
          res.json("Success");
        } else {
          res.json("Wrong password");
        }
      } else {
        res.json("User not found");
      }
    })
    .catch(err => res.status(500).json({ message: err.message }));
});

module.exports = router;
