const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const polymerRoutes = require('./routes/polymerRoutes');

// Import the FormDataModel (or adjust if the model name is different)
const FormDataModel = require('./models/FormData');

// Initialize the Express app
const app = express();

// Middleware
app.use(express.json());
app.use(cors());

// MongoDB Connection URL
const mongoURI = 'mongodb://127.0.0.1:27017/polymerML'; // Adjust this if needed
mongoose.connect(mongoURI, {
    useNewUrlParser: true,
    useUnifiedTopology: true
})
.then(() => {
    console.log('MongoDB connected successfully');
})
.catch((err) => {
    console.error('Error connecting to MongoDB:', err);
    process.exit(1); // Exit the process if the database connection fails
});

// Routes for categories, posts, and image uploads (if needed)
app.use('/api', polymerRoutes);

// User authentication routes (register & login)
app.post('/register', (req, res) => {
    console.log('Register Request:', req.body); // Log the request body
    const { name, email, password } = req.body;

    // Check if user is already registered
    FormDataModel.findOne({ email: email })
    .then(user => {
        if (user) {
            return res.json({
                status: "error",
                message: "User already exists!"
            });
        } else {
            // Register new user
            const newUser = new FormDataModel({
                name,
                email,
                password
            });

            newUser.save()
            .then(log_reg_form => {
                console.log("User registered:", log_reg_form);
                res.json({
                    status: "success",
                    message: "Registered successfully!",
                    token: "your_generated_token_here" // Replace with actual token logic
                });
            })
            .catch(err => {
                console.log("Error during registration:", err);
                res.json({
                    status: "error",
                    message: "Error during registration"
                });
            });
        }
    })
    .catch(err => console.log(err));
});

app.post('/login', (req, res) => {
    console.log('Login Request:', req.body); // Log the request body
    const { email, password } = req.body;

    // Find user by email
    FormDataModel.findOne({ email: email })
    .then(user => {
        if (user) {
            // Check if password matches
            if (user.password === password) {
                res.json({
                    status: "success",
                    message: "Login successful",
                    token: "your_generated_token_here" // Replace with actual token logic
                });
            } else {
                res.json({
                    status: "error",
                    message: "Wrong password"
                });
            }
        } else {
            res.json({
                status: "error",
                message: "No records found!"
            });
        }
    })
    .catch(err => {
        console.log("Error during login:", err);
        res.json({
            status: "error",
            message: "Error during login"
        });
    });
});

// Start the server
const port = 3001;
app.listen(port, () => {
    console.log(`Server running on http://127.0.0.1:${port}`);
});
