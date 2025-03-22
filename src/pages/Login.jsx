import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import backgroundImage from './b.png';

const Login = ({ setIsLoggedIn }) => {
    const [isLogin, setIsLogin] = useState(true); // Toggle between Login and Register forms
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const handleSubmit = (event) => {
        event.preventDefault();
        setLoading(true);

        if (!email || !password || (!isLogin && !name)) {
            alert('Please fill in all fields');
            setLoading(false);
            return;
        }

        const url = isLogin ? 'http://localhost:3001/login' : 'http://localhost:3001/register';
        const data = isLogin ? { email, password } : { name, email, password };

        axios.post(url, data)
            .then(result => {
                setLoading(false);

                if (result.data.status === "success") {
                    if (isLogin) {
                        alert('Login successful!');
                        localStorage.setItem('authToken', result.data.token);
                        setIsLoggedIn(true);
                        navigate('/predict');
                    } else {
                        alert('Registration successful! Please log in.');
                        setIsLogin(true);  // Switch to login form after successful registration
                    }
                } else if (result.data.status === "error") {
                    alert(result.data.message); // Display the message returned from the backend
                } else {
                    alert('Something went wrong, please try again later.');
                }
            })
            .catch(err => {
                console.error("Error during submit:", err);
                setLoading(false);
                alert('There was an error. Please try again later.');
            });
    };

    return (
        <div className="flex items-center justify-center h-screen mt-16"
       //bg-gradient-to-r from-blue-50 to-blue-300 
       style={{
                 backgroundImage: `url('${backgroundImage}')`,
                 backgroundSize: 'cover',
                 backgroundPosition: 'center',
                 backgroundAttachment: 'fixed',
      }}
       >
            <div className="bg-white shadow-lg rounded-lg p-6 max-w-md w-full">
                <h2 className="text-2xl font-bold text-center text-gray-800 mb-4">
                    {isLogin ? 'Login' : 'Register'}
                </h2>
                <form onSubmit={handleSubmit} className="space-y-4">
                    {!isLogin && (
                        <div>
                            <label className="block text-gray-700 font-semibold" htmlFor="name">Name</label>
                            <input
                                type="text"
                                id="name"
                                className="w-full mt-1 p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-purple-500 focus:outline-none"
                                placeholder="Enter your name"
                                onChange={(event) => setName(event.target.value)}
                                required
                            />
                        </div>
                    )}
                    <div>
                        <label className="block text-gray-700 font-semibold" htmlFor="email">Email</label>
                        <input
                            type="email"
                            id="email"
                            className="w-full mt-1 p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-purple-500 focus:outline-none"
                            placeholder="Enter your email"
                            onChange={(event) => setEmail(event.target.value)}
                            required
                        />
                    </div>
                    <div>
                        <label className="block text-gray-700 font-semibold" htmlFor="password">Password</label>
                        <input
                            type="password"
                            id="password"
                            className="w-full mt-1 p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-purple-500 focus:outline-none"
                            placeholder="Enter your password"
                            onChange={(event) => setPassword(event.target.value)}
                            required
                        />
                    </div>
                    <button
                        type="submit"
                        className={`w-full py-2 rounded-md text-white ${loading ? 'bg-blue-500' : 'bg-blue-600 hover:bg-blue-700'} transition duration-300`}
                        disabled={loading}
                    >
                        {loading ? `${isLogin ? "Logging in..." : "Registering..."}` : `${isLogin ? "Login" : "Register"}`}
                    </button>
                </form>

                <p className="text-center text-sm text-gray-600 mt-4">
                    {isLogin ? 'Don’t have an account?' : 'Already have an account?'}{' '}
                    <button
                        onClick={() => setIsLogin(!isLogin)}
                        className="text-blue-500 hover:underline"
                    >
                        {isLogin ? 'Register' : 'Login'}
                    </button>
                </p>
            </div>
        </div>
    );
};

export default Login;
