import { useState } from 'react';

const AnalysisPage = () => {
    const [polymerName, setPolymerName] = useState('');
    const [smilesRepresentation, setSmilesRepresentation] = useState('');
    const [uses, setUses] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const getAnalysis = async (name) => {
        setLoading(true);
        setError('');
        try {
            const response = await fetch('http://localhost:5000/analysis', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name }),
            });

            const data = await response.json();
            if (response.ok) {
                setSmilesRepresentation(data.smiles);
                setUses(data.uses);
            } else {
                setError(data.error);
            }
        } catch (err) {
            setError('An error occurred. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit = (event) => {
        event.preventDefault();
        if (polymerName) {
            getAnalysis(polymerName);
        }
    };

    return (
        <div className="flex justify-center items-center min-h-screen bg-white pt-20">
            <div className="bg-white p-8 rounded-lg shadow-lg w-full max-w-3xl">
                <h2 className="text-3xl font-bold text-gray-800 mb-6" style={{ fontFamily: 'Arial, sans-serif' }}>Polymer Name to SMILES & Uses</h2>
                <form onSubmit={handleSubmit}>
                    <div className="mb-4">
                        <label htmlFor="polymerName" className="block text-lg font-semibold text-gray-700 mb-2">
                            Polymer Name
                        </label>
                        <input
                            type="text"
                            id="polymerName"
                            placeholder="Enter Polymer Name"
                            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                            value={polymerName}
                            onChange={(event) => setPolymerName(event.target.value)}
                            required
                        />
                    </div>
                    <button type="submit" className="w-full py-2 px-4 bg-blue-500 text-white font-semibold rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" disabled={loading}>
                        {loading ? "Processing..." : "Get SMILES & Uses"}
                    </button>
                </form>

                {error && (
                    <div className="mt-6 text-red-600">{error}</div>
                )}

                {smilesRepresentation || uses ? (
                    <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                        {smilesRepresentation && (
                            <div className="p-4 border border-gray-300 rounded-lg shadow-lg">
                                <h3 className="text-xl font-semibold text-gray-800">SMILES Representation:</h3>
                                <p className="text-lg text-gray-700 mt-2">{smilesRepresentation}</p>
                            </div>
                        )}
                        {uses && (
                            <div className="p-4 border border-gray-300 rounded-lg shadow-lg">
                                <h3 className="text-xl font-semibold text-gray-800">Uses:</h3>
                                <p className="text-lg text-gray-700 mt-2">{uses}</p>
                            </div>
                        )}
                    </div>
                ) : null}
            </div>
        </div>
    );
};

export default AnalysisPage;
