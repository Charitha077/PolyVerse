import React from 'react';
import bg from './a.jpg';

export default function Contact() {
  const onSubmit = async (event) => {
    event.preventDefault();

    const formData = new FormData(event.target);

    // Add required Web3Forms fields
    formData.append("access_key", "5eeb395b-9148-4528-acc7-959b30b793ca");
    formData.append("debug_mode", "true"); // Enable debugging for Web3Forms

    const object = Object.fromEntries(formData);
    const json = JSON.stringify(object);

    try {
      const res = await fetch("https://api.web3forms.com/submit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: json,
      });

      const result = await res.json();
      console.log("Response:", result);

      if (result.success) {
        alert("Form submitted successfully!");
        event.target.reset(); // Clear the form
      } else {
        alert(`Error: ${result.message || "Something went wrong"}`);
      }
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred. Please try again.");
    }
  };

  return (
    <div
      className="min-h-screen flex flex-col bg-cover bg-center"
      style={{
        backgroundImage: `url('${bg}')`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundAttachment: 'fixed',
      }}
    >
      <header className="text-center mb-10 pt-12 mt-12"> {/* Added mt-24 for margin-top */}
        <h1 className="text-5xl font-bold text-gray-800 mb-4">Contact Us</h1>
        <p className="text-2xl text-gray-600 ">
          We'd love to hear from you. Reach out with any questions or feedback.
        </p>
      </header>

      <div className="flex-grow flex justify-center items-start px-6 pb-16 "> {/* Added pb-16 */}
        <form 
         onSubmit={onSubmit}
         className="max-w-lg mx-auto bg-white p-6 rounded-lg shadow-lg">
          <input
            type="text"
            name="name"  
            placeholder="Your Name"
            className="w-full border border-gray-300 rounded-md p-2 mb-4 text-lg"
          />
          <input
            type="email"
            name="email"  
            placeholder="Your Email"
            className="w-full border border-gray-300 rounded-md p-2 mb-4 text-lg"
          />
          <textarea
            name="message"  
            placeholder="Your Message"
            className="w-full border border-gray-300 rounded-md p-2 mb-4 text-lg"
            rows="5"
          />
          <button
            type="submit"
            className="bg-blue-600 text-white py-2 px-4 rounded-md text-lg"
          >
            Submit
          </button>
        </form>
      </div>
    </div>
  );
}
