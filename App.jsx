const askBot = async (query) => {
  try {
    const res = await fetch("http://localhost:5000/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }

    const data = await res.json();
    console.log("Bot says:", data.answer);
    return data.answer; // Return the answer for use in components
  } catch (error) {
    console.error("Error asking bot:", error);
    return null;
  }
};