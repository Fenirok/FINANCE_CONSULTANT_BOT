const btn = document.getElementById("btn");
const input = document.getElementById("input");
const chatLog = document.getElementById("chat-log");

btn.addEventListener("click", async () => {
    const userInput = input.value;
    if (userInput === "") return;

    // Send the user input to your chatbot API
    const response = await fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: userInput })
    });

    const data = await response.json();
    if (data.message) {
        const answer = document.createElement("div");
        answer.innerHTML = data.message;
        answer.classList.add("box", "answer");
        chatLog.appendChild(answer);
    }

    input.value = "";
});