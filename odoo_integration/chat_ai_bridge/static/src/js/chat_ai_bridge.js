/** @odoo-module **/

import { rpc } from "@web/core/network/rpc_service";

const callChatAI = (prompt) =>
    rpc("/chat_ai_bridge/ask", {
        prompt,
    });

const setupChatAIPage = () => {
    const container = document.querySelector(".o_chat_ai_bridge");
    if (!container) {
        return;
    }

    const promptInput = container.querySelector(".o_chat_ai_bridge__prompt");
    const submitButton = container.querySelector(".o_chat_ai_bridge__submit");
    const status = container.querySelector(".o_chat_ai_bridge__status");
    const output = container.querySelector(".o_chat_ai_bridge__output");

    const setStatus = (message, isError = false) => {
        status.textContent = message;
        status.classList.toggle("text-danger", isError);
    };

    const handleSubmit = async () => {
        const prompt = (promptInput.value || "").trim();
        if (!prompt) {
            setStatus("Please enter a prompt.", true);
            return;
        }

        setStatus("Sending request...");
        output.textContent = "";

        try {
            const result = await callChatAI(prompt);
            if (result.error) {
                setStatus(result.error, true);
                return;
            }
            setStatus("Response received.");
            output.textContent = JSON.stringify(result.data, null, 2);
        } catch (error) {
            setStatus(`Request failed: ${error}`, true);
        }
    };

    submitButton.addEventListener("click", handleSubmit);
    promptInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
            handleSubmit();
        }
    });
};

document.addEventListener("DOMContentLoaded", setupChatAIPage);
