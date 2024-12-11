import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";

function App() {
    const [messages, setMessages] = useState<string[]>([]);
    const [input, setInput] = useState("");

    async function greet() {
        // Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
        setMessages([...messages, await invoke("greet", { name: input })]);
    }

    return (
        <div className="flex flex-col h-screen bg-gray-100">
            {/* Scrollable text region */}
            <div className="flex-1 overflow-y-auto p-4 bg-white shadow-inner">
                {messages.length > 0 ? (
                    messages.map((msg, index) => (
                        <p key={index} className="mb-2 text-gray-800">
                            {msg}
                        </p>
                    ))
                ) : (
                    <p className="text-gray-400">
                        No messages yet. Start typing below!
                    </p>
                )}
            </div>

            {/* Input field with button */}
            <div className="p-4 bg-gray-200 border-t border-gray-300">
                <div className="flex items-center gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Type your message..."
                        className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring focus:ring-blue-300"
                    />
                    <button
                        onClick={greet}
                        className="px-4 py-2 text-black bg-white rounded-lg hover:bg-gray-300 focus:ring focus:ring-gray-600"
                    >
                        Send
                    </button>
                </div>
            </div>
        </div>
    );
}

export default App;
