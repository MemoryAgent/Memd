import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";
import DarkSVG from "./dark.svg";
import LightSVG from "./light.svg";
import { ScrollArea } from "./scroll-area";

function TopBar({
    onClear,
    theme,
    setTheme,
}: {
    onClear: () => void;
    theme: string;
    setTheme: (theme: string) => void;
}) {
    return (
        <div className="flex items-center justify-center p-4 bg-white shadow-md dark:border-white dark:bg-stone-900">
            <button
                className="p-2 text-black dark:text-white bg-white dark:bg-stone-900 rounded-full hover:bg-gray-200 focus:ring focus:ring-slate-100"
                onClick={() =>
                    theme == "dark" ? setTheme("light") : setTheme("dark")
                }
            >
                <img
                    src={theme == "dark" ? DarkSVG : LightSVG}
                    className="fill-black"
                />
            </button>
            <h1 className="flex-1 text-center text-3xl font-bold dark:font-medium dark:text-white">
                MEMD
            </h1>
            <button
                className="px-4 py-2  text-black dark:text-white bg-white dark:bg-stone-900 rounded-lg hover:bg-gray-200 focus:ring focus:ring-slate-100"
                onClick={onClear}
            >
                Clear
            </button>
        </div>
    );
}
const ChatHistory = ({ messages }: { messages: string[] }) => (
    <ScrollArea className="flex-1 overflow-y-auto p-4 bg-white dark:bg-stone-900 shadow-inner justify-center">
        {messages.length > 0 ? (
            messages.map((msg, index) => (
                // auto wrap text
                <p
                    key={index}
                    className="max-w-3xl mx-auto mb-2 text-gray-800 dark:text-white break-words"
                >
                    {msg}
                </p>
            ))
        ) : (
            <p className="text-gray-400 dark:text-white">
                What do you want to ask me?
            </p>
        )}
    </ScrollArea>
);

function UserInput({
    inputHandler,
}: {
    inputHandler: (input: string) => void;
}) {
    const [input, setInput] = useState("");
    return (
        <div className="p-4 bg-gray-200 dark:bg-stone-900 border-t border-gray-300 dark:border-black">
            <div className="flex items-center gap-2">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your message..."
                    className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring focus:ring-blue-300 dark:bg-stone-800 dark:text-white"
                />
                <button
                    onClick={() => {
                        inputHandler(input);
                    }}
                    className="px-4 py-2 text-black dark:text-white bg-white dark:bg-black rounded-lg hover:bg-gray-300 focus:ring focus:ring-gray-600"
                >
                    Send
                </button>
            </div>
        </div>
    );
}

function App() {
    const [messages, setMessages] = useState<string[]>([]);

    const [theme, setTheme] = useState("dark");

    async function greet(input: string) {
        setMessages([...messages, await invoke("greet", { name: input })]);
    }

    return (
        <div className={theme === "dark" ? "dark" : ""}>
            <div className="flex flex-col h-screen bg-gray-100">
                <TopBar
                    onClear={() => setMessages([])}
                    theme={theme}
                    setTheme={setTheme}
                />
                <ChatHistory messages={messages} />
                <UserInput inputHandler={greet} />
            </div>
        </div>
    );
}

export default App;
