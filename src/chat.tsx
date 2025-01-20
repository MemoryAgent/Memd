import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";
import DarkSVG from "./dark.svg";
import LightSVG from "./light.svg";
import Logo from "../src-tauri/icons/icon.png";
import UserAvatar from "./user.png";
import { ScrollArea } from "./scroll-area";
import { useTheme } from "./components/theme-provider";

type Message = {
    text: string;
    sender: "user" | "system";
};

function TopBar({
    onClear,
    onBench,
}: {
    onClear: () => void;
    onBench: () => void;
}) {
    const { theme, setTheme } = useTheme();
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
            <button
                className="px-4 py-2  text-black dark:text-white bg-white dark:bg-stone-900 rounded-lg hover:bg-gray-200 focus:ring focus:ring-slate-100"
                onClick={onBench}
            >
                Bench
            </button>
        </div>
    );
}

const QuestionRow = ({ question }: { question: string }) => (
    <div className="flex justify-items-end items-center gap-2 p-4 bg-gray-200 dark:bg-stone-800">
        <p className="flex-1"></p>
        <p className="text-gray-800 dark:text-white">{question}</p>
        <img src={UserAvatar} className="w-8 h-8 rounded-full" />
    </div>
);

const ResponseRow = ({ response }: { response: string }) => (
    <div className="flex items-center gap-2 p-4 dark:bg-stone-900">
        <img src={Logo} className="flex-shrink-0 w-8 h-8 rounded-full" />
        <p className="flex-1 text-gray-800 dark:text-white">{response}</p>
    </div>
);

const ChatHistory = ({ messages }: { messages: Message[] }) => (
    <ScrollArea className="flex-1 overflow-y-auto p-4 bg-white dark:bg-stone-900 shadow-inner justify-center">
        {messages.length > 0 ? (
            messages.map((message, index) =>
                message.sender === "user" ? (
                    <QuestionRow key={index} question={message.text} />
                ) : (
                    <ResponseRow key={index} response={message.text} />
                )
            )
        ) : (
            <p className="text-gray-400 dark:text-white">
                What do you want to ask me?
            </p>
        )}
    </ScrollArea>
);

function UserInput({
    inputHandler,
    uploadHandler,
}: {
    inputHandler: (input: string) => void;
    uploadHandler?: () => void;
}) {
    const [input, setInput] = useState("");
    return (
        <div className="p-4 bg-gray-200 dark:bg-stone-900 border-t border-gray-300 dark:border-black">
            <div className="flex items-center gap-2">
                <button
                    onClick={uploadHandler}
                    className="px-4 py-2 text-black dark:text-white bg-white dark:bg-black rounded-lg hover:bg-gray-300 focus:ring focus:ring-gray-600"
                >
                    Upload
                </button>
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

async function invokeErr<T>(cmd: string, args?: any) {
    try {
        return await invoke<T>(cmd, args);
    } catch (e) {
        console.error(e);
        return `An error occurred: ${e}`;
    }
}

function Chat() {
    const [messages, setMessages] = useState<Message[]>([]);

    async function greet(input: string) {
        setMessages([
            ...messages,
            { text: input, sender: "user" },
            {
                text: await invokeErr("chat", {
                    question: input,
                }),
                sender: "system",
            },
        ]);
    }

    async function open_bench() {
        await invokeErr("open_bench", {});
    }

    async function uploadFile() {
        await invokeErr("pick_file", {});
    }

    return (
        <div className="flex flex-col h-screen bg-gray-100 w-full">
            <TopBar onClear={() => setMessages([])} onBench={open_bench} />
            <ChatHistory messages={messages} />
            <UserInput inputHandler={greet} uploadHandler={uploadFile} />
        </div>
    );
}

export default Chat;
