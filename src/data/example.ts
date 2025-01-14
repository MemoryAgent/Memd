import { MemoryNode } from "./node";

export default {
    nodes: [
        {
            id: "world",
            summary: "Reminders",
            kind: "internal",
        },
        {
            id: "hello",
            content: {
                type: "text",
                content: "Please wake me up at 8AM tomorrow.",
            },
            entities: ["me"],
            gist: "The user asks the system to wake him up tomorrow",
            kind: "leaf",
        },
        {
            id: "alice",
            content: {
                type: "text",
                content: "Please call alice in 10AM tomorrow.",
            },
            entities: ["alice"],
            gist: "The user asks the system to call alice at 10AM tomorrow",
            kind: "leaf",
        },
        {
            id: "meeting",
            content: {
                type: "text",
                content: "Have a meeting at 2PM",
            },
            entities: ["meeting"],
            gist: "The user has a meeting at 2PM",
            kind: "leaf",
        },
        {
            id: "reading",
            content: {
                type: "text",
                content: "Remind me to read War and Peace",
            },
            entities: ["War and Peace"],
            gist: "Remind the user to read War and Peace",
            kind: "leaf",
        },
        {
            id: "book",
            kind: "internal",
            summary: "books"
        },
        {
            id: "wnp",
            content: {
                type: "fileRef",
                filePath: "C:/war_and_peace.txt"
            },
            gist: "The book war and peace by Leo Tolstoy.",
            kind: "leaf",
            entities: ["Leo Tolstoy", "War and Peace"]
        }
    ] as MemoryNode[],
    link: [
        {
            from: "hello",
            to: "world",
            relation: "contain",
            conf: 1.0,
        },
        {
            from: "alice",
            to: "world",
            relation: "contain",
            conf: 1.0,
        },
        {
            from: "meeting",
            to: "world",
            relation: "contain",
            conf: 1.0,
        },
        {
            from: "wnp",
            to: "book",
            relation: "contain",
            conf: 1.0,
        },
        {
            from: "reading",
            to: "world",
            relation: "contain",
            conf: 1.0,
        },
        {
            from: "reading",
            to: "wnp",
            relation: "correlated",
            conf: 1.0,
        },

    ],
};
