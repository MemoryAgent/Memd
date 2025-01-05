import { PathLike } from "fs";

type LeafContent =
    | {
          type: "text";
          content: string;
      }
    | {
          type: "fileRef";
          filePath: PathLike;
      };

type LeafNode = {
    content: LeafContent;
    entities: string[];
    gist: string;
    kind: "leaf";
};

type InternalNode = {
    summary: string;
    kind: "internal";
};

type AbstractionNode = {
    summary: string;
    kind: "abstraction";
};

export type MemoryNode = {
    id: string;
} & (LeafNode | InternalNode | AbstractionNode);

export type Relationship = {
    from: string;
    to: string;
    relation: string;
    conf: number;
};
