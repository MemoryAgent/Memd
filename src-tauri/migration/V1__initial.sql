CREATE TABLE document (
    id INTEGER PRIMARY KEY, -- https://www.sqlite.org/autoinc.html
    doc_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chunk (
    id INTEGER PRIMARY KEY,
    full_doc_id INTEGER,
    chunk_index INTEGER,
    tokens INTEGER,
    content TEXT,
    content_vector BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(full_doc_id) REFERENCES document(id) ON DELETE CASCADE
);

CREATE INDEX chunk_index ON chunk(full_doc_id);

CREATE TABLE entity (
    id INTEGER PRIMARY KEY,
    name TEXT,
    embedding BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE entity_chunk (
    id INTEGER PRIMARY KEY,
    entity_id INTEGER,
    chunk_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(entity_id) REFERENCES entity(id) ON DELETE CASCADE,
    FOREIGN KEY(chunk_id) REFERENCES chunk(id) ON DELETE CASCADE
);

CREATE INDEX entity_chunk_entity_index ON entity_chunk(entity_id);

CREATE INDEX entity_chunk_chunk_index ON entity_chunk(chunk_id);

CREATE TABLE relation (
    id INTEGER PRIMARY KEY,
    source_id INTEGER,
    target_id INTEGER,
    relationship TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_id) REFERENCES entity(id) ON DELETE CASCADE,
    FOREIGN KEY(target_id) REFERENCES entity(id) ON DELETE CASCADE
);

CREATE INDEX relation_source_index ON relation(source_id);

CREATE INDEX relation_target_index ON relation(target_id);
