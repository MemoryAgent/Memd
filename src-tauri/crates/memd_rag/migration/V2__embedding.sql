CREATE TABLE internal_vector_index (
	id INTEGER PRIMARY KEY,
	summary TEXT,
	embedding BLOB
);

CREATE TABLE internal_vector_children (
	parent_id INTEGER,
	child_id INTEGER,
	FOREIGN KEY(parent_id) REFERENCES internal_vector_index(id) ON DELETE CASCADE
);
