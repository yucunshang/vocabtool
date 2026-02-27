"""Create a test Kindle vocab.db for testing file upload."""
import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), "test_vocab.db")
if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("""CREATE TABLE WORDS (
    id TEXT PRIMARY KEY NOT NULL,
    word TEXT,
    stem TEXT,
    lang TEXT,
    category INTEGER DEFAULT 0,
    timestamp INTEGER DEFAULT 0,
    profileid TEXT
)""")

c.execute("""CREATE TABLE LOOKUPS (
    id TEXT PRIMARY KEY NOT NULL,
    word_key TEXT,
    book_key TEXT,
    dict_key TEXT,
    pos TEXT,
    usage TEXT,
    timestamp INTEGER DEFAULT 0
)""")

c.execute("""CREATE TABLE BOOK_INFO (
    id TEXT PRIMARY KEY NOT NULL,
    asin TEXT,
    guid TEXT,
    lang TEXT,
    title TEXT,
    authors TEXT
)""")

test_words = [
    ("en:serendipity", "serendipity", "serendipity", "en"),
    ("en:ubiquitous", "ubiquitous", "ubiquitous", "en"),
    ("en:ephemeral", "ephemeral", "ephemeral", "en"),
    ("en:resilience", "resilience", "resilience", "en"),
    ("en:pragmatic", "pragmatic", "pragmatic", "en"),
    ("en:eloquent", "eloquent", "eloquent", "en"),
    ("en:meticulous", "meticulous", "meticulous", "en"),
    ("en:ambiguous", "ambiguous", "ambiguous", "en"),
    ("en:benevolent", "benevolent", "benevolent", "en"),
    ("en:diligent", "diligent", "diligent", "en"),
    ("en:exquisite", "exquisite", "exquisite", "en"),
    ("en:formidable", "formidable", "formidable", "en"),
]

for wid, word, stem, lang in test_words:
    c.execute("INSERT INTO WORDS VALUES (?,?,?,?,0,0,NULL)", (wid, word, stem, lang))

for i, (wid, word, stem, lang) in enumerate(test_words):
    c.execute(
        "INSERT INTO LOOKUPS VALUES (?,?,?,?,?,?,?)",
        (f"lookup_{i}", wid, "book1", "dict1", "0", f"The word {word} appeared in context.", 0),
    )

c.execute("INSERT INTO BOOK_INFO VALUES ('book1','B001','guid1','en','Test Book','Test Author')")
conn.commit()
conn.close()

print(f"Created: {db_path}")
print(f"Size: {os.path.getsize(db_path)} bytes")

# Verify
conn = sqlite3.connect(db_path)
c = conn.cursor()
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print(f"Tables: {tables}")
for t in tables:
    cnt = c.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    print(f"  {t}: {cnt} rows")
words = [r[0] for r in c.execute("SELECT word FROM WORDS").fetchall()]
print(f"Words: {words}")
conn.close()
