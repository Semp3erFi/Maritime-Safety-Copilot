# Curated knowledge base

The notes here are curated educational summaries and study material for this MVP.
They are **not authoritative regulatory text**, an
operational checklist, or a complete statement of legal requirements. No full
regulatory publications are included. Knowledge Markdown files in this directory
are discovered dynamically. README files are excluded (case-insensitive). Other
documentation can be excluded with YAML `document_type: documentation`; no knowledge
filenames are hard-coded.

The names and high-level relationship of SOLAS IX and the ISM Code can be checked
against the [IMO introduction to the ISM Code](https://www.imo.org/en/OurWork/HumanElement/Pages/ISMCode.aspx).
For real work, obtain current official publications and applicable flag-state and
company guidance. The summaries cannot establish compliance or replace approved
operational procedures.

Add small UTF-8 `.md` or `.txt` files directly in this directory. Use one subject per
file and descriptive Markdown headings with identifiers. Optional YAML front matter
must appear at the start of the file, between `---` delimiters, for example:

```yaml
---
title: My study note
topic: safety_management
keywords:
  - learning
document_type: educational_summary
---
```

All front-matter fields are preserved as source `metadata`, including custom fields,
lists and nested mappings. YAML dates become ISO strings in JSON responses. The
source title comes from YAML `title`, otherwise the first Markdown H1, otherwise the
filename stem. The actual filename is always returned separately as `source`.
Only the body becomes retrieved excerpts; title, filename and metadata also
participate in keyword and embedding search. Referenced publications are metadata,
not evidence that the full regulatory text has been retrieved.

Files are reread each request; changed contents, metadata, filenames, additions and
removals are reflected in the in-memory embedding cache automatically on the next
semantic/hybrid query. No persistent index needs rebuilding. Invalid YAML produces
an error naming the affected file. Keep this
curated demo small. The Streamlit sidebar can also save UTF-8 `.md` files up to 1 MB
here using sanitized, unique filenames. Click **Save to knowledge base** after
selecting a file. The existing loader handles the saved file on the next retrieval;
no separate ingestion service is used.
