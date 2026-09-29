import pytest

from frontend.uploads import MAX_UPLOAD_BYTES, save_markdown


@pytest.mark.parametrize("name", ["../../notes.md", r"C:\private\CON.MD", "../..//a:b?.md"])
def test_upload_names_stay_in_data_and_preserve_existing_files(tmp_path, name):
    first = save_markdown(name, b"# First\nEvidence.", tmp_path)
    second = save_markdown(name, b"# Second\nOther evidence.", tmp_path)
    assert first.parent == second.parent == tmp_path.resolve()
    assert first != second
    assert first.name.startswith("upload_") and first.suffix == ".md"
    assert first.read_bytes() == b"# First\nEvidence."
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "name,content",
    [
        ("notes.pdf", b"text"),
        ("notes.md", b"  "),
        ("notes.md", b"\xff"),
        ("notes.md", b"null\x00byte"),
        ("notes.md", b"x" * (MAX_UPLOAD_BYTES + 1)),
    ],
    ids=["wrong-extension", "blank", "invalid-utf8", "null-byte", "too-large"],
)
def test_invalid_upload_is_rejected_without_writing(tmp_path, name, content):
    with pytest.raises(ValueError):
        save_markdown(name, content, tmp_path)
    assert not list(tmp_path.iterdir())
