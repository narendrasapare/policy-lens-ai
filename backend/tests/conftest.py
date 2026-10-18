import pytest


@pytest.fixture(autouse=True)
def isolate_startup_documents(tmp_path, monkeypatch):
    import policylens.main as application
    monkeypatch.setattr(application, "DATA_DIRECTORY", tmp_path)
