from app.config import Settings


def test_demo_mode_allows_placeholders():
    Settings(app_mode="demo", _env_file=None).assert_production_ready()


def test_production_rejects_placeholders():
    settings = Settings(app_mode="production", _env_file=None)
    try:
        settings.assert_production_ready()
    except RuntimeError as exc:
        assert "placeholders" in str(exc)
    else:
        raise AssertionError("Production mode accepted placeholder secrets")
