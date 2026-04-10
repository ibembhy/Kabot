from kabot.settings import load_settings


def test_settings_load() -> None:
    settings = load_settings()
    assert settings.app["name"] == "Kabot"
