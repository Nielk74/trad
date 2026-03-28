# tests/conftest.py
import pytest
from models.stt import STTModel
from models.translation import TranslationModel
from models.tts import TTSModel


def pytest_addoption(parser):
    parser.addoption(
        "--config",
        action="store",
        default="small",
        choices=["high", "medium", "small"],
        help="Model tier to use for tests",
    )


@pytest.fixture(scope="session")
def config(request):
    return request.config.getoption("--config")


@pytest.fixture(scope="session")
def stt_model(config):
    model = STTModel(config)
    model.load()
    return model


@pytest.fixture(scope="session")
def translation_model(config):
    model = TranslationModel(config)
    model.load()
    return model


@pytest.fixture(scope="session")
def tts_model(config):
    model = TTSModel(config)
    model.load()
    return model
