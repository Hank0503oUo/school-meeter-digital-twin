from __future__ import annotations

import pytest

from src.local_gemma_runtime import (
    EXTERNAL_GEMMA_MMPROJ_PATH,
    EXTERNAL_GEMMA_MODEL_PATH,
    EXTERNAL_LLAMA_SERVER_EXE,
    PACKAGED_GEMMA_MMPROJ_PATH,
    PACKAGED_GEMMA_MODEL_PATH,
    PACKAGED_LLAMA_SERVER_EXE,
    LocalGemmaConfig,
    build_llama_server_command,
    gemma_autostart_enabled,
    resolve_local_gemma_config,
)


def _expected(packaged, external):
    if packaged.is_file():
        return packaged
    if external.is_file():
        return external
    return packaged


def test_resolve_local_gemma_config_defaults():
    config = resolve_local_gemma_config({})

    assert config.model_path == _expected(PACKAGED_GEMMA_MODEL_PATH, EXTERNAL_GEMMA_MODEL_PATH)
    assert config.server_exe == _expected(PACKAGED_LLAMA_SERVER_EXE, EXTERNAL_LLAMA_SERVER_EXE)
    assert config.mmproj_path == _expected(PACKAGED_GEMMA_MMPROJ_PATH, EXTERNAL_GEMMA_MMPROJ_PATH)
    assert config.host == "127.0.0.1"
    assert config.port == 8088
    assert config.ctx_size == 4096
    assert config.gpu_layers is None
    assert config.base_url == "http://127.0.0.1:8088/v1"


def test_resolve_local_gemma_config_env_overrides(tmp_path):
    model = tmp_path / "gemma.gguf"
    server = tmp_path / "llama-server.exe"
    mmproj = tmp_path / "mmproj.gguf"
    config = resolve_local_gemma_config(
        {
            "ENERGY_GEMMA_MODEL_PATH": str(model),
            "ENERGY_LLAMA_SERVER_EXE": str(server),
            "ENERGY_GEMMA_MMPROJ_PATH": str(mmproj),
            "ENERGY_GEMMA_HOST": "localhost",
            "ENERGY_GEMMA_PORT": "9090",
            "ENERGY_GEMMA_CTX": "8192",
            "ENERGY_GEMMA_GPU_LAYERS": "33",
            "ENERGY_GEMMA_STARTUP_TIMEOUT_SECONDS": "3.5",
        }
    )

    assert config.model_path == model
    assert config.server_exe == server
    assert config.mmproj_path == mmproj
    assert config.host == "localhost"
    assert config.port == 9090
    assert config.ctx_size == 8192
    assert config.gpu_layers == 33
    assert config.startup_timeout_seconds == 3.5


def test_gemma_autostart_enabled_by_provider_or_flag():
    assert gemma_autostart_enabled({}) is True
    assert gemma_autostart_enabled({"ENERGY_LOCAL_LLM_PROVIDER": "gemma"}) is True
    assert gemma_autostart_enabled({"ENERGY_GEMMA_AUTOSTART": "yes"}) is True
    assert gemma_autostart_enabled({"ENERGY_LOCAL_LLM_PROVIDER": "lmstudio"}) is False


def test_build_llama_server_command_requires_server_exe(tmp_path):
    config = LocalGemmaConfig(model_path=tmp_path / "gemma.gguf", server_exe=None)

    with pytest.raises(RuntimeError):
        build_llama_server_command(config)


def test_build_llama_server_command_includes_gpu_layers(tmp_path):
    config = resolve_local_gemma_config(
        {
            "ENERGY_GEMMA_MODEL_PATH": str(tmp_path / "gemma.gguf"),
            "ENERGY_LLAMA_SERVER_EXE": str(tmp_path / "llama-server.exe"),
            "ENERGY_GEMMA_GPU_LAYERS": "12",
        }
    )

    command = build_llama_server_command(config)

    assert command[:3] == [str(tmp_path / "llama-server.exe"), "-m", str(tmp_path / "gemma.gguf")]
    assert "--n-gpu-layers" in command
    assert "12" in command


def test_build_llama_server_command_includes_existing_mmproj(tmp_path):
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_text("fake", encoding="utf-8")
    config = LocalGemmaConfig(
        model_path=tmp_path / "gemma.gguf",
        server_exe=tmp_path / "llama-server.exe",
        mmproj_path=mmproj,
    )

    command = build_llama_server_command(config)

    assert "--mmproj" in command
    assert str(mmproj) in command


def test_build_llama_server_command_omits_missing_mmproj(tmp_path):
    config = LocalGemmaConfig(
        model_path=tmp_path / "gemma.gguf",
        server_exe=tmp_path / "llama-server.exe",
        mmproj_path=tmp_path / "missing-mmproj.gguf",
    )

    command = build_llama_server_command(config)

    assert "--mmproj" not in command
