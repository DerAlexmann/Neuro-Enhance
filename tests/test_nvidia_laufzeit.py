"""NVIDIA-Bibliotheken fuer die EXE: Laden, Pruefen, Auspacken - ohne Netz.

Die Pakete sind kleine, hier gebaute Wheels; urllib liest sie ueber file://.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

from silberkorn import nvidia_laufzeit as nl


def wheel(pfad: Path, eintraege: dict[str, bytes]) -> nl.Paket:
    with zipfile.ZipFile(pfad, "w") as archiv:
        for name, inhalt in eintraege.items():
            archiv.writestr(name, inhalt)
    daten = pfad.read_bytes()
    return nl.Paket(pfad.stem, "1.0", len(daten), hashlib.sha256(daten).hexdigest(),
                    pfad.as_uri())


@pytest.fixture
def pakete(tmp_path, monkeypatch):
    quelle = tmp_path / "quelle"
    quelle.mkdir()
    a = wheel(quelle / "paket_a.whl", {
        "nvidia/cublas/bin/cublas64_12.dll": b"blas",
        "nvidia/cublas/bin/nvblas64_12.dll": b"ungenutzt",
        "nvidia/cublas/include/cublas.h": b"kopf",
        "nvidia/cublas/lib/cublas.lib": b"lib",
        "paket_a-1.0.dist-info/licenses/License.txt": b"Lizenz A",
        "paket_a-1.0.dist-info/METADATA": b"meta",
    })
    b = wheel(quelle / "paket_b.whl", {
        "nvidia/cudnn/bin/cudnn64_9.dll": b"dnn",
        "nvidia/cudnn/include/crt/host_defines.h": b"unterordner",
        "nvidia/../../boese/bin/ausbruch.dll": b"weg",
        "nvidia/cudnn/include/../../../boese.h": b"weg",
        "paket_b-1.0.dist-info/License.txt": b"Lizenz B",
    })
    monkeypatch.setattr(nl, "PAKETE", (a, b))
    monkeypatch.setattr(nl, "AUSGEPACKT", 1000)
    return a, b


def test_einrichten_packt_dlls_header_und_lizenzen_aus(tmp_path, pakete):
    ziel = tmp_path / "nvidia"
    meldungen = []
    ordner = nl.einrichten(lambda g, s: meldungen.append((g, s)), ordner=str(ziel))
    assert ordner == str(ziel)
    assert sorted(os.listdir(ziel / "bin")) == ["cublas64_12.dll", "cudnn64_9.dll"]
    assert (ziel / "bin" / "cudnn64_9.dll").read_bytes() == b"dnn"
    assert (ziel / "include" / "cublas.h").read_bytes() == b"kopf"
    assert (ziel / "include" / "crt" / "host_defines.h").read_bytes() == b"unterordner"
    assert sorted(os.listdir(ziel)) == ["bin", "fertig.json", "include", "lizenzen"]
    assert not (tmp_path / "boese").exists() and not (tmp_path / "boese.h").exists()
    assert (ziel / "lizenzen" / "paket_a.txt").read_bytes() == b"Lizenz A"
    assert (ziel / "lizenzen" / "paket_b.txt").read_bytes() == b"Lizenz B"
    assert json.loads((ziel / nl.MARKE).read_text()) == {
        "format": nl.FORMAT, "paket_a": "1.0", "paket_b": "1.0"}
    assert nl.vollstaendig(str(ziel))
    assert meldungen[-1][0] == meldungen[-1][1] == nl.download_groesse()
    assert not [n for n in os.listdir(ziel) if n.endswith(".teil")]


def test_falsche_pruefsumme_bricht_ohne_marke_ab(tmp_path, pakete):
    a, b = pakete
    nl.PAKETE = (a, nl.Paket(b.name, b.fassung, b.groesse, "0" * 64, b.url))
    ziel = tmp_path / "nvidia"
    with pytest.raises(nl.LaufzeitFehler, match="Prüfsumme"):
        nl.einrichten(ordner=str(ziel))
    assert not nl.vollstaendig(str(ziel))
    assert not [n for n in os.listdir(ziel) if n.endswith(".teil")]


def test_abbruch_hinterlaesst_kein_fertiges_einrichten(tmp_path, pakete):
    ziel = tmp_path / "nvidia"
    with pytest.raises(nl.Abbruch):
        nl.einrichten(lambda g, s: False, ordner=str(ziel))
    assert not nl.vollstaendig(str(ziel))


def test_fehlende_quelle_wird_zum_laufzeitfehler(tmp_path, pakete):
    a, _b = pakete
    nl.PAKETE = (nl.Paket(a.name, a.fassung, a.groesse, a.sha256,
                          (tmp_path / "gibt_es_nicht.whl").as_uri()),)
    with pytest.raises(nl.LaufzeitFehler):
        nl.einrichten(ordner=str(tmp_path / "nvidia"))


def test_zu_wenig_platz(tmp_path, pakete, monkeypatch):
    monkeypatch.setattr(nl, "AUSGEPACKT", 10 ** 18)
    with pytest.raises(nl.LaufzeitFehler, match="Speicherplatz"):
        nl.einrichten(ordner=str(tmp_path / "nvidia"))


def test_neue_fassung_gilt_als_unvollstaendig(tmp_path, pakete):
    ziel = tmp_path / "nvidia"
    nl.einrichten(ordner=str(ziel))
    a, b = nl.PAKETE
    nl.PAKETE = (a, nl.Paket(b.name, "2.0", b.groesse, b.sha256, b.url))
    assert not nl.vollstaendig(str(ziel))


def test_gefunden_sucht_in_beiden_kandidaten(tmp_path, pakete, monkeypatch):
    erster, zweiter = tmp_path / "programm", tmp_path / "benutzer"
    monkeypatch.setattr(nl, "ordner_kandidaten", lambda: [str(erster), str(zweiter)])
    assert nl.gefunden() is None
    nl.einrichten(ordner=str(zweiter))
    assert nl.gefunden() == str(zweiter)


def test_ausserhalb_der_exe_passiert_nichts(monkeypatch):
    monkeypatch.setattr(nl, "_bin_ordner", None)
    assert not nl.noetig()
    nl.bereitstellen()
    assert nl.dll_ordner() is None


def test_aktivieren_setzt_cuda_path_und_dll_suche(tmp_path, monkeypatch):
    monkeypatch.setattr(nl, "_bin_ordner", None)
    monkeypatch.setenv("PATH", "alt")
    monkeypatch.delenv("CUDA_PATH", raising=False)
    verzeichnisse = []
    monkeypatch.setattr(os, "add_dll_directory", verzeichnisse.append, raising=False)
    (tmp_path / "bin").mkdir()
    bin_ordner = nl.aktivieren(str(tmp_path))
    assert bin_ordner == str(tmp_path / "bin")
    assert os.environ["CUDA_PATH"] == str(tmp_path)
    assert os.environ["PATH"].startswith(bin_ordner + os.pathsep)
    assert verzeichnisse == [bin_ordner]
    assert nl.dll_ordner() == bin_ordner


def test_pakete_passen_zu_requirements():
    """Die EXE laedt dieselben Fassungen, mit denen die Python-Installation laeuft."""
    anforderungen = (Path(__file__).resolve().parent.parent / "requirements.txt").read_text()
    fassungen = {p.name: p.fassung for p in nl.PAKETE}
    assert "nvidia-cudnn-cu12==9.10.*" in anforderungen
    assert fassungen["nvidia-cudnn-cu12"].startswith("9.10.")
    assert "cuda-toolkit==12.8.*" in anforderungen
    assert fassungen["nvidia-cuda-runtime-cu12"].startswith("12.8.")
    for paket in nl.PAKETE:
        assert paket.url.startswith("https://files.pythonhosted.org/packages/")
        assert paket.url.endswith("-win_amd64.whl")
        assert len(paket.sha256) == 64
