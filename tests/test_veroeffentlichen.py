"""Fuer Veroeffentlichung speichern: Groesse, Wasserzeichen, Metadaten - ohne GPU."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from silberkorn import veroeffentlichen as v
from silberkorn.veroeffentlichen import Vorlage

GPS_IFD = 0x8825
KAMERA = 0x0110


def grau(hoehe=400, breite=600, wert=0.5):
    return np.full((hoehe, breite, 3), wert, np.float32)


def exif_mit_gps() -> bytes:
    exif = Image.Exif()
    exif[KAMERA] = "Testkamera"
    exif[GPS_IFD] = {1: "N", 2: (52.0, 31.0, 0.0)}
    return exif.tobytes()


# --------------------------------------------------------------------------
# Groesse
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("form", "vorlage", "erwartet"), [
    ((4000, 6000), Vorlage(kante=1920), (1280, 1920)),
    ((6000, 4000), Vorlage(kante=1920), (1920, 1280)),          # Hochformat
    ((1000, 1500), Vorlage(kante=1920), (1000, 1500)),          # nie vergroessern
    ((4000, 6000), Vorlage(groesse_art="prozent", prozent=25), (1000, 1500)),
    ((4000, 6000), Vorlage(groesse_art="prozent", prozent=100), (4000, 6000)),
])
def test_zielgroesse(form, vorlage, erwartet):
    assert v.zielgroesse(*form, vorlage) == erwartet


def test_verkleinern_haelt_flaechen():
    bild = v.verkleinern(grau(400, 600, 0.25), 200, 300)
    assert bild.shape == (200, 300, 3)
    assert np.allclose(bild, 0.25, atol=1e-4)


def test_vorlage_aus_dict_prueft_werte():
    vorlage = Vorlage.aus_dict({"kante": 10, "position": "irgendwo", "muster": 1,
                                "deckkraft": 500, "text": "Hallo", "unbekannt": 3})
    assert vorlage.kante == v.KANTE_MIN
    assert vorlage.position == v.POSITIONEN[0]
    assert vorlage.muster is False                # 1 ist kein bool
    assert vorlage.deckkraft == 100
    assert vorlage.text == "Hallo"
    assert Vorlage.aus_dict(None) == Vorlage()
    assert Vorlage.aus_dict(Vorlage(text="x").als_dict()) == Vorlage(text="x")


# --------------------------------------------------------------------------
# Wasserzeichen
# --------------------------------------------------------------------------

def geaendert(vorher, nachher):
    return np.abs(nachher - vorher).max(axis=2) > 1e-3


def test_ohne_text_und_logo_kein_wasserzeichen():
    vorlage = Vorlage()
    assert not vorlage.mit_wasserzeichen
    assert v.zeichen_bild(vorlage) is None
    bild = grau()
    assert v.wasserzeichen(bild, None, vorlage, None)[0] is bild


@pytest.mark.parametrize("position", v.POSITIONEN)
def test_wasserzeichen_an_seiner_stelle(position):
    vorlage = Vorlage(text="Silberkorn", position=position, wz_groesse=20)
    bild = grau()
    neu, _alpha = v.wasserzeichen(bild, None, vorlage, v.zeichen_bild(vorlage))
    ys, xs = np.nonzero(geaendert(bild, neu))
    assert len(xs)
    mitte_x, mitte_y = xs.mean() / 600, ys.mean() / 400
    if "rechts" in position:
        assert mitte_x > 0.7
    if "links" in position:
        assert mitte_x < 0.3
    if "oben" in position:
        assert mitte_y < 0.3
    if "unten" in position:
        assert mitte_y > 0.7
    if position == "mitte":
        assert 0.4 < mitte_x < 0.6 and 0.4 < mitte_y < 0.6
    # Breite etwa wie eingestellt (Schattenrand inklusive)
    assert 0.12 < (xs.max() - xs.min()) / 600 < 0.25


def test_muster_bedeckt_das_ganze_bild():
    vorlage = Vorlage(text="Silberkorn", muster=True, wz_groesse=15)
    bild = grau()
    neu, _alpha = v.wasserzeichen(bild, None, vorlage, v.zeichen_bild(vorlage))
    maske = geaendert(bild, neu)
    for viertel in (maske[:200, :300], maske[:200, 300:], maske[200:, :300], maske[200:, 300:]):
        assert viertel.mean() > 0.02


def test_deckkraft_und_farbe():
    bild = grau(wert=0.5)
    werte = {}
    for farbe in v.TEXTFARBEN:
        for deckkraft in (30, 100):
            vorlage = Vorlage(text="X", position="mitte", wz_groesse=40, deckkraft=deckkraft,
                              textfarbe=farbe)
            neu, _alpha = v.wasserzeichen(bild, None, vorlage, v.zeichen_bild(vorlage))
            werte[farbe, deckkraft] = (neu.max(), neu.min())
    assert werte["weiss", 100][0] > 0.99 and werte["weiss", 30][0] < 0.7
    assert werte["schwarz", 100][1] < 0.01 and werte["schwarz", 30][1] > 0.3


def test_logo_und_alphakanal(tmp_path):
    pfad = tmp_path / "logo.png"
    logo = Image.new("RGBA", (50, 50), (255, 0, 0, 0))
    logo.paste((255, 0, 0, 255), (10, 10, 40, 40))
    logo.save(pfad)
    vorlage = Vorlage(logo=str(pfad), position="oben_links", deckkraft=100)
    zeichen = v.zeichen_bild(vorlage, v.logo_laden(vorlage.logo))
    alpha = np.zeros((400, 600), np.float32)
    neu, neu_alpha = v.wasserzeichen(grau(), alpha, vorlage, zeichen)
    assert neu[..., 0].max() > 0.99 and neu[..., 1].min() < 0.01      # rot
    assert neu_alpha.max() > 0.99 and neu_alpha[-1, -1] == 0          # Logo deckt, Rest frei
    assert v.logo_laden(str(tmp_path / "fehlt.png")) is None


# --------------------------------------------------------------------------
# Metadaten und Speichern
# --------------------------------------------------------------------------

def test_exif_ohne_kameradaten():
    vorlage = Vorlage(urheber="Jörg Müller", copyright="© 2026 Jörg Müller")
    exif = Image.Exif()
    exif.load(v.exif_daten(exif_mit_gps(), vorlage))
    assert exif[v.EXIF_KUENSTLER] == "Jorg Muller"
    assert exif[v.EXIF_COPYRIGHT] == "(C) 2026 Jorg Muller"
    assert KAMERA not in exif and GPS_IFD not in exif


def test_exif_mit_kameradaten_aber_ohne_gps():
    exif = Image.Exif()
    exif.load(v.exif_daten(exif_mit_gps(), Vorlage(kameradaten_entfernen=False)))
    assert exif[KAMERA] == "Testkamera"
    assert GPS_IFD not in exif


def test_xmp():
    xmp = v.xmp_daten(Vorlage(urheber="A & B", copyright="© 2026 A", webadresse="https://x.y"),
                      ki_inhalt=True).decode()
    assert "<rdf:li>A &amp; B</rdf:li>" in xmp
    assert "© 2026 A" in xmp and "https://x.y" in xmp
    assert v.DATA_MINING in xmp and v.KI_QUELLE in xmp
    leer = v.xmp_daten(Vorlage(ki_training_verbieten=False))
    assert leer == b""


@pytest.mark.parametrize("endung", v.FORMATE)
def test_speichern(tmp_path, endung):
    pfad = tmp_path / f"bild{endung}"
    rgb = (np.random.default_rng(1).random((800, 1200, 3)) * 65535).astype(np.uint16)
    vorlage = Vorlage(kante=600, text="Test", urheber="Alex", copyright="© 2026 Alex")
    groesse = v.speichern(str(pfad), rgb, None, exif_mit_gps(), vorlage, ki_inhalt=False)
    assert groesse == (600, 400)
    with Image.open(pfad) as bild:
        assert bild.size == (600, 400)
        xmp = bild.info.get("xmp") or bild.info.get("XML:com.adobe.xmp")
        exif = bild.getexif()
    xmp = xmp.decode() if isinstance(xmp, bytes) else xmp
    assert "© 2026 Alex" in xmp and v.DATA_MINING in xmp and v.KI_QUELLE not in xmp
    assert exif[v.EXIF_KUENSTLER] == "Alex" and GPS_IFD not in exif


def test_speichern_mit_alpha(tmp_path):
    pfad = tmp_path / "frei.png"
    rgb = np.full((100, 200, 3), 128, np.uint8)
    alpha = np.zeros((100, 200), np.uint8)
    alpha[:, :100] = 255
    v.speichern(str(pfad), rgb, alpha, b"", Vorlage(kante=100))
    with Image.open(pfad) as bild:
        assert bild.mode == "RGBA" and bild.size == (100, 50)
        kanal = np.asarray(bild.getchannel("A"))
    assert kanal[:, :45].min() == 255 and kanal[:, 55:].max() == 0


# --------------------------------------------------------------------------
# Dialog
# --------------------------------------------------------------------------

def test_dialog(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QLabel, QPushButton

    from silberkorn.veroeffentlichen_dialog import VeroeffentlichenDialog
    _app = QApplication.instance() or QApplication([])
    vorschau = np.full((300, 450, 3), 100, np.uint8)
    dialog = VeroeffentlichenDialog(None, vorschau, (4000, 6000),
                                    Vorlage(text="Hallo", urheber="Alex", format=".png"))
    dialog.show()
    assert "1920 × 1280" in dialog.groesse_info.text()
    assert dialog.vorlage().copyright.endswith(" Alex")              # Vorschlag aus Urheber
    assert dialog.mit_format(".webp").format == ".webp"

    dialog.groesse_art.setCurrentIndex(dialog.groesse_art.findData("prozent"))
    dialog.prozent.setValue(10)
    assert "600 × 400" in dialog.groesse_info.text()
    assert dialog.kante.isHidden() and not dialog.prozent.isHidden()

    leer = dialog.bild.pixmap().toImage()
    dialog.muster.setChecked(True)
    assert dialog.bild.pixmap().toImage() != leer                     # Vorschau folgt
    assert not dialog.position.isEnabled()

    for knopf in dialog.findChildren(QPushButton):
        assert knopf.text().strip(), "unbeschrifteter Knopf"
    assert all(label.text() or label is dialog.bild or label.objectName() == "wert"
               for label in dialog.findChildren(QLabel) if label.isVisible())
    dialog.close()


# --------------------------------------------------------------------------
# Mit der Bearbeitung auf der GPU
# --------------------------------------------------------------------------

def test_sitzung_veroeffentlichen(tmp_path):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")
    from silberkorn import bilddatei
    from silberkorn.bearbeitung import Sitzung

    class Weiss:
        def raender(self, srgb, gross_b, gross_h, x, y, seed, fortschritt=None):
            return cp.ones((gross_h, gross_b, 3), dtype=cp.float32)

    daten = bilddatei.Bilddaten(pixel=np.full((300, 400, 3), 90, np.uint8), profil=None,
                                alpha=None, exif=exif_mit_gps(), pfad="t.png")
    sitzung = Sitzung(daten, vorschau_kante=80)
    try:
        sitzung.werte.belichtung = 0.5
        pfad = str(tmp_path / "web.jpg")
        _ms, groesse = sitzung.veroeffentlichen(pfad, Vorlage(kante=200, text="X"))
        assert groesse == (200, 150)
        assert sitzung.geaendert                     # gilt nicht als Speichern der Arbeit
        with Image.open(pfad) as bild:
            assert bild.size == (200, 150)
            assert np.asarray(bild)[75, 100].mean() > 100      # belichtet
            assert v.KI_QUELLE not in bild.info["xmp"].decode()

        assert not sitzung.ki_inhalt
        sitzung.ki_erweitern(Weiss(), 2.0, seed=1)
        assert sitzung.ki_inhalt
        sitzung.veroeffentlichen(pfad, Vorlage(kante=200))
        with Image.open(pfad) as bild:
            assert v.KI_QUELLE in bild.info["xmp"].decode()
    finally:
        sitzung.schliessen()
