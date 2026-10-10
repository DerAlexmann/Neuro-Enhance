"""Tests der Objektivprofile (lensfun): Datenbank lesen, Kamera und Objektiv finden,
Profil rechnen, in der Geometrie anwenden, Datenbank laden und aktualisieren.

Statt der echten Datenbank dient eine kleine im lensfun-Format; das Netz wird
nie benutzt.
"""

from __future__ import annotations

import io
import json
import math
import tarfile

import numpy as np
import pytest
from PIL import Image

from silberkorn import geometrie as g
from silberkorn import objektivprofile as op

DATENBANK = """<lensdatabase version="2">
    <mount>
        <name>Testbajonett Z</name>
        <compat>Testbajonett F</compat>
    </mount>
    <camera>
        <maker>Testwerk Corporation</maker>
        <maker lang="en">Testwerk</maker>
        <model>Testwerk Z 7</model>
        <model lang="en">Z 7</model>
        <mount>Testbajonett Z</mount>
        <cropfactor>1.0</cropfactor>
    </camera>
    <camera>
        <maker>Testwerk</maker>
        <model>Testwerk D 50</model>
        <mount>Testbajonett F</mount>
        <cropfactor>1.5</cropfactor>
    </camera>
    <lens>
        <maker>Testwerk</maker>
        <model>Testwerk AF-S 24-70mm f/4G ED VR</model>
        <mount>Testbajonett F</mount>
        <cropfactor>1.0</cropfactor>
        <calibration>
            <distortion model="ptlens" focal="24" a="0.01" b="-0.04" c="0.02"/>
            <distortion model="poly3" focal="70" k1="-0.01"/>
            <tca model="poly3" focal="24" vr="1.0004" cr="0.0002" br="-0.0001" vb="0.9998"/>
            <tca model="linear" focal="70" kr="1.0001" kb="0.9999"/>
            <vignetting model="pa"
                focal="24" aperture="4" distance="0.5" k1="-0.9" k2="0.3" k3="-0.1"/>
            <vignetting model="pa"
                focal="24" aperture="4" distance="1000" k1="-0.8" k2="0.3" k3="-0.1"/>
            <vignetting model="pa"
                focal="24" aperture="8" distance="1000" k1="-0.4" k2="0.1" k3="0"/>
            <vignetting model="pa"
                focal="70" aperture="4" distance="1000" k1="-0.5" k2="0.2" k3="0"/>
        </calibration>
    </lens>
    <lens>
        <maker>Testwerk</maker>
        <model>Testwerk AF-S 24-70mm f/4G ED VR</model>
        <mount>Testbajonett F</mount>
        <cropfactor>1.5</cropfactor>
        <calibration>
            <vignetting model="pa"
                focal="24" aperture="4" distance="1000" k1="-0.3" k2="0" k3="0"/>
        </calibration>
    </lens>
    <lens>
        <maker>Testwerk</maker>
        <model>Testwerk Z 24-70mm f/4 S</model>
        <mount>Testbajonett Z</mount>
        <cropfactor>1.0</cropfactor>
        <calibration>
            <distortion model="poly5" focal="24" k1="-0.05" k2="0.01"/>
        </calibration>
    </lens>
    <lens>
        <maker>Fremdwerk</maker>
        <model>Fremdwerk 24-70mm f/4 Fremd</model>
        <mount>Fremdbajonett</mount>
        <cropfactor>1.0</cropfactor>
        <calibration>
            <distortion model="poly3" focal="24" k1="0.2"/>
        </calibration>
    </lens>
</lensdatabase>
"""

Z7 = op.Aufnahme("TESTWERK CORPORATION", "TESTWERK Z 7", "VR 24-70mm f/4G", "", 24.0, 4.0,
                 (24.0, 70.0))


@pytest.fixture
def db(tmp_path):
    (tmp_path / "test.xml").write_text(DATENBANK, encoding="utf-8")
    return op.Datenbank.laden(str(tmp_path))


# ----------------------------------------------------------------------
# Lesen und Finden
# ----------------------------------------------------------------------

def test_datenbank_lesen(db):
    assert len(db.kameras) == 2 and len(db.objektive) == 4
    assert db.bajonette["Testbajonett Z"] == {"Testbajonett F"}
    objektiv = db.objektive[0]
    assert [b for b, _t in objektiv.verzeichnung] == [24, 70]
    assert len(objektiv.vignette) == 4


def test_entitaeten_werden_nicht_gelesen(tmp_path):
    (tmp_path / "boese.xml").write_text(
        '<!DOCTYPE l [<!ENTITY a "aaaa">]><lensdatabase><camera><maker>&a;</maker>'
        "<model>X</model><cropfactor>1</cropfactor></camera></lensdatabase>", encoding="utf-8")
    assert op.Datenbank.laden(str(tmp_path)).kameras == []


@pytest.mark.parametrize("name, brennweiten, licht, worte", [
    ("VR 24-120mm f/4G", (24, 120), 4, {"vr", "g"}),
    ("EF50mm f/1.8 STM", (50, 50), 1.8, {"ef", "stm"}),
    ("XF35mmF2 R WR", (35, 35), 2, {"xf", "r", "wr"}),
    ("smc PENTAX-DA 18-55mm F3.5-5.6 AL WR", (18, 55), 3.5, {"smc", "pentax", "da", "al", "wr"}),
    ("Sigma 105mm 1:2.8 Macro", (105, 105), 2.8, {"sigma", "macro"}),
])
def test_name_zerlegen(name, brennweiten, licht, worte):
    assert op.name_zerlegen(name) == (brennweiten, licht, worte)


def test_kamera_aus_exif_namen(db):
    assert op.kamera_finden(db, "TESTWERK CORPORATION", "TESTWERK Z 7").modelle[0] == \
        "Testwerk Z 7"
    assert op.kamera_finden(db, "Testwerk", "Z 7").crop == 1.0
    assert op.kamera_finden(db, "Testwerk", "Testwerk Z 70") is None
    assert op.kamera_finden(db, "Anderswerk", "Testwerk Z 7") is None


def test_objektiv_ueber_adapter_und_worte(db):
    kamera = op.kamera_finden(db, Z7.hersteller, Z7.kamera)
    gefunden = op.objektive_finden(db, Z7, kamera)
    # Beide Eintraege des F-Objektivs (zwei Crops) - nicht das Z-Objektiv ohne "VR",
    # nicht das fremde Bajonett
    assert [o.crop for o in gefunden] == [1.0, 1.5]
    assert {o.name for o in gefunden} == {"Testwerk AF-S 24-70mm f/4G ED VR"}


def test_falsche_lichtstaerke_findet_nichts(db):
    aufnahme = op.Aufnahme("Testwerk", "Testwerk Z 7", "VR 24-70mm f/2.8G", "", 24, 4)
    assert op.objektive_finden(db, aufnahme, None) == []


def test_brennweiten_aus_libraw_wenn_der_name_keine_hat(db):
    aufnahme = op.Aufnahme("Testwerk", "Testwerk Z 7", "AF-S VR f/4G ED", "", 24, 4, (24, 70))
    assert op.objektive_finden(db, aufnahme, None)


# ----------------------------------------------------------------------
# Profil
# ----------------------------------------------------------------------

def test_profil_rechnet_wie_lensfun(db):
    p = op.profil(db, Z7)
    assert p.arten == ("verzeichnung", "farbsaum", "vignette")
    assert p.kamera == "Z 7"
    # Kleinbild, 3:2: halbe Diagonale / halbe Hoehe = sqrt(1 + 1,5^2)
    s = math.hypot(1.5, 1)
    a, b, c = 0.01, -0.04, 0.02
    d = 1 - a - b - c
    assert p.terme[:4] == pytest.approx((c / d ** 2 * s, b / d ** 3 * s ** 2,
                                         a / d ** 4 * s ** 3, 0.0))
    assert p.terme[4:10] == pytest.approx((1.0004, 0.0002 * s, -0.0001 * s * s,
                                           0.9998, 0.0, 0.0))
    # Vignette in halben Diagonalen, Entfernung 1000 statt 0,5
    assert p.terme[10:] == pytest.approx((-0.8, 0.3, -0.1))


def test_profil_interpoliert_brennweite_und_blende(db):
    mitte = op.profil(db, op.Aufnahme(Z7.hersteller, Z7.kamera, Z7.objektiv, "", 47.0, 4.0))
    s = math.hypot(1.5, 1)
    p24 = op.profil(db, Z7).terme
    poly3 = -0.01 / (1 + 0.01) ** 3 * s * s          # 70 mm: poly3, d = 1 - k1
    assert mitte.terme[1] == pytest.approx((p24[1] + poly3) / 2)
    assert mitte.terme[10] == pytest.approx((-0.8 + -0.5) / 2)
    # Blende 5,6 liegt zwischen 4 und 8 - interpoliert im Kehrwert
    halb = op.profil(db, op.Aufnahme(Z7.hersteller, Z7.kamera, Z7.objektiv, "", 24.0, 5.6))
    t = (1 / 5.6 - 1 / 4) / (1 / 8 - 1 / 4)
    assert halb.terme[10] == pytest.approx(-0.8 + t * (-0.4 - -0.8))


def test_crop_kamera_nimmt_passenden_eintrag(db):
    """An der D 50 (Crop 1,5) gilt die Vignette ihres eigenen Eintrags, die Verzeichnung
    aus dem Kleinbild-Eintrag - umgerechnet auf den kleineren Sensor."""
    aufnahme = op.Aufnahme("Testwerk", "Testwerk D 50", "AF-S 24-70mm f/4G ED VR", "", 24, 4)
    p = op.profil(db, aufnahme)
    assert p.terme[10] == pytest.approx(-0.3)
    voll = op.profil(db, Z7)
    # Halbe Diagonale des Bildes in mm ist 1,5-mal kleiner: p1 skaliert mit r
    assert p.terme[0] == pytest.approx(voll.terme[0] / 1.5)
    assert p.terme[1] == pytest.approx(voll.terme[1] / 1.5 ** 2)


def test_ohne_blende_keine_vignette(db):
    p = op.profil(db, op.Aufnahme(Z7.hersteller, Z7.kamera, Z7.objektiv, "", 24.0, 0.0))
    assert "vignette" not in p.arten and p.terme[10:] == (0.0, 0.0, 0.0)


def test_zoom_ohne_brennweite_kein_profil(db):
    assert op.profil(db, op.Aufnahme(Z7.hersteller, Z7.kamera, Z7.objektiv, "", 0.0, 4.0)) \
        is None


# ----------------------------------------------------------------------
# In der Geometrie
# ----------------------------------------------------------------------

def test_geometrie_folgt_dem_polynom():
    terme = (0.05, -0.1, 0.02, 0.01) + op.NEUTRAL[4:]
    p = g._parameter((400, 600), g.Geometrie(profil=terme), 1.0, (400, 600))
    halbdiagonale = math.hypot(600, 400) / 2
    for r in (0.2, 0.5, 0.9):
        x, _y = g._verzeichnet(p, np.array([r * halbdiagonale]), np.array([0.0]))
        faktor = 1 + 0.05 * r - 0.1 * r ** 2 + 0.02 * r ** 3 + 0.01 * r ** 4
        assert x[0] == pytest.approx(r * halbdiagonale * faktor)


def test_farbsaum_und_vignette_am_ort_im_original():
    terme = (0.0,) * 4 + (1.01, 0.0, 0.0, 0.99, 0.0, 0.0, -0.5, 0.0, 0.0)
    p = g._parameter((400, 600), g.Geometrie(profil=terme), 1.0, (400, 600))
    halbdiagonale = math.hypot(600, 400) / 2
    orte, vignette = g._kanaele(p, np.array([0.5 * halbdiagonale]), np.array([0.0]))
    assert orte[0][0][0] == pytest.approx(0.505 * halbdiagonale)
    assert orte[1][0][0] == pytest.approx(0.5 * halbdiagonale)
    assert orte[2][0][0] == pytest.approx(0.495 * halbdiagonale)
    assert vignette[0] == pytest.approx(1 / (1 - 0.5 * 0.25))


def test_bezug_haelt_das_profil_am_original():
    """Auf einer erweiterten Leinwand wirkt das Profil um Mitte und Diagonale des
    Originals - derselbe Punkt des Originals landet am selben Ort."""
    terme = (0.05, -0.1, 0.02, 0.0) + op.NEUTRAL[4:]
    original = g._parameter((400, 600), g.Geometrie(profil=terme), 1.0, (400, 600))
    # Original 600 x 400 bei (100, 50) auf einer Leinwand von 900 x 500
    bezug = ((100 + 300) / 900, (50 + 200) / 500, math.hypot(600, 400) / math.hypot(900, 500))
    leinwand = g._parameter((500, 900), g.Geometrie(profil=terme + bezug), 1.0, (500, 900))
    x, y = np.array([250.0, -120.0]), np.array([90.0, 60.0])        # ab Mitte des Originals
    ox, oy = g._verzeichnet(original, x, y)
    versatz_x, versatz_y = 100 + 300 - 450, 50 + 200 - 250           # Mitte Original - Leinwand
    lx, ly = g._verzeichnet(leinwand, x + versatz_x, y + versatz_y)
    assert lx - versatz_x == pytest.approx(ox) and ly - versatz_y == pytest.approx(oy)


def test_tonnenkorrektur_braucht_zoom_ohne_leere_ecken():
    """Ein Profil, das die Ecken nach innen holt, zoomt so weit, dass nichts leer bleibt."""
    terme = (0.0, -0.08, 0.0, 0.0) + op.NEUTRAL[4:]
    geo = g.Geometrie(profil=terme)
    assert g.automatischer_zoom((400, 600), geo) == pytest.approx(1.0)
    terme = (0.0, 0.08, 0.0, 0.0) + op.NEUTRAL[4:]
    assert g.automatischer_zoom((400, 600), g.Geometrie(profil=terme)) > 1.0


# ----------------------------------------------------------------------
# Herunterladen und Aktualisieren
# ----------------------------------------------------------------------

def _archiv(dateien: dict[str, bytes], verweis: bool = False) -> bytes:
    puffer = io.BytesIO()
    with tarfile.open(fileobj=puffer, mode="w:bz2") as tar:
        for name, daten in dateien.items():
            info = tarfile.TarInfo(name)
            info.size = len(daten)
            tar.addfile(info, io.BytesIO(daten))
        if verweis:
            info = tarfile.TarInfo("verweis.xml")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../geheim.xml"
            tar.addfile(info)
    return puffer.getvalue()


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Ablage in tmp_path, Server als Woerterbuch URL -> Antwort."""
    ablage = tmp_path / "modelle" / "lensfun"
    monkeypatch.setattr(op, "ordner_kandidaten", lambda: [str(ablage)])
    antworten = {op.QUELLE + op.VERSIONEN: json.dumps([1791482221, [0, 1, 2], []]).encode(),
                 op.QUELLE + op.ARCHIV: _archiv({"test.xml": DATENBANK.encode()})}
    aufrufe = []

    def lesen(url, grenze, fortschritt=None):
        aufrufe.append(url)
        if fortschritt is not None:
            fortschritt(len(antworten[url]), len(antworten[url]))
        return antworten[url]

    monkeypatch.setattr(op, "_lesen", lesen)
    op._CACHE.clear()
    yield antworten, ablage, aufrufe
    op._CACHE.clear()


def test_laden_und_aktualisieren(server):
    antworten, ablage, aufrufe = server
    assert op.ordner() is None and op.datenbank() is None and op.aktualisierung() == 1791482221
    assert op.herunterladen() == str(ablage)
    assert op.stand()["zeitstempel"] == 1791482221
    assert len(op.datenbank().objektive) == 4
    assert op.aktualisierung() is None                     # schon aktuell
    antworten[op.QUELLE + op.VERSIONEN] = json.dumps([1791482999, [2], []]).encode()
    assert op.aktualisierung() == 1791482999
    op.herunterladen()
    assert op.stand()["zeitstempel"] == 1791482999
    assert not (ablage.parent / "lensfun.neu").exists() and not (ablage.parent /
                                                                  "lensfun.alt").exists()
    assert all(url.startswith(op.QUELLE) for url in aufrufe)


def test_archiv_nur_flache_xml(server):
    antworten, ablage, _aufrufe = server
    antworten[op.QUELLE + op.ARCHIV] = _archiv({
        "test.xml": DATENBANK.encode(), "../ausbruch.xml": b"<x/>", "unter/ordner.xml": b"<x/>",
        "programm.exe": b"MZ"}, verweis=True)
    op.herunterladen()
    assert sorted(p.name for p in ablage.iterdir()) == ["stand.json", "test.xml"]
    assert not (ablage.parent.parent / "ausbruch.xml").exists()


def test_kaputtes_archiv_laesst_die_alte_datenbank_stehen(server):
    antworten, ablage, _aufrufe = server
    op.herunterladen()
    antworten[op.QUELLE + op.VERSIONEN] = json.dumps([1791489999, [2], []]).encode()
    antworten[op.QUELLE + op.ARCHIV] = _archiv({"leer.xml": b"<lensdatabase/>"})
    with pytest.raises(op.DatenbankFehler):
        op.herunterladen()
    assert op.stand()["zeitstempel"] == 1791482221 and (ablage / "test.xml").exists()


def test_server_ohne_unser_format(server):
    antworten, _ablage, _aufrufe = server
    antworten[op.QUELLE + op.VERSIONEN] = json.dumps([1791489999, [3], []]).encode()
    with pytest.raises(op.DatenbankFehler):
        op.aktualisierung()


def test_suchen_ohne_datenbank(server):
    assert op.suchen(Z7) == op.Suche(False, Z7, None)
    op.herunterladen()
    assert op.suchen(Z7).profil.objektiv == "Testwerk AF-S 24-70mm f/4G ED VR"
    assert op.suchen(None) == op.Suche(True, None, None)


# ----------------------------------------------------------------------
# In der Sitzung
# ----------------------------------------------------------------------

@pytest.mark.parametrize("raw", [True, False])
def test_sitzung_wendet_profil_bei_raw_an(db, monkeypatch, raw):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")
    from silberkorn import bilddatei
    from silberkorn.bearbeitung import Sitzung
    monkeypatch.setattr(op, "suchen", lambda a: op.Suche(True, a, op.profil(db, a)))
    pixel = np.full((60, 90, 3), 30000, np.uint16)
    daten = bilddatei.Bilddaten(pixel=pixel, profil=None, alpha=None, exif=b"", pfad="t.nef",
                                raw=raw, aufnahme=Z7)
    s = Sitzung(daten, vorschau_kante=45)
    try:
        assert s.objektiv.profil is not None
        erwartet = tuple(op.profil(db, Z7).terme) + g.BEZUG_NEUTRAL
        assert s.werte.objektivprofil == (erwartet if raw else ())
        assert not s.geaendert                    # automatisch heisst nicht "geaendert"
        assert s.grundwerte().objektivprofil == s.werte.objektivprofil
        s.werte.objektivprofil = s.objektiv_terme()
        bild, _ms, _h = s.vorschau()
        # Vignette aufgehellt: Ecke heller als die Mitte
        hoehe, breite = bild.shape[:2]
        assert bild[1, 1].mean() > bild[hoehe // 2, breite // 2].mean()
    finally:
        s.schliessen()


# ----------------------------------------------------------------------
# Aufnahmedaten aus EXIF
# ----------------------------------------------------------------------

def test_aufnahme_aus_jpeg(tmp_path):
    from silberkorn import bilddatei
    exif = Image.Exif()
    exif[0x010F] = "Testwerk"
    exif[0x0110] = "Testwerk Z 7"
    details = exif.get_ifd(0x8769)
    details[0xA434] = "VR 24-70mm f/4G"
    details[0x920A] = 35.0
    details[0x829D] = 5.6
    details[0xA432] = (24.0, 70.0, 4.0, 4.0)
    pfad = str(tmp_path / "foto.jpg")
    Image.new("RGB", (16, 12), (90, 90, 90)).save(pfad, exif=exif)
    aufnahme = bilddatei.laden(pfad).aufnahme
    assert aufnahme == op.Aufnahme("Testwerk", "Testwerk Z 7", "VR 24-70mm f/4G", "", 35.0, 5.6,
                                   (24.0, 70.0))


def test_ohne_objektiv_keine_aufnahme(tmp_path):
    from silberkorn import bilddatei
    pfad = str(tmp_path / "foto.png")
    Image.new("RGB", (16, 12)).save(pfad)
    assert bilddatei.laden(pfad).aufnahme is None
