"""KI-Erweitern (Outpainting): Leinwand, Latents, Zeitplan, Farbkorrektur und Einsetzen.

Die Rechenhilfen nehmen NumPy- und CuPy-Arrays. Mit NumPy (und SciPy fuer die
Weichzeichnung) laufen die Tests ueberall, mit CuPy nur auf einer Grafikkarte;
fehlt etwas davon, ueberspringt sich der Fall. Nur der letzte Test braucht das
Modell selbst.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from silberkorn import ki


@pytest.fixture(params=["numpy", "cupy"])
def xp(request):
    if request.param == "numpy":
        pytest.importorskip("scipy")
        return np
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")
    return cp


def nach_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


# --------------------------------------------------------------------------
# Leinwand und Arbeitsgroesse
# --------------------------------------------------------------------------

@pytest.mark.parametrize("breite, hoehe, verhaeltnis", [
    (4000, 3000, 16 / 9), (4000, 3000, 1.0), (3000, 4000, 4 / 5), (6000, 4000, 9 / 16),
    (1000, 1000, 3 / 2)])
def test_leinwand_im_seitenverhaeltnis(breite, hoehe, verhaeltnis):
    gross_b, gross_h, x, y = ki.leinwand_planen(breite, hoehe, verhaeltnis)
    assert gross_b / gross_h == pytest.approx(verhaeltnis, rel=2e-3)
    # Erweitert wird nur in einer Richtung, das Original liegt mittig und ganz darin
    assert gross_b == breite or gross_h == hoehe
    assert 0 <= x <= gross_b - breite and 0 <= y <= gross_h - hoehe
    assert abs((gross_b - breite - x) - x) <= 1 and abs((gross_h - hoehe - y) - y) <= 1


def test_leinwand_rundum_und_im_raster():
    gross_b, gross_h, x, y = ki.leinwand_planen(1000, 800, 1000 / 800, anteil=0.8, raster=4)
    assert gross_b == 1250 and gross_h == 1000
    assert x % 4 == 0 and y % 4 == 0 and x > 0 and y > 0


def test_gleiches_verhaeltnis_erweitert_nicht():
    assert ki.leinwand_planen(1200, 800, 3 / 2) == (1200, 800, 0, 0)


def test_grosser_rand():
    # 4:3 auf 16:9: je Seite rund 17 % der Breite dazu - noch kein Hinweis
    gross_b, gross_h, x, y = ki.leinwand_planen(4000, 3000, 16 / 9)
    assert not ki.grosser_rand(4000, 3000, gross_b, gross_h, x, y)
    # Hochformat 2:3 auf 16:9: je Seite fast die ganze Breite
    gross_b, gross_h, x, y = ki.leinwand_planen(2000, 3000, 16 / 9)
    assert ki.grosser_rand(2000, 3000, gross_b, gross_h, x, y)
    links, oben, rechts, unten = ki.rand_anteile(2000, 3000, gross_b, gross_h, x, y)
    assert oben == unten == 0 and links == pytest.approx(rechts, abs=1e-3)


@pytest.mark.parametrize("gross_b, gross_h", [
    (5333, 3000), (4000, 4000), (3000, 5333), (800, 600), (16000, 2000), (1024, 1024)])
def test_arbeitsgroesse(gross_b, gross_h):
    ab, ah = ki.arbeitsgroesse(gross_b, gross_h)
    assert ab % 16 == 0 and ah % 16 == 0
    assert ab * ah <= ki.OUTPAINT_FLAECHE
    assert ab * ah > 0.9 * ki.OUTPAINT_FLAECHE
    assert ab / ah == pytest.approx(gross_b / gross_h, rel=0.05)


def test_eingabe_bauen(xp):
    srgb = xp.full((300, 400, 3), 0.25, dtype=xp.float32)
    gross_b, gross_h, x, y = ki.leinwand_planen(400, 300, 16 / 9)
    ab, ah = ki.arbeitsgroesse(gross_b, gross_h)
    eingabe, (ox, oy, ob, oh) = ki.eingabe_bauen(srgb, gross_b, gross_h, x, y, ab, ah)
    eingabe = nach_numpy(eingabe)
    assert eingabe.shape == (ah, ab, 3) and eingabe.dtype == np.float32
    assert oy == 0 and oh == ah and ox > 0 and ox + ob < ab
    assert np.allclose(eingabe[oy:oy + oh, ox:ox + ob], 0.25, atol=1e-6)
    assert np.array_equal(eingabe[:, :ox], np.broadcast_to(ki.OUTPAINT_GRUEN, (ah, ox, 3)))
    assert np.array_equal(eingabe[:, ox + ob:, 1], np.ones((ah, ab - ox - ob)))


# --------------------------------------------------------------------------
# Latents, Positionen, Zeitplan
# --------------------------------------------------------------------------

def test_patchify_und_packen_umkehrbar(xp):
    latent = xp.asarray(np.random.default_rng(1).standard_normal((1, 32, 12, 18)),
                        dtype=xp.float32)
    gepatcht = ki.patchify(latent)
    assert gepatcht.shape == (1, 128, 6, 9)
    # Kanal 4c + 2i + j ist der Punkt (2y + i, 2x + j) von Kanal c
    assert float(gepatcht[0, 4 * 5 + 2 * 1 + 0, 2, 3]) == float(latent[0, 5, 5, 6])
    assert float(gepatcht[0, 4 * 7 + 2 * 0 + 1, 4, 8]) == float(latent[0, 7, 8, 17])
    assert xp.array_equal(ki.unpatchify(gepatcht), latent)
    tokens = ki.packen(gepatcht)
    assert tokens.shape == (1, 54, 128)
    assert xp.array_equal(tokens[0, 3 * 9 + 4], gepatcht[0, :, 3, 4])
    assert xp.array_equal(ki.entpacken(tokens, 6, 9), gepatcht)


def test_positionen(xp):
    ids = nach_numpy(ki.positionen(3, 4, 10, xp))
    assert ids.shape == (12, 4) and ids.dtype == np.float32
    assert ids[0].tolist() == [10, 0, 0, 0]
    assert ids[5].tolist() == [10, 1, 1, 0]
    assert ids[11].tolist() == [10, 2, 3, 0]
    text = nach_numpy(ki.text_positionen(512, xp))
    assert text.shape == (512, 4)
    assert text[:, :3].max() == 0 and text[:, 3].tolist() == list(range(512))


def test_empirischer_mu():
    # Ueber 4300 Tokens haengt mu nur von der Bildgroesse ab
    assert ki.empirischer_mu(4096 + 512, 4) == pytest.approx(0.00016927 * 4608 + 0.45666666)
    assert ki.empirischer_mu(5000, 4) == ki.empirischer_mu(5000, 50)
    # darunter laeuft es mit der Schrittzahl von der Gerade fuer 10 auf die fuer 200
    klein = ki.empirischer_mu(1000, 10)
    assert klein == pytest.approx(8.73809524e-05 * 1000 + 1.89833333)


@pytest.mark.parametrize("plan", [
    ki.Zeitplan(), ki.Zeitplan(exponentiell=False), ki.Zeitplan(dynamisch=False),
    ki.Zeitplan(ende=0.02)])
def test_sigmas(plan):
    sigmas = ki.sigmas_berechnen(4, ki.empirischer_mu(4096, 4), plan)
    assert sigmas.dtype == np.float32 and sigmas.shape == (5,)
    assert sigmas[-1] == 0
    assert np.all(np.diff(sigmas) < 0)
    if plan.ende:
        assert sigmas[-2] == pytest.approx(plan.ende, abs=1e-6)
    else:
        assert sigmas[0] == pytest.approx(1.0)


def test_sigmas_dynamisch_exponentiell():
    """exp(mu) / (exp(mu) + 1/t - 1) auf linspace(1, 1/4, 4)."""
    mu = 1.2
    t = np.linspace(1.0, 0.25, 4)
    soll = np.exp(mu) / (np.exp(mu) + (1 / t - 1))
    assert np.allclose(ki.sigmas_berechnen(4, mu, ki.Zeitplan())[:-1], soll, atol=1e-6)


def test_euler_trifft_bei_gerader_bahn():
    """Flow Matching: x_t = (1 - t) x0 + t e, v = e - x0. Mit der wahren Geschwindigkeit
    landen die Euler-Schritte von reinem Rauschen genau auf x0."""
    rng = np.random.default_rng(2)
    x0, e = rng.standard_normal((2, 1, 10, 128)).astype(np.float32)
    sigmas = ki.sigmas_berechnen(4, 1.5, ki.Zeitplan())
    x = e
    for i in range(4):
        x = ki.euler_schritt(x, e - x0, sigmas[i], sigmas[i + 1])
    assert np.allclose(x, x0, atol=1e-5)


def test_rauschen_je_seed():
    a = ki.outpaint_rauschen(7, 64, 128)
    assert a.shape == (1, 64, 128) and a.dtype == np.float32
    assert np.array_equal(a, ki.outpaint_rauschen(7, 64, 128))
    assert not np.array_equal(a, ki.outpaint_rauschen(8, 64, 128))


def test_daten_laden(tmp_path):
    pfad = tmp_path / "daten.npz"
    np.savez(pfad, prompt=np.ones((1, 5, 8), np.float32), bn_mittel=np.arange(128.0),
             bn_std=np.full(128, 2.0), dynamisch=True, exponentiell=False, verschiebung=3.0,
             ende=0.0)
    daten = ki.outpaint_daten_laden(str(pfad))
    assert daten.prompt.shape == (1, 5, 8) and daten.mittel.shape == (1, 128, 1, 1)
    assert daten.streuung.dtype == np.float32 and float(daten.mittel[0, 3, 0, 0]) == 3.0
    assert daten.zeitplan == ki.Zeitplan(True, False, 3.0, 0.0)


def test_ablauf_mit_platzhalter_netzen(xp):
    """outpaint_rechnen gibt den Netzen, was Flux2KleinPipeline ihnen gibt: normierte
    Referenz-Tokens hinter den eigenen, passende IDs, Zeit = sigma - und rechnet nur
    mit den eigenen Tokens weiter."""
    ah, ab = 64, 96
    h, w = ah // 16, ab // 16
    n = h * w
    rng = np.random.default_rng(3)
    mittel = xp.asarray(rng.standard_normal((1, 128, 1, 1)), dtype=xp.float32)
    streuung = xp.asarray(rng.uniform(0.5, 2, (1, 128, 1, 1)), dtype=xp.float32)
    daten = ki.OutpaintDaten(xp.zeros((1, 7, 16), dtype=xp.float32), mittel, streuung,
                             ki.Zeitplan())
    roh = xp.asarray(rng.standard_normal((1, 32, ah // 8, ab // 8)), dtype=xp.float32)
    rauschen = ki.outpaint_rauschen(5, n, 128, xp)
    aufrufe = []

    def kodieren(bild):
        assert bild.shape == (1, 3, ah, ab)
        return roh

    def transformieren(e):
        aufrufe.append({k: nach_numpy(v).copy() for k, v in e.items()})
        assert e["latents"].shape == (1, 2 * n, 128) and e["bild_ids"].shape == (2 * n, 4)
        assert e["text_ids"].shape == (7, 4)
        return e["latents"] * 2          # v = 2x: x waechst je Schritt um (1 + 2 dsigma)

    def dekodieren(latent):
        aufrufe.append(nach_numpy(latent).copy())
        return xp.full((1, 3, ah, ab), 0.5, dtype=xp.float32)

    zaehler = []
    ergebnis = ki.outpaint_rechnen(xp.zeros((ah, ab, 3), dtype=xp.float32), rauschen, daten,
                                   kodieren, transformieren, dekodieren,
                                   lambda: zaehler.append(1))
    assert ergebnis.shape == (ah, ab, 3) and float(ergebnis.mean()) == pytest.approx(0.5)
    assert len(zaehler) == ki.OUTPAINT_SCHRITTE + 2
    erster, latent = aufrufe[0], aufrufe[-1]
    referenz = nach_numpy(ki.packen((ki.patchify(roh) - mittel) / streuung))
    assert np.allclose(erster["latents"][:, n:], referenz, atol=1e-6)
    assert np.array_equal(erster["latents"][:, :n], nach_numpy(rauschen))
    assert np.all(erster["bild_ids"][:n, 0] == 0) and np.all(erster["bild_ids"][n:, 0] == 10)
    sigmas = ki.sigmas_berechnen(ki.OUTPAINT_SCHRITTE, ki.empirischer_mu(n, 4), ki.Zeitplan())
    assert [float(a["zeit"][0]) for a in aufrufe[:-1]] == pytest.approx(list(sigmas[:-1]))
    faktor = math.prod(1 + 2 * float(sigmas[i + 1] - sigmas[i]) for i in range(4))
    soll = ki.unpatchify(ki.entpacken(rauschen * faktor, h, w) * streuung + mittel)
    assert latent.shape == (1, 32, ah // 8, ab // 8)
    assert np.allclose(latent, nach_numpy(soll), rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------
# Farbe angleichen und einsetzen
# --------------------------------------------------------------------------

def _szene(hoehe, breite):
    y, x = np.mgrid[0:hoehe, 0:breite].astype(np.float32)
    return np.stack([0.2 + 0.6 * x / breite, 0.3 + 0.4 * y / hoehe,
                     0.5 + 0.2 * np.sin(x / 9) * np.cos(y / 13)], axis=-1)


def test_farbstich_wird_herausgenommen(xp):
    """Das Modell liefert die Szene mit Farbstich (je Kanal anders) - im Original und
    in den neuen Raendern. Danach stimmt das Bild ueberall wieder mit der Szene."""
    wahr = _szene(128, 224)
    ox, oy, ob, oh = 48, 0, 128, 128
    stich = np.asarray([0.85, 1.05, 0.9], np.float32), np.asarray([0.08, -0.03, 0.05])
    roh = np.clip(wahr * stich[0] + stich[1], 0, 1).astype(np.float32)
    orig = wahr[oy:oy + oh, ox:ox + ob]
    vorher = np.abs(roh - wahr).mean()
    angeglichen = nach_numpy(ki.farbe_angleichen(xp.asarray(roh), xp.asarray(orig), ox, oy))
    assert angeglichen.shape == roh.shape and angeglichen.dtype == np.float32
    assert np.abs(angeglichen - wahr).mean() < vorher / 20
    assert np.abs(angeglichen - wahr).max() < 0.01


def test_oertliche_abweichung_laeuft_aus(xp):
    """Liegt das Modell nur in einer Ecke des Originals daneben, wird das dort
    ausgeglichen und laeuft in die Raender aus, statt als Kante stehen zu bleiben."""
    wahr = _szene(96, 160)
    ox, oy, ob = 32, 0, 96
    roh = wahr.copy()
    roh[:30, 32:60] += 0.1                     # Ecke oben links im Original zu hell
    roh[:30, :32] += 0.1                       # und der Rand daneben ebenso
    angeglichen = nach_numpy(ki.farbe_angleichen(xp.asarray(roh),
                                                 xp.asarray(wahr[:, ox:ox + ob]), ox, oy))
    assert np.abs(angeglichen[:30, 32:60] - wahr[:30, 32:60]).mean() < 0.05
    assert np.abs(angeglichen[:30, :32] - wahr[:30, :32]).mean() < 0.06
    # Weit weg von der Abweichung bleibt alles, wie es war
    assert np.abs(angeglichen[80:, 140:] - wahr[80:, 140:]).max() < 0.02


def _saum_nach_edt(gross_b, gross_h, x, y, breite, hoehe, saum):
    """Wie outpaint_referenz.einsetzen: Abstandstransformation, Leinwandrand ohne Naht."""
    from scipy import ndimage
    innen = np.zeros((gross_h, gross_b), bool)
    innen[y:y + hoehe, x:x + breite] = True
    rand = int(saum) + 1
    d = ndimage.distance_transform_edt(np.pad(innen, rand, mode="edge"))[rand:-rand, rand:-rand]
    return np.clip(d / saum, 0, 1)[y:y + hoehe, x:x + breite]


@pytest.mark.parametrize("lage", [
    (300, 200, 50, 0, 200, 200), (200, 300, 0, 50, 200, 200), (260, 260, 30, 30, 200, 200),
    (240, 200, 40, 0, 200, 200)])
def test_saum_wie_abstandstransformation(lage):
    pytest.importorskip("scipy")
    gross_b, gross_h, x, y, breite, hoehe = lage
    soll = _saum_nach_edt(*lage, 12.0)
    ist = ki.saum_deckkraft(gross_b, gross_h, x, y, breite, hoehe, 12.0)
    assert ist.shape == (hoehe, breite) and ist.dtype == np.float32
    assert np.allclose(ist, soll, atol=1e-6)


def test_saum_nur_an_erweiterten_seiten(xp):
    deckkraft = nach_numpy(ki.saum_deckkraft(300, 200, 50, 0, 200, 200, 10.0, xp))
    assert deckkraft[0, 100] == 1 and deckkraft[-1, 100] == 1        # oben, unten: Leinwandrand
    assert deckkraft[100, 0] == pytest.approx(0.1)                     # links: Naht
    assert deckkraft[100, -1] == pytest.approx(0.1)
    assert deckkraft[100, 9] == 1 and deckkraft[100, 100] == 1
    assert ki.saum_breite(4000, 3000) == 45 and ki.saum_breite(100, 100) == 4


def test_einsetzen(xp):
    gross = xp.zeros((10, 20, 3), dtype=xp.float32)
    orig = xp.ones((10, 8, 3), dtype=xp.float32)
    deckkraft = xp.ones((10, 8), dtype=xp.float32)
    deckkraft[:, 0] = 0.25
    ergebnis = nach_numpy(ki.einsetzen(gross, orig, 6, 0, deckkraft))
    assert float(gross.max()) == 0                                     # Eingabe unberuehrt
    assert np.all(ergebnis[:, 7:14] == 1) and np.all(ergebnis[:, 6] == 0.25)
    assert np.all(ergebnis[:, :6] == 0) and np.all(ergebnis[:, 14:] == 0)


# --------------------------------------------------------------------------
# Katalog
# --------------------------------------------------------------------------

def test_modell_im_katalog():
    modell = ki.ERWEITER_MODELLE["outpaint"]
    assert ki.ALLE_MODELLE["outpaint"] is modell
    assert modell.release == ki.MODELL_RELEASE_8
    namen = ki.dateien(modell)
    assert namen[0] == ki.OUTPAINT_TRANSFORMER and namen[-1] == "LICENSE-FLUX2-klein.txt"
    for name in (*ki.OUTPAINT_GEWICHTE, ki.OUTPAINT_KODIERER, ki.OUTPAINT_DEKODIERER,
                 ki.OUTPAINT_DATEN, ki.OUTPAINT_NOTICE):
        assert name in namen
    # GitHub nimmt je Release-Datei hoechstens 2 GiB an
    assert all(ki.DATEIEN[n][1] < 2**31 for n in namen)
    assert ki.download_groesse(modell) > 4e9 or ki.vorhanden(modell)


def test_mindestens_8_gb():
    modell = ki.ERWEITER_MODELLE["outpaint"]
    assert not ki.angeboten(modell, "S") and ki.angeboten(modell, "M")
    assert ki.genug_vram(modell, int(7.6 * 2**30))                    # 8-GB-Karte
    assert not ki.genug_vram(modell, int(5.8 * 2**30))                # 6-GB-Karte, auch Stufe M
    assert ki.genug_vram(ki.ENTFERN_MODELLE["lama"], 0)


def test_lizenztexte_wie_im_projekt():
    """Lizenz und NOTICE, die mit dem Modell geladen werden, liegen in LICENSES/."""
    import hashlib
    import pathlib
    lizenzen = pathlib.Path(__file__).resolve().parent.parent / "LICENSES"
    for datei, name in (("Apache-2.0.txt", "LICENSE-FLUX2-klein.txt"),
                        ("NOTICE-FLUX2-klein.txt", ki.OUTPAINT_NOTICE)):
        inhalt = (lizenzen / datei).read_bytes()
        assert ki.DATEIEN[name] == (hashlib.sha256(inhalt).hexdigest(), len(inhalt))


def test_erweitern_mit_modell():
    """Nur mit Grafikkarte und geladenem Modell: Seitenverhaeltnis, Original unveraendert."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")
    modell = ki.ERWEITER_MODELLE["outpaint"]
    if not ki.vorhanden(modell) or modell.sha256 == ki.OFFEN:
        pytest.skip("Modell nicht geladen")
    srgb = cp.asarray(_szene(600, 800))
    ergebnis = ki.Erweiterer(modell).erweitern(srgb, 16 / 9, seed=1)
    gross_b, gross_h, x, y = ki.leinwand_planen(800, 600, 16 / 9)
    assert ergebnis.shape == (gross_h, gross_b, 3)
    saum = int(ki.saum_breite(800, 600))
    innen = ergebnis[y:y + 600, x + saum:x + 800 - saum]
    assert float(cp.abs(innen - srgb[:, saum:800 - saum]).max()) < 1e-5
    assert float(ergebnis[:, :x, 1].mean()) < 0.9                      # kein Gruen geblieben
