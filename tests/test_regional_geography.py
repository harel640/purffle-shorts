"""P3: a Level 3-4 area is matched by geography, not only by the one region name a place arrives with.

Fort Pilar in Zamboanga City is in Mindanao (a named Level 3-4 area), but its list region was "Zamboanga City"
and an LLM calls it "Zamboanga Peninsula" - neither contains "mindanao", so it passed as "Philippines, Level 2,
ok". Every warned area now has its provinces and towns as names, and reference towns / border lines as
geography (advisory_areas.py, advisory_area_data.py)."""

import pytest

from purffle_shorts import advisory_areas as G
from purffle_shorts import pipeline as P
from purffle_shorts import safety
from purffle_shorts import topics as T
from purffle_shorts.config import Settings

FEED = {"india": 2, "philippines": 2, "egypt": 3, "mexico": 2, "colombia": 2, "japan": 1}


@pytest.fixture(autouse=True)
def _feed(monkeypatch):
    monkeypatch.setattr(safety, "advisories", lambda: FEED)


def v(country, region="", place="", lat=None, lon=None):
    return safety.verdict_for(safety.country_names(country), region, place=place, lat=lat, lon=lon)


# ------------------------------------------------------------------ Fort Pilar, the case that got through
FORT_PILAR = (6.904, 122.077)


@pytest.mark.parametrize("region, place, coords", [
    ("Zamboanga City", "Fort Pilar", None),          # the Scout list's region
    ("Zamboanga Peninsula", "", None),                # what the LLM answers
    ("", "Fort Pilar", FORT_PILAR),                   # coordinates alone
    ("Region IX", "Fort Pilar", FORT_PILAR),          # a region name the table has never heard of
])
def test_fort_pilar_is_blocked_however_it_arrives(region, place, coords):
    lat, lon = coords or (None, None)
    r = v("Philippines", region, place, lat, lon)
    assert r.unsafe and r.level == 4


@pytest.mark.parametrize("name, region, lat, lon", [
    ("Zamboanga City", "Zamboanga Peninsula", 6.9214, 122.079),
    ("Davao City", "Davao Region", 7.0731, 125.6128),
    ("Cotabato City", "Bangsamoro", 7.2236, 124.2464),
    ("Jolo", "Sulu", 6.0522, 121.0022),
])
def test_mindanao_and_sulu_blocked_by_name_and_by_coordinates(name, region, lat, lon):
    assert v("Philippines", region, name).unsafe                         # by name
    assert v("Philippines", "Somewhere", "A Place", lat, lon).unsafe     # by coordinates alone


@pytest.mark.parametrize("name, region, lat, lon", [
    ("Mactan Shrine", "Central Visayas", 10.3103, 124.0153),
    ("Manila", "Metro Manila", 14.5958, 120.9772),
    ("Vigan", "Ilocos Region", 17.5747, 120.3869),
])
def test_visayas_and_luzon_stay_open(name, region, lat, lon):
    assert not v("Philippines", region, name, lat, lon).unsafe


def test_india_named_areas_and_the_border_buffer():
    assert v("India", "Jammu and Kashmir", "Srinagar", 34.09, 74.79).unsafe
    assert v("India", "Manipur", "Imphal", 24.817, 93.9368).unsafe
    assert v("India", "", "Srinagar").unsafe                              # by name, no region
    wagah = v("India", "Punjab", "Border crossing", 31.6047, 74.5731)     # no warned name: the buffer catches it
    assert wagah.unsafe and "India-Pakistan border" in wagah.reason
    assert not v("India", "Punjab", "Golden Temple", 31.62, 74.8765).unsafe    # Amritsar, ~25 km away
    assert not v("India", "Rajasthan", "Jaipur", 26.9124, 75.7873).unsafe
    assert not v("India", "Karnataka", "Hampi", 15.335, 76.462).unsafe


def test_unknown_region_without_coordinates_still_fails_closed():
    r = v("India")
    assert r.unsafe and "neither the region nor the coordinates" in r.reason
    assert not v("India", "", "", 15.335, 76.462).unsafe                  # coordinates are enough to clear it


def test_names_that_exist_twice_are_left_to_the_coordinates():
    # Bishnupur is also the West Bengal temple town, Aurangabad also the gateway to Ajanta: not keywords.
    assert not v("India", "West Bengal", "Bishnupur", 23.0747, 87.3175).unsafe
    assert not v("India", "Maharashtra", "Aurangabad", 19.88, 75.32).unsafe
    assert v("India", "", "Bishnupur", 24.6297, 93.7594).unsafe           # the Manipur one, by coordinates
    assert not v("Colombia", "Valle del Cauca", "Cali", 3.4206, -76.5222).unsafe   # "Cauca" in another name


def test_keywords_match_whole_words_only():
    a = G.Area("Test", 4, ("durg", "gaya"))
    assert a.named_in("Durg, Chhattisgarh") and a.named_in("Bodh Gaya")
    assert not a.named_in("Durgapur") and not a.named_in("Durga temple") and not a.named_in("Gayatri")


def test_geometry_primitives():
    square = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
    hull = G._hull(square + [(0.5, 0.5)])
    assert len(hull) == 4
    assert G.distance_to_hull_km((0.5, 0.5), hull) == 0
    assert 105 < G.distance_to_hull_km((2.0, 0.5), hull) < 115          # ~1 degree of latitude
    assert 10 < G.distance_to_line_km((0.1, 0.5), [(0.0, 0.0), (0.0, 1.0)]) < 12


def test_every_area_can_be_located():
    for country, areas in safety.REGIONAL_WARNINGS.items():
        for a in map(G.as_area, areas):
            assert a.keywords, a.name
            assert a.has_geometry or a.name.startswith("Maoist"), f"{country}: {a.name} has no geography"
            assert all(len(g) >= 2 for g in a.groups), a.name


def test_old_style_entries_still_work(monkeypatch):
    monkeypatch.setattr(safety, "REGIONAL_WARNINGS", {"india": [("jammu", 4)]})
    assert v("India", "Jammu").unsafe and not v("India", "Kerala").unsafe


# ------------------------------------------------------------------ the pipeline hands over the coordinates
class RegionLLM:
    def __init__(self, country, region):
        self.reply = {"country": country, "aliases": [], "region": region}

    def complete_json(self, *a, **k):
        return self.reply


def _studio(tmp_path, llm=None):
    st = P.Studio(Settings(output_dir=str(tmp_path / "o"), data_dir=str(tmp_path / "d"), upload=False,
                           visual_sources=["wikimedia"], safety_filter=True))
    st._llm = llm
    return st


def test_a_listed_place_is_checked_by_its_coordinates(tmp_path):
    pick = T.Topic("Fort Pilar, Philippines", "list", key="list:x:fort pilar|philippines", place="Fort Pilar",
                   country="Philippines", region="", lat=FORT_PILAR[0], lon=FORT_PILAR[1])
    assert _studio(tmp_path)._advisory(pick).unsafe
    shrine = T.Topic("Mactan Shrine, Philippines", "list", key="k", place="Mactan Shrine", country="Philippines",
                     region="Central Visayas", lat=10.3103, lon=124.0153)
    assert not _studio(tmp_path)._advisory(shrine).unsafe


def test_the_resolved_place_is_checked_by_its_coordinates(tmp_path):
    st = _studio(tmp_path, RegionLLM("Philippines", "Western Visayas"))     # a wrong region from the LLM
    pick = T.Topic("hidden gems in Asia", "niche")
    assert st._advisory(pick, place="Old Fort", country="Philippines", coords=FORT_PILAR).unsafe
    assert not st._advisory(pick, place="Old Fort", country="Philippines", coords=(10.7, 122.56)).unsafe


# ------------------------------------------------------------------ regression: real places, Wikipedia coordinates
# (country, place, lat, lon). Built with the areas on 2026-11-11. MUST_BLOCK rows are in or within a few km of a
# warned area; MUST_ALLOW rows are outside - many of them neighbours of one (Meghalaya, Sikkim, Tripura, West
# Bengal, Ladakh, Himachal, Kutch, the Visayas, Palawan).
MUST_BLOCK = [
    ('india', 'Srinagar', 34.09, 74.79),
    ('india', 'Gulmarg', 34.05, 74.38),
    ('india', 'Pahalgam', 34.01, 75.19),
    ('india', 'Jammu', 32.73, 74.87),
    ('india', 'Vaishno Devi', 33.0299, 74.9482),
    ('india', 'Dal Lake', 34.12, 74.87),
    ('india', 'Imphal', 24.8074, 93.9384),
    ('india', 'Loktak Lake', 24.55, 93.7833),
    ('india', 'Moirang', 24.5, 93.77),
    ('india', 'Wagah', 31.6047, 74.5731),
    ('india', 'Attari', 31.6008, 74.6056),
    ('india', 'Hussainiwala', 30.9977, 74.5471),
    ('india', 'Munabao', 25.75, 70.283),
    ('india', 'Raipur', 21.2444, 81.6306),
    ('india', 'Chitrakote Falls', 19.2065, 81.7),
    ('india', 'Jagdalpur', 19.18, 81.92),
    ('india', 'Ranchi', 23.36, 85.33),
    ('india', 'Netarhat', 23.4753, 84.2672),
    ('india', 'Patna', 25.5939, 85.1375),
    ('india', 'Bodh Gaya', 24.695, 84.9925),
    ('india', 'Nalanda', 25.1366, 85.4426),
    ('india', 'Vikramashila', 25.3247, 87.2853),
    ('india', 'Puri', 19.81, 85.83),
    ('india', 'Konark Sun Temple', 19.8875, 86.0947),
    ('india', 'Chilika Lake', 19.7167, 85.3167),
    ('india', 'Simlipal National Park', 21.8333, 86.3333),
    ('india', 'Tawang Monastery', 27.5864, 91.8575),
    ('india', 'Ziro', 27.5417, 93.8161),
    ('india', 'Kohima', 25.67, 94.1),
    ('india', 'Dzüko Valley', 25.5533, 94.0671),
    ('india', 'Aizawl', 23.733, 92.718),
    ('india', 'Majuli', 27, 94.2167),
    ('india', 'Kaziranga National Park', 26.6667, 93.35),
    ('india', 'Haflong', 25.169, 93.016),
    ('india', 'Silchar', 24.82, 92.8),
    ('india', 'Manas National Park', 26.7167, 90.9333),
    ('india', 'Sualkuchi', 26.17, 91.57),
    ('india', 'Hornbill Festival', 25.6275, 94.1094),
    ('india', 'Rajgir', 25.03, 85.42),
    ('india', 'Vaishali (ancient city)', 26.0137, 85.109),
    ('india', 'Barabar Caves', 25.005, 85.063),
    ('india', 'Bhitarkanika National Park', 20.75, 87),
    ('india', 'Daringbadi', 19.9072, 84.1228),
    ('india', 'Hirakud Dam', 21.57, 83.87),
    ('india', 'Dassam Falls', 23.1434, 85.4664),
    ('india', 'Maluti', 24.1611, 87.6747),
    ('india', 'Sirpur', 21.3452, 82.1848),
    ('india', 'Mainpat', 22.837, 83.315),
    ('india', 'Phawngpui', 22.6315, 93.0388),
    ('india', 'Namdapha National Park', 27.4833, 96.3833),
    ('india', 'Mechuka', 28.605, 94.1281),
    ('india', 'Sivasagar', 26.9889, 94.639),
    ('india', 'Umananda Island', 26.1964, 91.745),
    ('philippines', 'Fort Pilar', 6.9011, 122.0822),
    ('philippines', 'Zamboanga City', 6.9042, 122.0761),
    ('philippines', 'Dapitan', 8.6549, 123.4243),
    ('philippines', 'Davao City', 7.0662, 125.6094),
    ('philippines', 'Island Garden City of Samal', 7.0783, 125.7117),
    ('philippines', 'Cotabato City', 7.22, 124.25),
    ('philippines', 'Jolo, Sulu', 6.05, 121),
    ('philippines', 'Marawi', 8.0031, 124.285),
    ('philippines', 'Lake Sebu', 6.2292, 124.7058),
    ('philippines', 'Mambajao', 9.25, 124.72),
    ('philippines', 'General Luna, Surigao del Norte', 9.783, 126.156),
    ('philippines', 'San Jose, Dinagat Islands', 10.0083, 125.5889),
    ('philippines', 'Mount Apo', 6.9875, 125.2708),
    ('philippines', 'Lake Lanao', 7.88, 124.2525),
    ('philippines', 'Hinatuan Enchanted River', 8.3696, 126.3384),
    ('philippines', 'Tinuy-an Falls', 8.1711, 126.2294),
    ('philippines', 'Bongao', 5.0292, 119.7731),
    ('philippines', 'Turtle Islands, Tawi-Tawi', 6.073, 118.312),
    ('philippines', 'Dahican Beach', 6.9183, 126.27),
    ('philippines', 'Mount Hamiguitan', 6.74, 126.1817),
    ('philippines', 'Rizal Shrine (Dapitan)', 8.6672, 123.4167),
    ('philippines', 'Sohoton Cove', 9.6736, 125.9483),
    ('philippines', 'Britania Islands', 8.85, 126.38),
    ('philippines', 'Lamitan', 6.6582, 122.1457),
    ('philippines', 'Iligan', 8.23, 124.25),
    ('philippines', 'Maria Cristina Falls', 8.1832, 124.195),
    ('philippines', 'Lake Holon', 6.1133, 124.8917),
    ('philippines', 'Mount Kitanglad', 8.1428, 124.9125),
    ('philippines', 'Cagayan de Oro', 8.48, 124.65),
    ('philippines', 'Butuan', 8.948, 125.543),
    ('egypt', 'Sharm El Sheikh', 27.915, 34.3275),
    ('egypt', 'Dahab', 28.4931, 34.5047),
    ('egypt', "Saint Catherine's Monastery", 28.5556, 33.9761),
    ('egypt', 'Taba, Egypt', 29.4917, 34.8944),
    ('egypt', 'Mount Sinai', 28.5394, 33.9754),
    ('egypt', 'Arish', 31.1321, 33.8034),
    ('mexico', 'Acapulco', 16.8636, -99.8825),
    ('mexico', 'Taxco', 18.5564, -99.605),
    ('mexico', 'Morelia', 19.7683, -101.1894),
    ('mexico', 'Pátzcuaro', 19.5164, -101.6097),
    ('mexico', 'Mazatlán', 23.2167, -106.4167),
    ('mexico', 'Culiacán', 24.8069, -107.3939),
    ('mexico', 'Tampico', 22.2553, -97.8686),
    ('mexico', 'Zacatecas City', 22.7736, -102.5736),
    ('mexico', 'Manzanillo, Colima', 19.0522, -104.3158),
    ('mexico', 'Zihuatanejo', 17.6444, -101.5511),
    ('mexico', 'Monarch Butterfly Biosphere Reserve', 19.6064, -100.2417),
    ('colombia', 'Popayán', 2.4542, -76.6092),
    ('colombia', 'Cúcuta', 7.8942, -72.5039),
    ('colombia', 'Arauca, Arauca', 7.0903, -70.7617),
    ('colombia', 'Tierradentro', 2.5833, -76.0333),
]
MUST_ALLOW = [
    ('india', 'Jaipur', 26.915, 75.82),
    ('india', 'Hampi', 15.3344, 76.4622),
    ('india', 'Amritsar', 31.64, 74.86),
    ('india', 'Leh', 34.1529, 77.5773),
    ('india', 'Padum', 33.4639, 76.8786),
    ('india', 'Chamba, Himachal Pradesh', 32.1482, 76.4186),
    ('india', 'Dalhousie, India', 32.53, 75.98),
    ('india', 'Shillong', 25.5822, 91.8944),
    ('india', 'Sohra', 25.284, 91.721),
    ('india', 'Mawlynnong', 25.2017, 91.9161),
    ('india', 'Dawki', 25.1833, 92.0167),
    ('india', 'Tura, Meghalaya', 25.52, 90.22),
    ('india', 'Gangtok', 27.33, 88.62),
    ('india', 'Darjeeling', 27.0375, 88.2631),
    ('india', 'Agartala', 23.8314, 91.2869),
    ('india', 'Kalimpong', 27.06, 88.47),
    ('india', 'Varanasi', 25.3189, 83.0128),
    ('india', 'Kushinagar', 26.741, 83.888),
    ('india', 'Kolkata', 22.5675, 88.37),
    ('india', 'Shimla', 31.1033, 77.1722),
    ('india', 'Manali, Himachal Pradesh', 32.2432, 77.1892),
    ('india', 'Kaza, India', 32.2256, 78.0714),
    ('india', 'Jaisalmer', 26.913, 70.915),
    ('india', 'Kuldhara', 26.87, 70.785),
    ('india', 'Bikaner', 28.017, 73.312),
    ('india', 'Barmer', 25.75, 71.38),
    ('india', 'Bhuj', 23.25, 69.67),
    ('india', 'Dholavira', 23.8886, 70.2136),
    ('india', 'Lucknow', 26.85, 80.95),
    ('india', 'Hyderabad', 17.3617, 78.4747),
    ('india', 'Visakhapatnam', 17.6939, 83.2922),
    ('india', 'Khajuraho', 24.85, 79.925),
    ('india', 'Sanchi', 23.4792, 77.7397),
    ('india', 'Agra', 27.18, 78.02),
    ('india', 'Bishnupur, Bankura', 23.075, 87.317),
    ('india', 'Shantiniketan', 23.68, 87.68),
    ('india', 'Murshidabad', 24.18, 88.27),
    ('india', 'Unakoti', 24.3167, 92.0667),
    ('india', 'Neermahal', 23.84, 91.28),
    ('india', 'Jowai', 25.4422, 92.1947),
    ('india', 'Umiam Lake', 25.6532, 91.8843),
    ('india', 'Nohkalikai Falls', 25.2756, 91.6867),
    ('india', 'Pelling', 27.18, 88.14),
    ('india', 'Yuksom', 27.3733, 88.2208),
    ('india', 'Spiti Valley', 32.2469, 78.0522),
    ('india', 'Bhimbetka rock shelters', 22.9384, 77.6131),
    ('india', 'Mandu, Madhya Pradesh', 22.3353, 75.4158),
    ('india', 'Orchha', 25.35, 78.64),
    ('india', 'Gwalior Fort', 26.2303, 78.1689),
    ('india', 'Rani ki vav', 23.8589, 72.1017),
    ('india', 'Lonar Lake', 19.975, 76.5075),
    ('india', 'Ajanta Caves', 20.5533, 75.7003),
    ('india', 'Araku Valley', 18.3333, 82.8667),
    ('india', 'Kumbhalgarh', 25.1489, 73.5803),
    ('india', 'Chittorgarh', 24.88, 74.63),
    ('india', 'Bundi', 25.44, 75.64),
    ('india', 'Pushkar', 26.4877, 74.5559),
    ('india', 'Hemis Monastery', 33.9125, 77.7028),
    ('india', 'Lamayuru Monastery', 34.2828, 76.7744),
    ('india', 'Nubra Valley', 34.6, 77.7),
    ('india', 'Pangong Tso', 33.7179, 78.8968),
    ('india', 'Tso Moriri', 32.9, 78.3),
    ('india', 'Dharamshala', 32.2153, 76.3186),
    ('india', 'Kangra Fort', 32.0881, 76.2578),
    ('india', 'Pathankot', 32.2694, 75.6519),
    ('india', 'Gurdaspur', 32.0333, 75.4),
    ('india', 'Malana', 32.0628, 77.2606),
    ('india', 'Kheerganga', 32.0097, 77.3153),
    ('india', 'Rishikesh', 30.1083, 78.2972),
    ('india', 'Mussoorie', 30.45, 78.08),
    ('india', 'Valley of Flowers National Park', 30.73, 79.6175),
    ('india', 'Kedarnath Temple', 30.7352, 79.0669),
    ('philippines', 'Mactan Shrine', 10.3111, 124.0153),
    ('philippines', 'Cebu City', 10.293, 123.902),
    ('philippines', 'Manila', 14.5958, 120.9772),
    ('philippines', 'Vigan', 17.5747, 120.3869),
    ('philippines', 'Dumaguete', 9.3103, 123.3081),
    ('philippines', 'Siquijor', 9.18, 123.58),
    ('philippines', 'Panglao', 9.5667, 123.775),
    ('philippines', 'Chocolate Hills', 9.9167, 124.1667),
    ('philippines', 'Tagbilaran', 9.65, 123.85),
    ('philippines', 'Puerto Princesa', 9.74, 118.744),
    ('philippines', 'El Nido, Palawan', 11.18, 119.39),
    ('philippines', 'Coron, Palawan', 11.9989, 120.2053),
    ('philippines', 'Banaue Rice Terraces', 16.9, 121.05),
    ('philippines', 'Sagada', 17.0842, 120.9008),
    ('philippines', 'Boracay', 11.9689, 121.9239),
    ('philippines', 'Iloilo City', 10.7, 122.57),
    ('philippines', 'Bacolod', 10.6765, 122.9509),
    ('philippines', 'Tacloban', 11.24, 125),
    ('philippines', 'Kalanggaman Island', 11.1146, 124.2508),
    ('philippines', 'Maasin, Southern Leyte', 10.13, 124.85),
    ('philippines', 'Balabac, Palawan', 7.986, 117.065),
    ('philippines', "Brooke's Point", 8.78, 117.83),
    ('philippines', 'Sohoton Natural Bridge National Park', 11.3583, 125.1584),
    ('philippines', 'Apo Island', 9.0667, 123.2667),
    ('philippines', 'Malapascua', 11.3347, 124.1161),
    ('philippines', 'Bantayan Island', 11.1625, 123.7847),
    ('philippines', 'San Ricardo, Southern Leyte', 9.92, 125.28),
    ('philippines', 'Limasawa', 9.9078, 125.075),
    ('philippines', 'Guiuan', 11.03, 125.73),
    ('philippines', 'Anda, Bohol', 9.744, 124.576),
    ('philippines', 'Moalboal', 9.95, 123.4),
    ('philippines', 'Oslob', 9.55, 123.4),
    ('philippines', 'Tubbataha Reef', 8.9533, 119.8675),
    ('philippines', 'Puerto Princesa Subterranean River National Park', 10.1667, 118.9167),
    ('philippines', 'Batanes', 20.58, 121.9),
    ('philippines', 'Corregidor', 14.3856, 120.5731),
    ('egypt', 'Cairo', 30.0444, 31.2358),
    ('egypt', 'Luxor', 25.6967, 32.6444),
    ('egypt', 'Hurghada', 27.2578, 33.8117),
    ('egypt', 'Giza pyramid complex', 29.9761, 31.1328),
    ('egypt', 'Siwa Oasis', 29.2053, 25.5194),
    ('egypt', 'Aswan', 24.0889, 32.8997),
    ('egypt', 'Alexandria', 31.1975, 29.8925),
    ('egypt', 'Ismailia', 30.5833, 32.2667),
    ('egypt', 'Abu Simbel', 22.3369, 31.6256),
    ('egypt', 'Marsa Alam', 25.0667, 34.9),
    ('egypt', 'El Gouna', 27.385, 33.6792),
    ('mexico', 'Mexico City', 19.4333, -99.1333),
    ('mexico', 'Guadalajara', 20.6767, -103.3475),
    ('mexico', 'Puerto Vallarta', 20.6458, -105.2222),
    ('mexico', 'Oaxaca City', 17.0606, -96.7253),
    ('mexico', 'Monterrey', 25.6844, -100.3181),
    ('mexico', 'Cuernavaca', 18.9186, -99.2342),
    ('mexico', 'Guanajuato City', 21.0178, -101.2567),
    ('mexico', 'San Miguel de Allende', 20.9142, -100.7436),
    ('mexico', 'Tequila, Jalisco', 20.8828, -103.8367),
    ('mexico', 'Durango City', 24.025, -104.6675),
    ('mexico', 'Aguascalientes City', 21.876, -102.296),
    ('mexico', 'Querétaro City', 20.5875, -100.3928),
    ('mexico', 'Puebla City', 19.0333, -98.1833),
    ('mexico', 'Chihuahua City', 28.6369, -106.0769),
    ('mexico', 'Hermosillo', 29.0989, -110.9542),
    ('mexico', 'Saltillo', 25.4231, -100.9919),
    ('mexico', 'San Luis Potosí City', 22.1511, -100.9761),
    ('mexico', 'Tepic', 21.5083, -104.8931),
    ('mexico', 'Real de Catorce', 23.6897, -100.8875),
    ('mexico', 'Xilitla', 21.384, -98.99),
    ('colombia', 'Bogotá', 4.7111, -74.0722),
    ('colombia', 'Cali', 3.4206, -76.5222),
    ('colombia', 'Medellín', 6.2308, -75.5906),
    ('colombia', 'Cartagena, Colombia', 10.4, -75.5),
    ('colombia', 'Bucaramanga', 7.1333, -73),
    ('colombia', 'Pasto, Colombia', 1.2078, -77.2772),
    ('colombia', 'Villavicencio', 4.15, -73.6333),
    ('colombia', 'Ipiales', 0.8303, -77.6444),
    ('colombia', 'Las Lajas Sanctuary', 0.8055, -77.586),
    ('colombia', 'Barichara', 6.6333, -73.2333),
]


def _areas(country):
    return [G.as_area(a) for a in safety.REGIONAL_WARNINGS[country]]


@pytest.mark.parametrize("country, place, lat, lon", MUST_BLOCK)
def test_must_block_by_coordinates(country, place, lat, lon):
    assert any(a.contains(lat, lon) for a in _areas(country)), place


@pytest.mark.parametrize("country, place, lat, lon", MUST_ALLOW)
def test_must_allow_by_coordinates(country, place, lat, lon):
    assert not [a.name for a in _areas(country) if a.contains(lat, lon)], place
