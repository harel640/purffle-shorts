"""Reference towns of the travel advisories' named Level 3-4 areas: English Wikipedia coordinates,
fetched 2026-11-11 (see advisory_areas.py for how they are used). Generated; edit the area lists,
not the numbers, and re-validate. Each group is one convex hull."""

# fmt: off
GROUPS = {
    ('india', 'Jammu and Kashmir'): [
        [
            (34.09, 74.79),  # Srinagar
            (33.7353, 75.1478),  # Anantnag
            (34.2087, 74.3435),  # Baramulla
            (34.5258, 74.2553),  # Kupwara
            (34.6333, 74.8333),  # Gurez
            (34.4231, 74.6361),  # Bandipora
            (34.23, 74.78),  # Ganderbal
            (34.3015, 75.2925),  # Sonamarg
            (34.01, 75.19),  # Pahalgam
            (33.32, 75.77),  # Kishtwar
            (33.13, 75.57),  # Doda, Jammu and Kashmir
            (32.98, 75.7137),  # Bhaderwah
            (33.25, 75.25),  # Ramban
            (32.916, 75.1419),  # Udhampur
            (33.0805, 74.8366),  # Reasi
            (33.3767, 74.3122),  # Rajouri
            (33.7703, 74.0925),  # Poonch, Jammu and Kashmir
            (32.73, 74.87),  # Jammu
            (32.385, 75.517),  # Kathua
            (32.57, 75.12),  # Samba, India
            (32.87, 74.73),  # Akhnoor
            (34.0861, 74.0333),  # Uri, Jammu and Kashmir
            (34.39, 73.86),  # Tangdhar
            (34.5594, 76.1256),  # Kargil
            (34.4305, 75.7514),  # Dras
            (32.5, 75.82),  # Basohli
            (34.05, 74.38),  # Gulmarg
            (33.5847, 75.3086),  # Kokernag
            (33.7178, 74.8333),  # Shopian
            (33.64, 75.02),  # Kulgam
            (33.8741, 74.9001),  # Pulwama
            (34.015, 74.722),  # Budgam
            (33.42, 75.2),  # Banihal
            (32.4106, 75.6144),  # Lakhanpur, Jammu and Kashmir
            (33.1557, 76.0929),  # Paddar
            (34.6505, 73.9507),  # Keran, Jammu and Kashmir
            (33.8642, 75.5353),  # Warwan Valley
        ],
    ],
    ('india', 'Manipur'): [
        [
            (24.8074, 93.9384),  # Imphal
            (24.6297, 93.7594),  # Bishnupur, Manipur
            (24.63, 94.02),  # Thoubal
            (24.4985, 93.9821),  # Kakching
            (24.346, 93.7),  # Churachandpur
            (24.3312, 94.0011),  # Chandel, Manipur
            (24.385, 94.144),  # Tengnoupal
            (24.2477, 94.3031),  # Moreh
            (25.1, 94.3616),  # Ukhrul
            (24.8583, 94.5098),  # Kamjong
            (25.2712, 94.0274),  # Senapati, Manipur
            (25.1489, 93.9706),  # Kangpokpi
            (24.99, 93.5),  # Tamenglong
            (24.7842, 93.6035),  # Noney
            (24.8106, 93.1155),  # Jiribam
            (24.2616, 93.1895),  # Pherzawl
            (24.2833, 93.2833),  # Thanlon
            (24.249, 93.1064),  # Parbung
        ],
    ],
    ('india', 'Chhattisgarh'): [
        [
            (21.2444, 81.6306),  # Raipur
            (22.09, 82.15),  # Bilaspur, Chhattisgarh
            (21.19, 81.28),  # Durg
            (21.21, 81.38),  # Bhilai
            (21.0968, 81.0291),  # Rajnandgaon
            (22.02, 81.25),  # Kawardha
            (22.07, 81.68),  # Mungeli
            (22.35, 82.68),  # Korba
            (22.017, 82.567),  # Janjgir
            (21.8974, 83.395),  # Raigarh
            (22.8851, 84.1371),  # Jashpur Nagar
            (23.12, 83.2),  # Ambikapur
            (23.22, 82.85),  # Surajpur
            (23.25, 82.55),  # Baikunthpur, Koriya
            (23.1922, 82.2003),  # Manendragarh
            (23.605, 83.617),  # Balrampur, Chhattisgarh
            (21.11, 82.1),  # Mahasamund
            (20.63, 82.05),  # Gariaband
            (20.71, 81.55),  # Dhamtari
            (20.27, 81.49),  # Kanker
            (19.5898, 81.6623),  # Kondagaon
            (19.7167, 81.25),  # Narayanpur
            (19.18, 81.92),  # Jagdalpur
            (18.9, 81.339),  # Dantewada
            (18.7917, 80.8167),  # Bijapur, Chhattisgarh
            (18.4, 81.6667),  # Sukma
            (17.8, 81.3833),  # Konta
            (20.73, 81.2),  # Balod
            (21.7, 81.53),  # Bemetara
            (21.67, 82.17),  # Baloda Bazar
            (21.5864, 83.0779),  # Sarangarh
            (21.42, 80.97),  # Khairagarh
            (22.77, 81.95),  # Pendra
            (23.8, 83.7),  # Ramanujganj
            (23.28, 83.9),  # Kusmi, Balrampur
            (18.8639, 80.3833),  # Bhopalpatnam
            (21.33, 83),  # Saraipali
        ],
    ],
    ('india', 'Jharkhand'): [
        [
            (23.36, 85.33),  # Ranchi
            (22.7925, 86.1842),  # Jamshedpur
            (23.7998, 86.4305),  # Dhanbad
            (23.67, 86.15),  # Bokaro Steel City
            (24.48, 86.7),  # Deoghar
            (24.42, 87.34),  # Dumka
            (24.83, 87.22),  # Godda
            (25.25, 87.65),  # Sahibganj
            (24.63, 87.85),  # Pakur
            (23.9633, 86.8014),  # Jamtara
            (24.1868, 86.3058),  # Giridih
            (24.4666, 85.5943),  # Koderma
            (23.98, 85.35),  # Hazaribagh
            (24.2066, 84.8713),  # Chatra
            (23.745, 84.505),  # Latehar
            (24.03, 84.07),  # Medininagar
            (24.18, 83.82),  # Garhwa
            (23.43, 84.68),  # Lohardaga
            (23.0444, 84.5417),  # Gumla
            (22.62, 84.52),  # Simdega
            (22.57, 85.82),  # Chaibasa
            (22.7022, 85.9316),  # Seraikela
            (23.081, 85.2774),  # Khunti
            (23.62, 85.48),  # Ramgarh Cantonment
            (25.05, 87.84),  # Rajmahal
            (22.3485, 86.7167),  # Bahragora
            (23.4753, 84.2672),  # Netarhat
            (22.1083, 85.2951),  # Kiriburu
        ],
    ],
    ('india', 'Bihar'): [
        [
            (27.43, 83.91),  # Valmiki Nagar
            (26.8014, 84.5028),  # Bettiah
            (26.65, 84.9167),  # Motihari
            (26.9833, 84.85),  # Raxaul
            (26.5942, 85.5006),  # Sitamarhi
            (26.47, 84.43),  # Gopalganj, India
            (26.22, 84.36),  # Siwan
            (25.7848, 84.7274),  # Chhapra
            (25.6925, 85.2084),  # Hajipur
            (26.1225, 85.3906),  # Muzaffarpur
            (26.17, 85.9),  # Darbhanga
            (26.3519, 86.0719),  # Madhubani, India
            (25.8629, 85.7811),  # Samastipur
            (25.42, 86.13),  # Begusarai
            (25.5939, 85.1375),  # Patna
        ],
        [
            (25.5667, 83.9833),  # Buxar
            (25.05, 83.62),  # Bhabua
            (24.95, 84.03),  # Sasaram
            (25.5514, 84.6603),  # Arrah
            (25.5939, 85.1375),  # Patna
            (25.2125, 84.9853),  # Jehanabad
            (24.75, 85.01),  # Gaya, India
            (24.7, 84.35),  # Aurangabad, Bihar
            (24.8867, 85.5433),  # Nawada
            (24.9278, 86.2239),  # Jamui
            (25.381, 86.465),  # Munger
            (25.25, 87),  # Bhagalpur
            (24.88, 86.92),  # Banka, India
            (25.27, 87.22),  # Kahalgaon
            (25.42, 86.13),  # Begusarai
        ],
        [
            (26.126, 86.605),  # Supaul
            (26.15, 87.52),  # Araria
            (25.778, 87.476),  # Purnia
            (26.0986, 87.9453),  # Kishanganj
            (25.53, 87.58),  # Katihar
            (26.45, 88.13),  # Thakurganj
            (25.25, 87),  # Bhagalpur
            (25.42, 86.13),  # Begusarai
            (26.17, 85.9),  # Darbhanga
            (26.3519, 86.0719),  # Madhubani, India
            (25.88, 86.6),  # Saharsa
            (25.92, 86.792),  # Madhepura
            (26.3961, 87.2597),  # Jogbani
        ],
    ],
    ('india', 'Odisha'): [
        [
            (20.27, 85.84),  # Bhubaneswar
            (20.5236, 85.7881),  # Cuttack
            (19.81, 85.83),  # Puri
            (19.315, 84.7941),  # Berhampur
            (21.47, 83.97),  # Sambalpur
            (22.2492, 84.8828),  # Rourkela
            (21.4969, 86.9225),  # Balasore
            (21.94, 86.72),  # Baripada
            (21.63, 85.58),  # Keonjhar
            (22.12, 84.03),  # Sundargarh
            (21.85, 84.03),  # Jharsuguda
            (21.3333, 83.6167),  # Bargarh
            (20.72, 83.48),  # Balangir
            (20.8061, 82.5338),  # Nuapada
            (19.91, 83.128),  # Bhawanipatna
            (19.23, 82.55),  # Nabarangpur
            (18.8561, 82.7347),  # Koraput
            (18.3627, 81.8934),  # Malkangiri
            (19.17, 83.42),  # Rayagada
            (19.08, 83.82),  # Gunupur
            (18.7781, 84.0933),  # Paralakhemundi
            (20.47, 84.23),  # Phulbani
            (20.84, 84.32),  # Boudh
            (20.8381, 85.0956),  # Angul
            (20.67, 85.6),  # Dhenkanal
            (20.85, 86.33),  # Jajpur
            (20.5033, 86.4194),  # Kendrapara
            (20.27, 86.17),  # Jagatsinghpur
            (21.06, 86.5),  # Bhadrak
            (21.47, 87.02),  # Chandipur
            (20.316, 86.61),  # Paradeep
            (19.27, 84.92),  # Gopalpur, Odisha
            (20.95, 85.23),  # Talcher
            (21.8167, 87.2167),  # Jaleswar
            (20.28, 82.77),  # Khariar
            (19.665, 82.212),  # Umerkote
            (19.3234, 82.5175),  # Papadahandi
            (21.57, 86.57),  # Udala
            (20.8233, 86.9628),  # Dhamra Port
            (20.7442, 87.0003),  # Gahirmatha Marine Wildlife Sanctuary
            (21.605, 87.466),  # Talsari
            (19.9795, 86.2694),  # Astaranga
        ],
    ],
    ('india', 'Arunachal Pradesh'): [
        [
            (27.1, 93.62),  # Itanagar
            (27.5883, 91.8653),  # Tawang
            (27.265, 92.42),  # Bomdila
            (27.3616, 93.0405),  # Seppa
            (27.5417, 93.8161),  # Ziro
            (27.9861, 94.2208),  # Daporijo
            (28.1707, 94.7995),  # Aalo
            (28.07, 95.33),  # Pasighat
            (28.6104, 95.0475),  # Yingkiong
            (27.92, 96.17),  # Tezu
            (28.7981, 95.9036),  # Anini
            (28.1428, 95.8428),  # Roing
            (27.1387, 95.7365),  # Changlang
            (26.9908, 95.5018),  # Khonsa
            (26.9, 95.3),  # Longding
            (27.6689, 95.8714),  # Namsai
            (28.605, 94.1281),  # Mechuka
            (28.9922, 94.8947),  # Tuting
            (28.0629, 96.548),  # Hayuliang
            (28.2803, 97.0178),  # Kibithu
            (27.1919, 96.9997),  # Vijoynagar
            (27.9167, 93.35),  # Koloriang
            (27.7237, 91.8916),  # Bum La Pass
            (27.5048, 92.1047),  # Sela Pass
            (27.49, 96.2),  # Miao, Arunachal Pradesh
            (27.529, 91.713),  # Lumla
            (28.4347, 93.2039),  # Taksing
            (28.395, 93.838),  # Nacho, Arunachal Pradesh
            (27.011, 92.646),  # Bhalukpong
        ],
    ],
    ('india', 'Nagaland'): [
        [
            (25.67, 94.1),  # Kohima
            (25.9092, 93.7266),  # Dimapur
            (26.32, 94.5),  # Mokokchung
            (26.2716, 94.8314),  # Tuensang
            (26.7208, 95.0286),  # Mon, Nagaland
            (26.1, 94.27),  # Wokha
            (26.0146, 94.5264),  # Zunheboto
            (25.7096, 94.4644),  # Phek
            (25.888, 94.785),  # Kiphire
            (26.4685, 94.8092),  # Longleng
            (25.51, 93.74),  # Peren
            (26.2055, 95.0151),  # Noklak
            (25.571, 94.301),  # Pfutsero
            (25.8064, 93.7786),  # Chümoukedima
            (25.6837, 94.626),  # Meluri
            (26.9031, 95.0825),  # Tizit
            (25.9247, 94.2149),  # Tseminyü
            (25.6475, 93.6693),  # Jalukie
        ],
    ],
    ('india', 'Mizoram'): [
        [
            (23.733, 92.718),  # Aizawl
            (22.88, 92.73),  # Lunglei
            (23.456, 93.329),  # Champhai
            (24.2274, 92.6789),  # Kolasib
            (23.9251, 92.4914),  # Mamit
            (23.309, 92.8567),  # Serchhip
            (22.48, 92.97),  # Saiha
            (22.532, 92.899),  # Lawngtlai
            (23.5345, 93.183),  # Khawzawl
            (22.9826, 92.9113),  # Hnahthial
            (23.6896, 92.9557),  # Saitual
            (24.5001, 92.7638),  # Vairengte
            (22.912, 92.498),  # Tlabung
            (23.32, 92.75),  # Thenzawl
            (23.3652, 93.3855),  # Zokhawthar
            (22.3167, 93.0333),  # Tuipang
            (22.62, 92.639),  # Parva, Mizoram
        ],
    ],
    ('india', 'Assam'): [
        [
            (26.02, 90),  # Dhubri
            (25.7165, 89.8986),  # Hatsingimari
            (26.17, 90.62),  # Goalpara
            (26.4, 90.27),  # Kokrajhar
            (26.4769, 90.5583),  # Bongaigaon
            (26.4742, 90.5644),  # Kajalgaon
            (26.32, 91),  # Barpeta
            (26.445, 91.44),  # Nalbari
            (26.6641, 91.3407),  # Mushalpur
            (26.47, 91.63),  # Rangia
            (26.1722, 91.7458),  # Guwahati
            (26.2522, 92.3422),  # Morigaon
            (26.3504, 92.6796),  # Nagaon
            (26, 92.87),  # Hojai
            (26.43, 92.03),  # Mangaldai
            (26.7452, 92.0962),  # Udalguri
            (26.6283, 92.7997),  # Tezpur
            (26.7167, 90.9333),  # Manas National Park
            (26.4197, 89.9842),  # Gossaigaon
            (26.17, 91.57),  # Sualkuchi
            (26.2411, 92.0583),  # Pobitora Wildlife Sanctuary
        ],
        [
            (26.3504, 92.6796),  # Nagaon
            (26.6283, 92.7997),  # Tezpur
            (26.7278, 93.1517),  # Biswanath Chariali
            (26.88, 93.63),  # Gohpur
            (27.2414, 94.1056),  # North Lakhimpur
            (27.48, 94.58),  # Dhemaji
            (26.52, 93.97),  # Golaghat
            (26.75, 94.22),  # Jorhat
            (27, 94.2167),  # Majuli
            (26.9889, 94.639),  # Sivasagar
            (27.07, 95.03),  # Sonari, Assam
            (27.47, 94.92),  # Dibrugarh
            (27.4883, 95.3555),  # Tinsukia
            (27.28, 95.68),  # Margherita, Assam
            (27.83, 95.67),  # Sadiya
            (27.38, 95.63),  # Digboi
            (26.6667, 93.35),  # Kaziranga National Park
            (27.6667, 95.3833),  # Dibru-Saikhowa National Park
            (27.83, 95.22),  # Jonai
            (26.63, 93.6),  # Bokakhat
        ],
        [
            (25.83, 93.43),  # Diphu
            (25.9233, 92.6083),  # Hamren
            (25.169, 93.016),  # Haflong
            (25.2953, 93.1394),  # Maibang
            (25.5136, 92.737),  # Umrangso
            (26.02, 93.78),  # Bokajan
            (26.0872, 93.1381),  # Howraghat
            (25.92, 92.95),  # Lanka, Assam
            (26.2081, 93.0501),  # Dokmoka
            (25.17, 93.12),  # Mahur
        ],
        [
            (24.82, 92.8),  # Silchar
            (24.87, 92.35),  # Karimganj
            (24.68, 92.57),  # Hailakandi
            (24.9, 92.6),  # Badarpur, Assam
            (24.8, 93.02),  # Lakhipur
            (24.6086, 92.3214),  # Patharkandi
            (24.7833, 93.1),  # Jirighat
        ],
    ],
    ('philippines', 'Mindanao'): [
        [
            (6.9042, 122.0761),  # Zamboanga City
            (8.5872, 123.3408),  # Dipolog
            (8.6549, 123.4243),  # Dapitan
            (8.2386, 122.9986),  # Sindangan
            (8.1228, 122.6736),  # Liloy
            (7.7067, 122.1361),  # Siocon
            (7.2928, 122.0669),  # Sibuco
            (8.0649, 122.5247),  # Labason
            (7.8272, 123.4364),  # Pagadian
            (7.7822, 122.5867),  # Ipil, Zamboanga Sibugay
            (7.7968, 122.7627),  # Kabasalan
            (7.5783, 123.1659),  # Margosatubig
            (8.15, 123.85),  # Ozamiz
            (8.48, 123.8),  # Oroquieta
            (8.23, 124.25),  # Iligan
            (8.0031, 124.285),  # Marawi
            (8.48, 124.65),  # Cagayan de Oro
            (8.82, 125.1),  # Gingoog
            (8.948, 125.543),  # Butuan
            (9.12, 125.53),  # Cabadbaran
            (9.7897, 125.4958),  # Surigao City
            (9.0789, 126.1986),  # Tandag
            (8.2133, 126.3161),  # Bislig
            (6.9535, 126.2163),  # Mati, Davao Oriental
            (7.5752, 126.5585),  # Baganga
            (6.6536, 126.0717),  # Governor Generoso
            (7.0662, 125.6094),  # Davao City
            (6.7454, 125.3559),  # Digos
            (6.4108, 125.6144),  # Malita
            (5.9126, 125.6443),  # Jose Abad Santos, Davao Occidental
            (5.8231, 125.2042),  # Glan, Sarangani
            (6.12, 125.17),  # General Santos
            (6.5557, 124.0507),  # Kalamansig
            (6.63, 124.07),  # Lebak, Sultan Kudarat
            (7.22, 124.25),  # Cotabato City
            (8.1564, 125.1333),  # Malaybalay
            (7.9042, 125.0928),  # Valencia, Bukidnon
            (7.0083, 125.0894),  # Kidapawan
            (6.5, 124.85),  # Koronadal
            (6.6884, 124.6787),  # Tacurong
            (6.63, 124.6),  # Isulan
            (8.61, 125.92),  # Prosperidad
            (8.71, 125.75),  # Bayugan
            (7.6034, 125.967),  # Nabunturan
            (7.4478, 125.8078),  # Tagum
            (8.0381, 126.4127),  # Lingig
            (7.79, 126.4531),  # Cateel
            (8.633, 126.0932),  # Lianga
            (9.3356, 125.9769),  # Cantilan
            (9.5742, 125.7328),  # Claver, Surigao del Norte
            (9.657, 125.6016),  # Placer, Surigao del Norte
            (8.99, 125.34),  # Nasipit
            (8.67, 123.6),  # Baliangao
            (8.55, 123.57),  # Sapang Dalaga
            (6.22, 124.2),  # Palimbang
            (6.03, 124.48),  # Maitum
            (5.9333, 124.9667),  # Maasim
            (5.412, 125.422),  # Sarangani, Davao Occidental
            (7.0783, 125.7117),  # Island Garden City of Samal
            (7.3744, 124.2686),  # Parang, Maguindanao del Norte
            (7.5297, 123.3661),  # Dimataling
            (7.8547, 123.5783),  # Tukuran
            (8.115, 123.9024),  # Kolambugan
        ],
        [
            (9.25, 124.72),  # Mambajao
            (9.13, 124.68),  # Catarman, Camiguin
            (9.15, 124.78),  # Mahinog
            (9.098, 124.786),  # Guinsiliban
            (9.12, 124.72),  # Sagay, Camiguin
        ],
        [
            (10.0083, 125.5889),  # San Jose, Dinagat Islands
            (10.3586, 125.5816),  # Loreto, Dinagat Islands
            (10.0654, 125.5968),  # Basilisa
            (9.92, 125.67),  # Cagdianao
            (10.196, 125.5328),  # Libjo
            (10.3253, 125.5572),  # Tubajon
            (9.9561, 125.5933),  # Dinagat, Dinagat Islands
        ],
        [
            (9.783, 126.156),  # General Luna, Surigao del Norte
            (9.7578, 126.0528),  # Dapa, Surigao del Norte
            (9.869, 125.97),  # Del Carmen, Surigao del Norte
            (10.018, 126.074),  # Burgos, Surigao del Norte
            (9.8639, 126.1008),  # Pilar, Surigao del Norte
            (9.958, 126.007),  # San Benito, Surigao del Norte
            (10.02, 126.038),  # Santa Monica, Surigao del Norte
            (9.618, 125.966),  # Socorro, Surigao del Norte
        ],
    ],
    ('philippines', 'Sulu Archipelago'): [
        [
            (6.7, 121.97),  # Isabela, Basilan
            (6.6582, 122.1457),  # Lamitan
            (6.546, 121.87),  # Maluso
            (6.5006, 122.1193),  # Tipo-Tipo
            (6.05, 121),  # Jolo, Sulu
            (5.5462, 120.8145),  # Siasi
            (5.8444, 121.1439),  # Pata, Sulu
            (6.3, 120.58),  # Pangutaran
            (5.0292, 119.7731),  # Bongao
            (4.6615, 119.3919),  # Sitangkai
            (4.841, 119.459),  # Sibutu
            (4.898, 119.8213),  # Simunul
            (5.268, 120.079),  # Languyan
            (5.0899, 120.2729),  # Sapa-Sapa
            (5.7048, 120.8818),  # Tapul
            (5.88, 121.27),  # Kalingalan Caluang
            (6.975, 118.514),  # Mapun
            (6.073, 118.312),  # Turtle Islands, Tawi-Tawi
            (5.135, 120.346),  # Tandubas
            (5.18, 120.48),  # South Ubian
        ],
    ],
    ('philippines', 'Marawi'): [
        [
            (8.0031, 124.285),  # Marawi
            (7.88, 124.2525),  # Lake Lanao
        ],
    ],
    ('egypt', 'Sinai'): [
        [
            (31.1321, 33.8034),  # Arish
            (31.2808, 34.2414),  # Rafah, Egypt
            (31.2119, 34.1106),  # Sheikh Zuweid
            (31.0181, 33.0111),  # Bir al-Abd
            (29.9, 33.75),  # Nekhel
            (29.4917, 34.8944),  # Taba, Egypt
            (29.0333, 34.6667),  # Nuweiba
            (28.4931, 34.5047),  # Dahab
            (27.915, 34.3275),  # Sharm El Sheikh
            (28.5644, 33.9524),  # Saint Catherine, Egypt
            (28.2417, 33.6222),  # El Tor, Egypt
            (28.9, 33.2),  # Abu Rudeis
            (29.0454, 33.1107),  # Abu Zenima
            (29.5917, 32.7056),  # Ras Sedr
            (27.7694, 34.2097),  # Ras Muhammad National Park
            (31.25, 32.3167),  # Port Fuad
        ],
    ],
    ('mexico', 'Colima'): [
        [
            (19.2433, -103.7247),  # Colima City
            (19.0522, -104.3158),  # Manzanillo, Colima
            (18.9089, -103.8747),  # Tecomán
            (18.936, -103.965),  # Armería
            (19.3208, -103.7603),  # Comala
            (19.25, -103.7333),  # Villa de Álvarez
            (19.3833, -104.05),  # Minatitlán, Colima
            (19.3281, -103.6028),  # Cuauhtémoc, Colima
        ],
        [
            (18.7844, -110.975),  # Socorro Island
            (18.3583, -114.725),  # Clarion Island
            (19.3044, -110.8144),  # San Benedicto Island
        ],
    ],
    ('mexico', 'Guerrero'): [
        [
            (17.55, -99.5),  # Chilpancingo
            (16.8636, -99.8825),  # Acapulco
            (18.345, -99.5383),  # Iguala
            (18.5564, -99.605),  # Taxco
            (17.6444, -101.5511),  # Zihuatanejo
            (17.5383, -101.2739),  # Petatlán
            (17.25, -100.6833),  # Tecpan de Galeana
            (16.6833, -98.4167),  # Ometepec
            (16.4717, -98.4153),  # Cuajinicuilapa
            (17.5461, -98.5764),  # Tlapa de Comonfort
            (18.3472, -100.6531),  # Ciudad Altamirano
            (18.3256, -100.6992),  # Coyuca de Catalán
            (18.2833, -100.2667),  # Arcelia
            (18.35, -99.85),  # Teloloapan
            (17.5944, -99.1778),  # Chilapa de Álvarez
            (17.8333, -98.85),  # Olinalá
            (16.5833, -98.8167),  # Marquelia
            (17.9667, -101.8167),  # La Unión, Guerrero
            (17.8167, -98.5667),  # Huamuxtitlán
            (18.4, -100.4667),  # Tlalchapa
        ],
    ],
    ('mexico', 'Michoacán'): [
        [
            (19.7683, -101.1894),  # Morelia
            (19.4208, -102.0628),  # Uruapan
            (19.9833, -102.2833),  # Zamora, Michoacán
            (19.0886, -102.3508),  # Apatzingán
            (17.9561, -102.1922),  # Lázaro Cárdenas, Michoacán
            (19.5164, -101.6097),  # Pátzcuaro
            (19.4, -100.3667),  # Zitácuaro
            (19.6917, -100.5536),  # Ciudad Hidalgo, Michoacán
            (20.05, -102.7333),  # Sahuayo
            (20.3333, -102.0167),  # La Piedad
            (19.8976, -100.45),  # Maravatío
            (18.5833, -100.8833),  # Huetamo
            (18.5833, -103.5167),  # Aquila, Michoacán
            (18.7833, -103.1667),  # Coalcomán
            (19.1833, -102.85),  # Tepalcatepec
            (20.0881, -101.5156),  # Puruándiro
            (20.2819, -102.3319),  # Yurécuaro
            (19.2, -100.5667),  # Tuzantla
            (19.9539, -100.1636),  # Contepec
        ],
    ],
    ('mexico', 'Sinaloa'): [
        [
            (24.8069, -107.3939),  # Culiacán
            (23.2167, -106.4167),  # Mazatlán
            (25.7835, -108.9937),  # Los Mochis
            (25.5744, -108.4706),  # Guasave
            (25.4639, -108.0794),  # Guamúchil
            (26.4214, -108.62),  # El Fuerte, Sinaloa
            (22.835, -105.778),  # Escuinapa
            (22.9922, -105.8572),  # El Rosario, Sinaloa
            (23.2883, -106.0675),  # Concordia, Sinaloa
            (24.4125, -106.6917),  # Cosalá
            (25.3631, -107.5517),  # Badiraguato
            (26.7086, -108.3253),  # Choix
            (24.7656, -107.7019),  # Navolato
            (25.6056, -109.05),  # Topolobampo
            (22.5471, -105.75),  # Teacapán
        ],
    ],
    ('mexico', 'Tamaulipas'): [
        [
            (23.7389, -99.1431),  # Ciudad Victoria
            (26.0922, -98.2778),  # Reynosa
            (25.8797, -97.5042),  # Matamoros, Tamaulipas
            (27.4861, -99.5069),  # Nuevo Laredo
            (22.2553, -97.8686),  # Tampico
            (22.2475, -97.8375),  # Ciudad Madero
            (22.3375, -97.8694),  # Altamira, Tamaulipas
            (22.7333, -98.95),  # Ciudad Mante
            (25.9861, -98.0889),  # Río Bravo, Tamaulipas
            (25.6736, -97.8144),  # Valle Hermoso, Tamaulipas
            (24.8481, -98.1594),  # San Fernando, Tamaulipas
            (23.7667, -98.2167),  # Soto la Marina
            (23.4167, -99.3833),  # Jaumave
            (23, -99.72),  # Tula, Tamaulipas
            (26.4003, -99.0253),  # Ciudad Miguel Alemán
            (26.0561, -97.9522),  # Nuevo Progreso, Tamaulipas
            (23.7667, -97.7833),  # La Pesca
            (26.4306, -99.1486),  # Ciudad Mier
        ],
    ],
    ('mexico', 'Zacatecas'): [
        [
            (22.7736, -102.5736),  # Zacatecas City
            (23.175, -102.8675),  # Fresnillo
            (23.6333, -103.6397),  # Sombrerete
            (23.8225, -103.0361),  # Río Grande, Zacatecas
            (24.6167, -101.4167),  # Concepción del Oro
            (24.6383, -101.5553),  # Mazapil
            (22.2972, -101.575),  # Pinos, Zacatecas
            (22.5667, -102.25),  # Ojocaliente
            (22.3536, -102.8831),  # Villanueva, Zacatecas
            (22.2667, -101.9833),  # Loreto, Zacatecas
            (24.2911, -103.3939),  # Juan Aldama, Zacatecas
            (23.4761, -103.8806),  # Chalchihuites
            (22.7667, -103.5667),  # Valparaíso, Zacatecas
            (22.6481, -102.9903),  # Jerez de García Salinas
        ],
        [
            (21.6375, -102.9778),  # Jalpa, Zacatecas
            (21.4103, -103.1175),  # Juchipila
            (21.3642, -102.8464),  # Nochistlán
            (21.7815, -103.3059),  # Tlaltenango de Sánchez Román
            (22.3536, -102.8831),  # Villanueva, Zacatecas
            (22.6481, -102.9903),  # Jerez de García Salinas
            (22.7667, -103.5667),  # Valparaíso, Zacatecas
        ],
    ],
    ('colombia', 'Arauca'): [
        [
            (7.0903, -70.7617),  # Arauca, Arauca
            (6.9556, -71.8722),  # Saravena
            (6.4667, -71.7333),  # Tame, Arauca
            (7.0303, -71.4356),  # Arauquita
            (6.7922, -71.7756),  # Fortul
            (6.3, -70.2),  # Cravo Norte
            (6.2833, -71.1),  # Puerto Rondón
        ],
    ],
    ('colombia', 'Cauca'): [
        [
            (2.4542, -76.6092),  # Popayán
            (3.0167, -76.4833),  # Santander de Quilichao
            (2.5703, -77.8856),  # Guapi
            (2.7719, -77.665),  # Timbiquí
            (2.8459, -77.2481),  # López de Micay
            (2.1142, -76.9831),  # El Bordo
            (2.6408, -76.5284),  # Piendamó
            (2.6167, -76.3833),  # Silvia, Cauca
            (2.55, -76.0667),  # Inzá
            (2.646, -75.973),  # Belalcázar, Cauca
            (1.9708, -76.9694),  # Bolívar, Cauca
            (1.1194, -76.3269),  # Piamonte, Cauca
            (1.7294, -76.6044),  # Santa Rosa, Cauca
            (3.25, -76.4167),  # Puerto Tejada
            (3.25, -76.25),  # Miranda, Cauca
            (2.9581, -76.2714),  # Toribío
            (2.2431, -77.2786),  # Argelia, Cauca
            (1.8, -77.1667),  # Mercaderes
            (2.9675, -78.1803),  # Gorgona Island
        ],
    ],
    ('colombia', 'Norte de Santander'): [
        [
            (7.8942, -72.5039),  # Cúcuta
            (8.2333, -73.35),  # Ocaña, Colombia
            (7.3781, -72.6525),  # Pamplona, Colombia
            (8.65, -72.7333),  # Tibú
            (8.0475, -73.0836),  # El Tarra
            (8.8333, -73.2),  # Convención
            (7.6167, -72.6),  # Chinácota
            (7.3131, -72.4875),  # Toledo, Norte de Santander
            (8.0833, -72.8),  # Sardinata
            (8.4333, -72.2833),  # Teorama
            (8, -73.2),  # Ábrego
            (8.3167, -73.15),  # Hacarí
            (7.5833, -72.4833),  # Ragonvalia
            (7.5167, -72.4833),  # Herrán, Norte de Santander
        ],
    ],
}

# The India-Pakistan border: (polyline of (lat, lon), buffer km). See advisory_areas / safety.
BORDER_LINES = {
    ('india', 'India-Pakistan border'): [
        ([(23.58, 68.15), (23.967, 68.8), (24.219, 69.081), (24.27, 69.6), (24.28, 70.2), (24.27, 70.75), (24.45, 71.08), (24.72, 71.06), (25.2, 70.66), (25.72, 70.24)], 25.0),
        ([(25.72, 70.24), (26.05, 69.95), (26.45, 69.55), (26.9, 69.5), (27.3, 69.65), (27.62, 69.95), (27.95, 70.25), (28.15, 70.7), (28.45, 71.3), (28.7, 71.95), (28.95, 72.45), (29.2, 73.0), (29.55, 73.3), (29.9, 73.6), (30.136, 73.917)], 40.0),
        ([(27.95, 70.25), (28.1, 70.8), (28.2, 71.5), (28.15, 71.9), (28.4, 72.2), (28.75, 72.35), (29.1, 72.95)], 40.0),
        ([(30.136, 73.917), (30.378, 73.867), (30.42, 73.93), (30.9977, 74.5471), (31.1444, 74.56), (31.6008, 74.6056), (31.6047, 74.5731), (31.84, 74.76), (32.032, 75.029), (32.27, 75.36), (32.42, 75.3), (32.6, 74.71), (32.93, 74.37)], 15.0),
    ],
}
# fmt: on
