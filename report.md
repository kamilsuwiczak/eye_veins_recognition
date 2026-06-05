# Raport: Wykrywanie naczyń dna siatkówki oka

## 1. Skład grupy
*(Uzupełnij)*

## 2. Zastosowany język programowania oraz dodatkowe biblioteki

- **Język**: Python 3.14
- **Interfejs**: Streamlit
- **Przetwarzanie obrazu**: OpenCV (`opencv-python`), scikit-image (`skimage.filters.frangi`)
- **Uczenie maszynowe**: scikit-learn (`RandomForestClassifier`), imbalanced-learn (`specificity_score`, `geometric_mean_score`)
- **Głębokie uczenie**: PyTorch (`torch`, `torchvision`)
- **Inne**: NumPy, joblib

## 3. Opis zastosowanych metod

### 3.1. Przetwarzanie obrazów (Filtr Frangiego) — wymagania obowiązkowe

#### Poszczególne kroki przetwarzania:

1. **Wstępne przetworzenie obrazu:**
   - Wczytanie obrazu RGB z bazy HRF (High Resolution Fundus).
   - Ekstrakcja **zielonego kanału** — kanał ten najlepiej kontrastuje naczynia krwionośne na tle siatkówki.
   - Zastosowanie **CLAHE** (Contrast Limited Adaptive Histogram Equalization) z parametrami `clipLimit=2.0`, `tileGridSize=(8,8)` w celu lokalnej normalizacji kontrastu.
   - Zastosowanie maski pola widzenia (FOV) w celu ograniczenia przetwarzania do wnętrza oka.

2. **Właściwe przetworzenie obrazu:**
   - Zastosowanie **filtru Frangiego** (z biblioteki scikit-image) do wykrywania struktur rurkowych (naczyń). Filtr analizuje wartości własne macierzy hesjanu obrazu w wielu skalach (`sigmas` od 1.5 do 4.0 z krokiem 0.5).
   - Parametr `black_ridges=True` wskazuje, że naczynia są ciemniejszymi strukturami na jasnym tle (po ekstrakcji zielonego kanału).

3. **Końcowe przetwarzanie obrazu:**
   - Normalizacja odpowiedzi filtru Frangiego do zakresu [0, 255].
   - Erozja maski FOV o zadaną liczbę pikseli (domyślnie 15 px) w celu usunięcia artefaktów na krawędzi oka.
   - **Progowanie** (thresholding) z progiem domyślnym 15, generujące binarną maskę naczyń.

#### Uzasadnienie:
Filtr Frangiego jest powszechnie stosowaną metodą detekcji naczyń krwionośnych w obrazach siatkówki. Analiza wieloskalowa pozwala wykrywać zarówno szerokie, jak i wąskie naczynia. Metoda CLAHE poprawia kontrast lokalny, co zwiększa skuteczność filtru.

---

### 3.2. Uczenie maszynowe (Random Forest) — wymagania na 4.0

#### Przygotowanie danych:
1. Po wstępnym przetworzeniu (zielony kanał + CLAHE) obraz jest analizowany piksel po pikselu.
2. Dla każdego piksela wyznaczane są **4 cechy**:
   - **Odpowiedź filtru Frangiego** — wartość znormalizowana [0, 255].
   - **Transformacja Black-Hat** — operacja morfologiczna z jądrem eliptycznym, wykrywająca ciemne struktury mniejsze od jądra.
   - **Gradient Sobela** — magnituda gradientu (krawędzi) w kierunkach X i Y.
   - **Lokalna wariancja** — odchylenie standardowe jasności w oknie k×k.

#### Wstępne przetwarzanie zbioru uczącego:
- Zastosowano **undersampling**: losowy wybór po `N` próbek z każdej klasy (domyślnie 5000 na klasę) w celu zrównoważenia rozkładu klas.
- Tylko piksele wewnątrz maski FOV są brane pod uwagę.

#### Zastosowany klasyfikator:
- **Random Forest** (`sklearn.ensemble.RandomForestClassifier`):
  - `n_estimators=50` (liczba drzew)
  - `max_depth=8` (domyślna maksymalna głębokość)
  - `random_state=42` (powtarzalność)
  - `n_jobs=-1` (wykorzystanie wszystkich rdzeni)

#### Wstępna ocena (hold-out):
- Trening przeprowadzany na jednym obrazie (np. `01_g.jpg`), testowanie na innym, niezależnym obrazie (np. `02_g.jpg`).
- Wyniki: patrz sekcja 5 (Tabela porównawcza).

#### Uzasadnienie:
Random Forest dobrze radzi sobie z niezrównoważonymi danymi i jest odporny na overfitting dzięki mechanizmowi bagging. Wybrane cechy odzwierciedlają lokalne właściwości tekstury i struktury naczyń.

---

### 3.3. Głębokie uczenie (U-Net) — wymagania na 5.0

#### Architektura sieci:
- **MiniUNet** — lekka wersja architektury U-Net zaimplementowana w PyTorch.
- Struktura: koder-dekoder z połączeniami pomijającymi (skip connections).
- Kanały: 16 → 32 → 64 → 128 (bottleneck) → 64 → 32 → 16 → 1.
- Bloki konwolucyjne: `Conv2d(3×3)` → `BatchNorm` → `ReLU` (podwójne w każdym bloku).
- Wyjście: sigmoidalna mapa prawdopodobieństwa [0, 1].
- Automatyczne dopełnienie wejścia do wielokrotności 8 i odcięcie wyjścia.
- Łączna liczba parametrów: ~482 000.

#### Funkcja straty:
- **BCEDiceLoss** — kombinacja wagowa (50/50):
  - Binary Cross-Entropy (BCE): karze za błędne klasyfikacje.
  - Dice Loss: mierzy stopień pokrycia (overlap) z maską ekspercką, co jest szczególnie skuteczne przy niezrównoważonych klasach.

#### Przygotowanie danych:
- Wejście sieci: 1-kanałowy obraz (zielony kanał po korekcji CLAHE), znormalizowany do [0, 1].
- Trening na **losowych wycinkach (patchach) 256×256 px** wyciętych z obrazów treningowych.
- Filtrowanie wycinków: co najmniej 30% pikseli wycinku musi znajdować się wewnątrz oka (FOV).
- 30 wycinków na obraz treningowy, batch size = 8.
- Zbiór treningowy: 9 obrazów z bazy HRF (`01_dr`, `01_g`, `01_h`, `02_dr`, `02_g`, `02_h`, `03_dr`, `03_g`, `03_h`).

#### Parametry uczenia:
- Optymalizator: **Adam** z learning rate = 0.001.
- Liczba epok: 20.
- Urządzenie: CPU (Intel i5-1235U).

#### Wstępna ocena (hold-out):
- Testowanie na 5 niezależnych obrazach, które nie uczestniczyły w treningu.
- Wyniki: patrz sekcja 5 (Tabela porównawcza).

#### Uzasadnienie:
Architektura U-Net jest standardem w segmentacji obrazów medycznych. Połączenia pomijające (skip connections) pozwalają łączyć informacje o kontekście globalnym (z głębszych warstw) z precyzyjną lokalizacją (z płytszych warstw). Kombinacja BCE i Dice Loss pomaga radzić sobie z silnym niezrównoważeniem klas.

---

## 4. Wizualizacja wyników

Wizualizacje zostały przygotowane za pomocą aplikacji Streamlit. Poniżej przedstawiono wyniki dla 5 obrazów testowych hold-out.

Dla każdego obrazu prezentowane są:
- Obraz oryginalny
- Maska ekspercka (Ground Truth)
- Maska wygenerowana przez filtr Frangiego
- Maska wygenerowana przez Random Forest
- Maska wygenerowana przez U-Net

*(Wyniki wizualizacji należy uruchomić za pomocą aplikacji Streamlit: `streamlit run main.py`)*

---

## 5. Analiza wyników (Tabela porównawcza)

Wyniki ewaluacji dla 5 obrazów testowych hold-out (`04_g`, `04_h`, `04_dr`, `05_g`, `05_h`) wygenerowane przez skrypt `run_evaluation.py`.

### Wyniki ewaluacji na 5 obrazach testowych hold-out

Obrazy testowe: `04_g.jpg`, `04_h.jpg`, `04_dr.JPG`, `05_g.jpg`, `05_h.jpg` — żaden z nich nie był użyty w procesie uczenia.

| Obraz | Metoda | Accuracy | Sensitivity | Specificity | G-Mean | Arith. Mean | TP | TN | FP | FN |
|------------|--------|----------|-------------|-------------|--------|-------------|--------|---------|-------|-------|
| 04_g.jpg | Frangi | 0.9590 | 0.7108 | 0.9799 | 0.8346 | 0.8454 | 18 851 | 309 095 | 6 339 | 7 668 |
| 04_g.jpg | RF | 0.8975 | 0.9115 | 0.8964 | 0.9039 | 0.9039 | 24 978 | 298 240 | 34 487 | 2 426 |
| 04_g.jpg | U-Net | 0.9671 | 0.6840 | 0.9904 | 0.8231 | 0.8372 | 18 744 | 329 530 | 3 197 | 8 660 |
| 04_h.jpg | Frangi | 0.9424 | 0.6452 | 0.9806 | 0.7954 | 0.8129 | 25 148 | 297 077 | 5 883 | 13 827 |
| 04_h.jpg | RF | 0.8556 | 0.9258 | 0.8468 | 0.8854 | 0.8863 | 37 182 | 270 948 | 49 014 | 2 978 |
| 04_h.jpg | U-Net | 0.9659 | 0.7709 | 0.9904 | 0.8738 | 0.8807 | 30 960 | 316 899 | 3 063 | 9 200 |
| 04_dr.JPG | Frangi | 0.9339 | 0.6869 | 0.9520 | 0.8086 | 0.8194 | 16 057 | 303 302 | 15 290 | 7 320 |
| 04_dr.JPG | RF | 0.7777 | 0.9087 | 0.7683 | 0.8355 | 0.8385 | 21 861 | 258 216 | 77 870 | 2 197 |
| 04_dr.JPG | U-Net | 0.9704 | 0.7313 | 0.9875 | 0.8498 | 0.8594 | 17 593 | 331 877 | 4 209 | 6 465 |
| 05_g.jpg | Frangi | 0.9586 | 0.5848 | 0.9908 | 0.7612 | 0.7878 | 15 856 | 311 947 | 2 911 | 11 256 |
| 05_g.jpg | RF | 0.9229 | 0.8663 | 0.9276 | 0.8965 | 0.8970 | 24 275 | 308 096 | 24 034 | 3 746 |
| 05_g.jpg | U-Net | 0.9697 | 0.7464 | 0.9886 | 0.8590 | 0.8675 | 20 915 | 328 333 | 3 797 | 7 106 |
| 05_h.jpg | Frangi | 0.9541 | 0.6838 | 0.9872 | 0.8216 | 0.8355 | 25 514 | 300 688 | 3 903 | 11 796 |
| 05_h.jpg | RF | 0.9048 | 0.9207 | 0.9029 | 0.9118 | 0.9118 | 35 465 | 290 328 | 31 231 | 3 054 |
| 05_h.jpg | U-Net | 0.9624 | 0.6787 | 0.9964 | 0.8224 | 0.8376 | 26 144 | 320 392 | 1 167 | 12 375 |

### Średnie wyniki w przekroju metod

| Metoda | Śr. Accuracy | Śr. Sensitivity | Śr. Specificity | Śr. G-Mean | Śr. Arith. Mean |
|--------|-------------|-----------------|-----------------|------------|-----------------|
| **Frangi** | 0.9496 | 0.6623 | 0.9781 | 0.8043 | 0.8202 |
| **Random Forest** | 0.8717 | 0.9066 | 0.8684 | 0.8866 | 0.8875 |
| **U-Net** | 0.9671 | 0.7223 | 0.9907 | 0.8456 | 0.8565 |

### Interpretacja miar:
- **Accuracy (Trafność)**: Odsetek poprawnie zaklasyfikowanych pikseli. Z uwagi na niezrównoważony rozkład klas (tło >> naczynia) ta miara jest naturalnie wysoka nawet dla słabych modeli.
- **Sensitivity (Czułość)**: Odsetek prawidłowo wykrytych pikseli naczyń (True Positive Rate). Kluczowa miara skuteczności.
- **Specificity (Swoistość)**: Odsetek prawidłowo zaklasyfikowanych pikseli tła (True Negative Rate).
- **G-Mean (Średnia geometryczna)**: √(Sensitivity × Specificity) — miara zrównoważona, odporna na niezrównoważony rozkład klas.
- **Arithmetic Mean**: (Sensitivity + Specificity) / 2 — alternatywna miara zrównoważona.

### Analiza wyników:

**Filtr Frangiego (baseline)**:
- Najniższa czułość (śr. 0.6623) — metoda traci wiele wąskich naczyń.
- Bardzo wysoka swoistość (śr. 0.9781) — mało fałszywych alarmów.
- Dobra trafność ogólna (0.9496), ale wynika to głównie z dominacji klasy tła.
- G-Mean (0.8043) wskazuje na niezrównoważoną klasyfikację.

**Random Forest**:
- Najwyższa czułość (śr. 0.9066) — najlepsza zdolność wykrywania naczyń spośród wszystkich metod.
- Najwyższy G-Mean (0.8866) — najlepsza zrównoważona miara.
- Najniższa swoistość (0.8684) i trafność (0.8717) — dużo fałszywych alarmów (FP), szczególnie widoczne w obrazach z retinopatią (04_dr: FP=77 870).
- Algorytm jest agresywny — nadmiernie klasyfikuje piksele jako naczynia.

**U-Net**:
- Najwyższa trafność (śr. 0.9671) i swoistość (śr. 0.9907) — najlepsza precyzja wśród metod.
- Czułość wyższa niż Frangi (0.7223 vs 0.6623), ale niższa niż RF — sieć jest ostrożna.
- G-Mean (0.8456) — pośrednia wartość, lepsze zrównoważenie niż Frangi.
- Bardzo niski FP — U-Net generuje mało fałszywych alarmów (np. 05_h: zaledwie 1 167 FP).
- Model mógłby osiągnąć jeszcze lepsze wyniki z dłuższym treningiem, większą liczbą danych lub augmentacją.

---

## 6. Wnioski

1. **Klasyczne przetwarzanie obrazu (Frangi)** stanowi solidny baseline o wysokiej swoistości, ale traci wąskie naczynia krwionośne. Skuteczność jest silnie zależna od ręcznego doboru progów binaryzacji.
2. **Random Forest** osiąga najwyższe wartości czułości i G-Mean, co czyni go najlepszą metodą pod względem wykrywania naczyń. Wadą jest duża liczba fałszywych alarmów — model agresywnie klasyfikuje piksele jako naczynia, co obniża swoistość.
3. **U-Net** osiąga najwyższą trafność i swoistość spośród wszystkich metod, generując bardzo czyste maski z minimalną liczbą fałszywych alarmów. Czułość jest wyższa niż Frangiego, ale niższa niż Random Forest. Dalsze ulepszenia (więcej epok, augmentacja danych, wyższe rozdzielczości) mogłyby poprawić czułość.
4. **Żadna metoda nie dominuje bezwzględnie** we wszystkich miarach — wybór optymalnej metody zależy od zastosowania klinicznego (czy ważniejsze jest wykrycie wszystkich naczyń, czy unikanie fałszywych alarmów).
5. Zastosowanie kombinacji **BCE + Dice Loss** okazało się skuteczne w radzeniu sobie z silnym niezrównoważeniem klas (naczynia stanowią ok. 8% pikseli).
6. Miary zrównoważone (**G-Mean, średnia arytmetyczna czułości i swoistości**) są kluczowe do oceny jakości w zadaniach z niezrównoważonym rozkładem klas, ponieważ sama trafność (accuracy) jest myląco wysoka.
