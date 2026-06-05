import os
import cv2
import numpy as np
import streamlit as st

from src.config import Config
from src.image_loader import ImageLoader
from src.metrics import MetricsCalculator
from src.preprocessing import Preprocessor
from src.detectors import FrangiDetector, RFDetector, UNetDetector


class App:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.loader = None
        self.image_files = Config.get_image_files()

    def run(self):
        st.set_page_config(layout="wide")
        st.title("👁️ Detekcja Naczyń Krwionośnych Dna Oka")

        if not all(os.path.exists(d) for d in [Config.IMG_DIR, Config.FOV_DIR, Config.GT_DIR]):
            st.error(f"Upewnij się, że foldery danych istnieją!")
            return

        if not self.image_files:
            st.warning("Brak obrazów w folderze danych.")
            return

        st.sidebar.header("Tryb działania")
        mode = st.sidebar.radio("Wybierz metodę analizy:", [
            "Przetwarzanie (Filtr Frangiego)",
            "Uczenie Maszynowe (Random Forest)",
            "Głęboka Sieć Neuronowa (U-Net)",
        ])

        st.sidebar.subheader("Rozmiar przetwarzania")
        max_dim = st.sidebar.slider("Maksymalny wymiar obrazu (px)", 800, 4800, 800, 800)
        self.loader = ImageLoader(max_dim)

        st.sidebar.subheader("Parametry filtra Frangiego")
        scale_min = st.sidebar.slider("Minimalna skala (Sigma)", 1.0, 15.0, 1.5, 0.5)
        scale_max = st.sidebar.slider("Maksymalna skala (Sigma)", 2.0, 30.0, 4.0, 0.5)
        scale_step = st.sidebar.slider("Krok (Sigma)", 0.1, 3.0, 0.5, 0.1)
        sigmas = (scale_min, scale_max, scale_step)

        if mode == "Przetwarzanie (Filtr Frangiego)":
            self._run_frangi(sigmas)
        elif mode == "Uczenie Maszynowe (Random Forest)":
            self._run_rf(sigmas)
        else:
            self._run_unet()

    def _load_triplet(self, image_name):
        base = os.path.splitext(image_name)[0]
        img = self.loader.load(os.path.join(Config.IMG_DIR, image_name))
        fov = self.loader.load(Config.get_fov_path(image_name), grayscale=True)
        gt = self.loader.load(Config.get_gt_path(image_name), grayscale=True)
        return img, fov, gt

    def _show_metrics(self, pred_mask, gt_mask, fov_mask):
        if gt_mask is None:
            return
        if pred_mask.shape != gt_mask.shape:
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        metrics = MetricsCalculator.calculate(pred_mask, gt_mask, fov_mask)
        if metrics is None:
            return

        st.markdown("---")
        st.subheader("Ewaluacja")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        c2.metric("Sensitivity", f"{metrics['Sensitivity']:.4f}")
        c3.metric("Specificity", f"{metrics['Specificity']:.4f}")
        c4.metric("G-Mean", f"{metrics['Geometric Mean']:.4f}")
        c5.metric("Arith. Mean", f"{metrics['Arithmetic Mean']:.4f}")

        st.markdown("**Macierz pomyłek:**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("TP", metrics["TP"])
        m2.metric("TN", metrics["TN"])
        m3.metric("FP", metrics["FP"])
        m4.metric("FN", metrics["FN"])

    def _run_frangi(self, sigmas):
        st.sidebar.header("Ustawienia obrazu")
        selected = st.sidebar.selectbox("Wybierz obraz:", self.image_files)

        st.sidebar.subheader("Postprocessing")
        threshold = st.sidebar.slider("Próg binaryzacji", 0, 255, 15)
        fov_margin = st.sidebar.slider("Odcięcie marginesu FOV (px)", 0, 150, 15)

        img, fov, gt = self._load_triplet(selected)
        if img is None:
            return

        with st.spinner("Przetwarzanie..."):
            detector = FrangiDetector(self.preprocessor)
            mask, eroded_fov, g_clahe, frangi_raw = detector.predict(
                img, fov, sigmas, threshold, fov_margin
            )

        st.subheader("Etapy przetwarzania")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(img, caption="1. Oryginał", use_container_width=True)
        with c2:
            clahe_display = self.preprocessor.apply_fov(g_clahe, self.preprocessor.binarize_fov(fov))
            st.image(clahe_display, caption="2. Zielony kanał + CLAHE", use_container_width=True, clamp=True)
        with c3:
            st.image(frangi_raw, caption="3. Wynik Frangiego", use_container_width=True, clamp=True)

        st.markdown("---")
        st.subheader("Porównanie masek")
        c4, c5 = st.columns(2)
        with c4:
            st.image(mask, caption="Maska wygenerowana", use_container_width=True, clamp=True)
        with c5:
            if gt is not None:
                st.image(gt, caption="Maska ekspercka", use_container_width=True, clamp=True)

        self._show_metrics(mask, gt, eroded_fov)

    def _run_rf(self, sigmas):
        st.sidebar.header("Zarządzanie modelem")
        action = st.sidebar.radio("Opcje modelu:", ["Trenuj i zapisz nowy", "Wczytaj z dysku i testuj"])

        max_depth = st.sidebar.slider("Max Depth", 2, 30, 8, 1)
        model_exists = os.path.exists(Config.RF_MODEL_PATH)

        if action == "Wczytaj z dysku i testuj" and not model_exists:
            st.sidebar.error("Brak modelu na dysku! Najpierw wytrenuj.")

        st.sidebar.header("Wybór zbiorów")
        if action == "Trenuj i zapisz nowy":
            train_file = st.sidebar.selectbox("Obraz treningowy:", self.image_files, index=0)
            samples = st.sidebar.slider("Próbki na klasę (undersampling)", 1000, 20000, 5000, 1000)

        test_file = st.sidebar.selectbox("Obraz testowy:", self.image_files,
                                          index=1 if len(self.image_files) > 1 else 0)

        if action == "Trenuj i zapisz nowy" and train_file == test_file:
            st.warning("⚠️ Ten sam obraz treningowy i testowy — ryzyko overfittingu.")

        btn = "Uruchom Trening i Test" if action == "Trenuj i zapisz nowy" else "Uruchom Testowanie"

        if not st.sidebar.button(btn):
            return

        detector = RFDetector(self.preprocessor)
        detector.create_model(max_depth)

        if action == "Trenuj i zapisz nowy":
            with st.spinner("Trening Random Forest..."):
                img_train, fov_train, gt_train = self._load_triplet(train_file)
                detector.train(img_train, fov_train, gt_train, sigmas, samples)
                detector.save()
                st.success(f"Model zapisany do `{Config.RF_MODEL_PATH}`")
        else:
            if not detector.load():
                st.error("Nie znaleziono modelu.")
                return
            st.success("Wczytano model z dysku.")

        with st.spinner("Predykcja..."):
            img_test, fov_test, gt_test = self._load_triplet(test_file)
            mask, fov_bin, g_clahe, frangi_raw = detector.predict(img_test, fov_test, sigmas)

        st.subheader(f"Testowanie: {test_file}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(img_test, caption="Oryginał", use_container_width=True)
        with c2:
            st.image(g_clahe, caption="Zielony kanał (CLAHE)", use_container_width=True, clamp=True)
        with c3:
            st.image(frangi_raw, caption="Filtr Frangiego", use_container_width=True, clamp=True)

        st.markdown("---")
        st.subheader("Porównanie masek")
        c4, c5 = st.columns(2)
        with c4:
            st.image(mask, caption="Maska (Random Forest)", use_container_width=True, clamp=True)
        with c5:
            if gt_test is not None:
                st.image(gt_test, caption="Maska ekspercka", use_container_width=True, clamp=True)

        self._show_metrics(mask, gt_test, fov_bin)

    def _run_unet(self):
        st.sidebar.header("Zarządzanie modelem U-Net")
        action = st.sidebar.radio("Opcje modelu:", [
            "Wczytaj z dysku i testuj",
            "Trenuj nowy model",
        ])

        unet_exists = os.path.exists(Config.UNET_MODEL_PATH)
        if action == "Wczytaj z dysku i testuj" and not unet_exists:
            st.sidebar.warning("Brak modelu! Najpierw wytrenuj (python train_unet.py).")

        st.sidebar.header("Wybór zbiorów")
        if action == "Trenuj nowy model":
            train_file = st.sidebar.selectbox("Obraz treningowy (U-Net):", self.image_files, index=0)
            unet_epochs = st.sidebar.slider("Liczba epok", 5, 50, 20, 5)
            unet_crops = st.sidebar.slider("Liczba wycinków na epokę", 10, 100, 40, 10)

        test_file = st.sidebar.selectbox("Obraz testowy (U-Net):", self.image_files,
                                          index=min(3, len(self.image_files) - 1))
        unet_threshold = st.sidebar.slider("Próg binaryzacji U-Net", 0.0, 1.0, 0.5, 0.05)

        btn = "🚀 Trenuj i Testuj" if action == "Trenuj nowy model" else "🔍 Testuj"

        if not st.sidebar.button(btn):
            return

        detector = UNetDetector(self.preprocessor)

        if action == "Trenuj nowy model":
            st.subheader(f"Trening U-Net: {train_file}")

            with st.spinner("Ładowanie danych..."):
                img_train, fov_train, gt_train = self._load_triplet(train_file)
                if img_train is None or gt_train is None:
                    st.error("Nie udało się załadować danych treningowych.")
                    return
                g_clahe = self.preprocessor.extract_green_clahe(img_train)
                fov_bin = self.preprocessor.binarize_fov(fov_train)
                g_clahe = self.preprocessor.apply_fov(g_clahe, fov_bin)

            if unet_exists:
                detector.load()
                st.info("Wczytano istniejące wagi (fine-tuning).")

            progress_bar = st.progress(0)
            loss_text = st.empty()

            def on_progress(epoch, total, loss):
                progress_bar.progress((epoch + 1) / total)
                loss_text.markdown(f"**Epoka {epoch + 1}/{total}** — Loss: `{loss:.4f}`")

            detector.train_on_image(g_clahe, gt_train, fov_bin, unet_epochs, unet_crops,
                                    progress_callback=on_progress)
            detector.save()
            st.success(f"Model zapisany do `{Config.UNET_MODEL_PATH}`")
        else:
            if not detector.load():
                st.error("Nie znaleziono modelu.")
                return
            st.success("Wczytano model U-Net z dysku.")

        st.markdown("---")
        st.subheader(f"Predykcja: {test_file}")

        with st.spinner("Przetwarzanie..."):
            img_test, fov_test, gt_test = self._load_triplet(test_file)
            if img_test is None:
                st.error("Nie udało się załadować obrazu testowego.")
                return
            mask, fov_bin, g_clahe, prob_map = detector.predict(img_test, fov_test, unet_threshold)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(img_test, caption="1. Oryginał", use_container_width=True)
        with c2:
            st.image(g_clahe, caption="2. Zielony kanał + CLAHE", use_container_width=True, clamp=True)
        with c3:
            prob_display = np.uint8(prob_map * 255)
            prob_colored = cv2.applyColorMap(prob_display, cv2.COLORMAP_JET)
            prob_colored = cv2.cvtColor(prob_colored, cv2.COLOR_BGR2RGB)
            st.image(prob_colored, caption="3. Mapa prawdopodobieństwa", use_container_width=True)

        st.markdown("---")
        st.subheader("Porównanie masek")
        c4, c5 = st.columns(2)
        with c4:
            st.image(mask, caption=f"Maska U-Net (próg={unet_threshold:.2f})",
                     use_container_width=True, clamp=True)
        with c5:
            if gt_test is not None:
                st.image(gt_test, caption="Maska ekspercka", use_container_width=True, clamp=True)

        self._show_metrics(mask, gt_test, fov_bin)


if __name__ == "__main__":
    App().run()