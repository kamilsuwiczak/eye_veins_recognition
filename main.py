import streamlit as st
import cv2
import numpy as np
import os
import joblib 
from skimage.filters import frangi
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from imblearn.metrics import specificity_score, geometric_mean_score
from sklearn.ensemble import RandomForestClassifier

IMG_DIR = "data/images/"
FOV_DIR = "data/mask/"   
GT_DIR = "data/manual1/" 
MODEL_PATH = "saved_rf_model.pkl" 

@st.cache_resource
def get_rf_model(max_depth):
    return RandomForestClassifier(n_estimators=50, max_depth=max_depth, random_state=42, n_jobs=-1)

def load_image(path, grayscale=False, max_dim=800):
    if not os.path.exists(path):
        return None
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    if img is not None:
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            interp = cv2.INTER_NEAREST if grayscale else cv2.INTER_AREA
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return img

def extract_dense_features(img_rgb, fov_mask, frangi_params):
    b, g, r = cv2.split(img_rgb)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(g)
    
    if fov_mask is not None:
        _, fov_mask_bin = cv2.threshold(fov_mask, 127, 255, cv2.THRESH_BINARY)
    else:
        fov_mask_bin = np.ones_like(g_clahe) * 255
        
    h_img, w_img = g_clahe.shape
    scale_factor = max(h_img, w_img) / 800.0
    k_val = int(5 * scale_factor)
    k_size = k_val if k_val % 2 == 1 else k_val + 1 
    k_size = max(3, k_size) 
    
    g_masked = cv2.bitwise_and(g_clahe, g_clahe, mask=fov_mask_bin)

    vessels = frangi(
        g_masked, 
        sigmas=np.arange(frangi_params[0], frangi_params[1], frangi_params[2]), 
        black_ridges=True
    )
    f1 = cv2.normalize(vessels, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    f2 = cv2.morphologyEx(g_clahe, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)

    sobelx = cv2.Sobel(g_clahe, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(g_clahe, cv2.CV_32F, 0, 1, ksize=3)
    f3 = cv2.magnitude(sobelx, sobely)
    f3 = cv2.normalize(f3, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.float32)

    g_float = g_clahe.astype(np.float32)
    mean_val = cv2.blur(g_float, (k_size, k_size))
    mean_sq = cv2.blur(g_float**2, (k_size, k_size))
    f4 = np.sqrt(np.maximum(0, mean_sq - mean_val**2))

    features = np.dstack([f1, f2, f3, f4])
    
    return g_clahe, features, fov_mask_bin, f1.astype(np.uint8)

def prepare_ml_data(features, expert_mask, fov_mask, samples_per_class=5000):
    X_flat = features.reshape(-1, features.shape[-1])
    y_flat = (expert_mask.flatten() > 127).astype(np.uint8)
    fov_flat = (fov_mask.flatten() > 127)

    valid_indices = np.where(fov_flat)[0]
    
    X_valid = X_flat[valid_indices]
    y_valid = y_flat[valid_indices]

    pos_indices = np.where(y_valid == 1)[0]
    neg_indices = np.where(y_valid == 0)[0]

    n_pos = min(samples_per_class, len(pos_indices))
    n_neg = min(samples_per_class, len(neg_indices))

    if n_pos == 0 or n_neg == 0:
        return None, None

    np.random.seed(42)
    sampled_pos = np.random.choice(pos_indices, n_pos, replace=False)
    sampled_neg = np.random.choice(neg_indices, n_neg, replace=False)

    sampled_indices = np.concatenate([sampled_pos, sampled_neg])

    X_train = X_valid[sampled_indices]
    y_train = y_valid[sampled_indices]

    return X_train, y_train

def calculate_metrics(pred_mask, true_mask, fov_mask=None):
    pred = (pred_mask > 127).astype(np.uint8)
    true = (true_mask > 127).astype(np.uint8)
    
    if fov_mask is not None:
        valid_pixels = (fov_mask > 127)
        y_pred = pred[valid_pixels]
        y_true = true[valid_pixels]
    else:
        y_pred = pred.flatten()
        y_true = true.flatten()
        
    if len(y_true) == 0:
        return None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = specificity_score(y_true, y_pred)
    g_mean = geometric_mean_score(y_true, y_pred)
    arithmetic_mean = (sensitivity + specificity) / 2.0
    
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Accuracy": accuracy, "Sensitivity": sensitivity, "Specificity": specificity,
        "Arithmetic Mean": arithmetic_mean, "Geometric Mean": g_mean
    }

def main():
    st.set_page_config(layout="wide")
    st.title("👁️ Detekcja Naczyń Krwionośnych Dna Oka")
    st.markdown("Aplikacja do klasycznego przetwarzania obrazu i Uczenia Maszynowego.")

    if not all(os.path.exists(d) for d in [IMG_DIR, FOV_DIR, GT_DIR]):
        st.error(f"Upewnij się, że foldery `{IMG_DIR}`, `{FOV_DIR}` i `{GT_DIR}` istnieją!")
        return

    image_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    if not image_files:
        st.warning(f"Brak obrazów w folderze {IMG_DIR}.")
        return

    st.sidebar.header("Tryb działania")
    mode = st.sidebar.radio("Wybierz metodę analizy:", ["Przetwarzanie (Filtr Frangiego)", "Uczenie Maszynowe (Random Forest)"])

    st.sidebar.subheader("Rozmiar przetwarzania")
    max_dim = st.sidebar.slider("Maksymalny wymiar obrazu (px)", min_value=800, max_value=4800, value=800, step=800)

    st.sidebar.subheader("Parametry filtra Frangiego")
    scale_min = st.sidebar.slider("Minimalna skala (Sigma)", 1.0, 15.0, 1.5, 0.5)
    scale_max = st.sidebar.slider("Maksymalna skala (Sigma)", 2.0, 30.0, 4.0, 0.5)
    scale_step = st.sidebar.slider("Krok (Sigma)", 0.1, 3.0, 0.5, 0.1)

    if mode == "Przetwarzanie (Filtr Frangiego)":
        st.sidebar.header("Ustawienia Obrazu")
        selected_file = st.sidebar.selectbox("Wybierz obraz do analizy:", image_files)
        
        st.sidebar.subheader("Postprocessing (Tylko Klasyczne)")
        threshold = st.sidebar.slider("Próg binaryzacji (Threshold)", 0, 255, 15)
        fov_margin = st.sidebar.slider("Odcięcie marginesu FOV (piksele)", 0, 150, 15)

        base_name = os.path.splitext(selected_file)[0]
        original_img = load_image(os.path.join(IMG_DIR, selected_file), max_dim=max_dim)
        fov_mask = load_image(os.path.join(FOV_DIR, f"{base_name}_mask.tif"), grayscale=True, max_dim=max_dim)
        expert_mask = load_image(os.path.join(GT_DIR, selected_file) if os.path.exists(os.path.join(GT_DIR, selected_file)) else os.path.join(GT_DIR, f"{base_name}.tif"), grayscale=True, max_dim=max_dim)

        if original_img is None: return

        with st.spinner("Przetwarzanie (Frangi)..."):
            b, g, r = cv2.split(original_img)
            g_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
            
            fov_mask_bin = cv2.threshold(fov_mask, 127, 255, cv2.THRESH_BINARY)[1] if fov_mask is not None else None
            
            vessels_raw = frangi(g_clahe, sigmas=np.arange(scale_min, scale_max, scale_step), black_ridges=True)
            vessels_norm = np.uint8(cv2.normalize(vessels_raw, None, 0, 255, cv2.NORM_MINMAX))
            
            frangi_display = vessels_norm.copy()
            clahe_display = g_clahe.copy()
            
            eroded_fov = None
            if fov_mask_bin is not None:
                if fov_margin > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fov_margin*2+1, fov_margin*2+1))
                    eroded_fov = cv2.erode(fov_mask_bin, k, iterations=1)
                else:
                    eroded_fov = fov_mask_bin
                vessels_norm = cv2.bitwise_and(vessels_norm, vessels_norm, mask=eroded_fov)
                clahe_display = cv2.bitwise_and(clahe_display, clahe_display, mask=fov_mask_bin)
                
            _, my_mask = cv2.threshold(vessels_norm, threshold, 255, cv2.THRESH_BINARY)

        st.subheader("Etapy przetwarzania")
        c1, c2, c3 = st.columns(3)
        with c1: st.image(original_img, caption="1. Oryginał", use_container_width=True)
        with c2: st.image(clahe_display, caption="2. Zielony kanał + CLAHE", use_container_width=True, clamp=True)
        with c3: st.image(frangi_display, caption="3. Wynik Frangiego (Surowy)", use_container_width=True, clamp=True)

        st.markdown("---")
        st.subheader("Porównanie masek")
        c4, c5 = st.columns(2)
        with c4: st.image(my_mask, caption="Maska wygenerowana", use_container_width=True, clamp=True)
        with c5: 
            if expert_mask is not None: st.image(expert_mask, caption="Maska ekspercka", use_container_width=True, clamp=True)

        if expert_mask is not None:
            if my_mask.shape != expert_mask.shape:
                expert_mask = cv2.resize(expert_mask, (my_mask.shape[1], my_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            metrics = calculate_metrics(my_mask, expert_mask, fov_mask=eroded_fov)
            
            st.markdown("---")
            st.subheader("Ewaluacja")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            m2.metric("Sensitivity", f"{metrics['Sensitivity']:.4f}")
            m3.metric("Specificity", f"{metrics['Specificity']:.4f}")
            m4.metric("G-Mean", f"{metrics['Geometric Mean']:.4f}")
            m5.metric("TN (Prawidłowe Tło)", metrics['TN'])


    elif mode == "Uczenie Maszynowe (Random Forest)":
        st.sidebar.header("Zarządzanie Modelem")
        ml_action = st.sidebar.radio("Opcje modelu:", ["Trenuj i zapisz nowy", "Wczytaj z dysku i testuj"])
        
        rf_max_depth = st.sidebar.slider("Maksymalna głębokość drzewa (Max Depth)", 2, 30, 8, 1, 
                                         help="Mniejsza wartość = mniejsze przeuczenie (overfitting), ale model będzie łapał mniej detali.")
        
        model_exists = os.path.exists(MODEL_PATH)
        
        if ml_action == "Wczytaj z dysku i testuj" and not model_exists:
            st.sidebar.error(f"Brak pliku `{MODEL_PATH}`! Najpierw wytrenuj model.")
            
        st.sidebar.header("Wybór zbiorów")
        if ml_action == "Trenuj i zapisz nowy":
            train_file = st.sidebar.selectbox("Zdjęcie TRENINGOWE (Uczenie):", image_files, index=0)
            samples = st.sidebar.slider("Ilość próbek na klasę (Undersampling)", 1000, 20000, 5000, 1000)
            
        test_file = st.sidebar.selectbox("Zdjęcie TESTOWE (Ewaluacja):", image_files, index=1 if len(image_files)>1 else 0)

        if ml_action == "Trenuj i zapisz nowy" and train_file == test_file:
            st.warning("⚠️ OSTRZEŻENIE: Uczysz i testujesz model na tym samym obrazie. To spowoduje przeuczenie (Overfitting).")

        btn_text = "Uruchom Trening i Test" if ml_action == "Trenuj i zapisz nowy" else "Uruchom Testowanie (Pomiń trening)"
        
        if st.sidebar.button(btn_text):
            
            model = None
            frangi_params_dynamic = (scale_min, scale_max, scale_step)
            
            if ml_action == "Trenuj i zapisz nowy":
                with st.spinner(f"Krok 1/3: Ekstrakcja cech (Train). Używana rozdzielczość: {max_dim}px..."):
                    t_base = os.path.splitext(train_file)[0]
                    img_train = load_image(os.path.join(IMG_DIR, train_file), max_dim=max_dim)
                    fov_train = load_image(os.path.join(FOV_DIR, f"{t_base}_mask.tif"), grayscale=True, max_dim=max_dim)
                    gt_train = load_image(os.path.join(GT_DIR, train_file) if os.path.exists(os.path.join(GT_DIR, train_file)) else os.path.join(GT_DIR, f"{t_base}.tif"), grayscale=True, max_dim=max_dim)
                    
                    _, train_features, train_fov_bin, _ = extract_dense_features(img_train, fov_train, frangi_params=frangi_params_dynamic)
                    X_train, y_train = prepare_ml_data(train_features, gt_train, train_fov_bin, samples)

                with st.spinner("Krok 2/3: Trening i zapisywanie modelu..."):
                    model = get_rf_model(rf_max_depth)
                    model.fit(X_train, y_train)
                    joblib.dump(model, MODEL_PATH)
                    st.success(f"Model pomyślnie zapisany jako `{MODEL_PATH}`!")
            else:
                if model_exists:
                    with st.spinner("Ładowanie modelu z dysku..."):
                        model = joblib.load(MODEL_PATH)
                        st.success("Wczytano gotowy model. Przechodzę do predykcji.")
                else:
                    st.error("Przerwano: Nie znaleziono modelu na dysku.")
                    return

            if model is not None:
                step_text = "Krok 3/3" if ml_action == "Trenuj i zapisz nowy" else "Krok 1/1"
                with st.spinner(f"{step_text}: Ekstrakcja cech i predykcja (Test)..."):
                    test_base = os.path.splitext(test_file)[0]
                    img_test = load_image(os.path.join(IMG_DIR, test_file), max_dim=max_dim)
                    fov_test = load_image(os.path.join(FOV_DIR, f"{test_base}_mask.tif"), grayscale=True, max_dim=max_dim)
                    gt_test = load_image(os.path.join(GT_DIR, test_file) if os.path.exists(os.path.join(GT_DIR, test_file)) else os.path.join(GT_DIR, f"{test_base}.tif"), grayscale=True, max_dim=max_dim)
                    
                    test_clahe, test_features, test_fov_bin, test_frangi_raw = extract_dense_features(img_test, fov_test, frangi_params=frangi_params_dynamic)
                    
                    X_test_flat = test_features.reshape(-1, test_features.shape[-1])
                    test_valid_idx = np.where(test_fov_bin.flatten() > 127)[0]
                    
                    X_test_valid = X_test_flat[test_valid_idx]
                    y_pred_valid = model.predict(X_test_valid)
                    
                    ml_mask_flat = np.zeros(X_test_flat.shape[0], dtype=np.uint8)
                    ml_mask_flat[test_valid_idx] = y_pred_valid * 255
                    ml_mask_2d = ml_mask_flat.reshape(img_test.shape[:2])

                st.subheader(f"Testowanie na niezależnym obrazie: {test_file}")
                
                st.markdown("**Wybrane cechy wejściowe dla modelu:**")
                c1, c2, c3 = st.columns(3)
                with c1: st.image(img_test, caption="Obraz Oryginalny", use_container_width=True)
                with c2: st.image(test_clahe, caption="Zielony kanał (CLAHE)", use_container_width=True, clamp=True)
                with c3: st.image(test_frangi_raw, caption="Filtr Frangiego", use_container_width=True, clamp=True)

                st.markdown("---")
                st.subheader("Porównanie masek")
                
                c4, c5 = st.columns(2)
                with c4: 
                    st.image(ml_mask_2d, caption="Twoja wygenerowana maska (Random Forest)", use_container_width=True, clamp=True)
                
                with c5:
                    if gt_test is not None:
                        st.image(gt_test, caption="Maska ekspercka (Ground Truth)", use_container_width=True, clamp=True)
                    else:
                        st.warning("Nie znaleziono maski eksperckiej.")

                if gt_test is not None:
                    st.markdown("---")
                    st.subheader("Ewaluacja Predykcji ML")
                    
                    if ml_mask_2d.shape != gt_test.shape:
                        gt_test = cv2.resize(gt_test, (ml_mask_2d.shape[1], ml_mask_2d.shape[0]), interpolation=cv2.INTER_NEAREST)

                    ml_metrics = calculate_metrics(ml_mask_2d, gt_test, fov_mask=test_fov_bin)
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Accuracy", f"{ml_metrics['Accuracy']:.4f}")
                    col2.metric("Sensitivity", f"{ml_metrics['Sensitivity']:.4f}")
                    col3.metric("Specificity", f"{ml_metrics['Specificity']:.4f}")
                    col4.metric("G-Mean", f"{ml_metrics['Geometric Mean']:.4f}")
                    col5.metric("Fałszywie Pozytywne (FP)", ml_metrics['FP'])

if __name__ == "__main__":
    main()