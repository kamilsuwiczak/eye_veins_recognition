import streamlit as st
import cv2
import numpy as np
import os
from skimage.filters import frangi

IMG_DIR = "data/images/"
FOV_DIR = "data/mask/"   
GT_DIR = "data/manual1/" 

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

def process_fundus_image(img_rgb, fov_mask, frangi_scale_range, frangi_scale_step, threshold, fov_margin):
    b, g, r = cv2.split(img_rgb)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(g)
    
    if fov_mask is not None:
        _, fov_mask_bin = cv2.threshold(fov_mask, 127, 255, cv2.THRESH_BINARY)
    else:
        fov_mask_bin = None

    vessels = frangi(
        g_clahe, 
        sigmas=np.arange(frangi_scale_range[0], frangi_scale_range[1], frangi_scale_step), 
        black_ridges=True
    )
    
    vessels_norm = cv2.normalize(vessels, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    vessels_norm = np.uint8(vessels_norm)
    
    eroded_fov = None
    if fov_mask_bin is not None:
        if fov_margin > 0:
            kernel_size = fov_margin * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            eroded_fov = cv2.erode(fov_mask_bin, kernel, iterations=1)
        else:
            eroded_fov = fov_mask_bin
            
        vessels_norm = cv2.bitwise_and(vessels_norm, vessels_norm, mask=eroded_fov)
        
        g_clahe = cv2.bitwise_and(g_clahe, g_clahe, mask=fov_mask_bin)
        
    _, binary_mask = cv2.threshold(vessels_norm, threshold, 255, cv2.THRESH_BINARY)
        
    return g_clahe, vessels_norm, binary_mask, eroded_fov

def calculate_metrics(pred_mask, true_mask, fov_mask=None):
    pred = (pred_mask > 127).astype(np.uint8)
    true = (true_mask > 127).astype(np.uint8)
    
    if fov_mask is not None:
        valid_pixels = (fov_mask > 127)
        pred = pred[valid_pixels]
        true = true[valid_pixels]
    else:
        pred = pred.flatten()
        true = true.flatten()
        
    TP = np.sum((pred == 1) & (true == 1))
    TN = np.sum((pred == 0) & (true == 0))
    FP = np.sum((pred == 1) & (true == 0))
    FN = np.sum((pred == 0) & (true == 1))
    
    total_pixels = TP + TN + FP + FN
    actual_positives = TP + FN
    actual_negatives = TN + FP
    
    accuracy = (TP + TN) / total_pixels if total_pixels > 0 else 0
    sensitivity = TP / actual_positives if actual_positives > 0 else 0
    specificity = TN / actual_negatives if actual_negatives > 0 else 0
    
    arithmetic_mean = (sensitivity + specificity) / 2.0
    geometric_mean = np.sqrt(sensitivity * specificity)
    
    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Arithmetic Mean": arithmetic_mean,
        "Geometric Mean": geometric_mean
    }

def main():
    st.set_page_config(layout="wide")
    st.title("Detekcja Naczyń Krwionośnych Dna Oka")
    st.markdown("Aplikacja porównująca klasyczne przetwarzanie obrazu z maskami eksperckimi.")

    if not all(os.path.exists(d) for d in [IMG_DIR, FOV_DIR, GT_DIR]):
        st.error(f"Upewnij się, że foldery `{IMG_DIR}`, `{FOV_DIR}` i `{GT_DIR}` istnieją w katalogu projektu!")
        return

    image_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    if not image_files:
        st.warning(f"Brak obrazów w folderze {IMG_DIR}.")
        return

    st.sidebar.header("Ustawienia i Wybór")
    selected_file = st.sidebar.selectbox("Wybierz obraz do analizy:", image_files)
    
    st.sidebar.subheader("Rozmiar przetwarzania")
    max_dim = st.sidebar.slider(
        "Maksymalny wymiar obrazu (px)", 
        min_value=800, max_value=4800, value=800, step=800,
    )
    
    st.sidebar.subheader("Parametry filtra Frangiego")
    scale_min = st.sidebar.slider("Minimalna skala (Sigma)", 1.0, 10.0, 1.5, 0.5)
    scale_max = st.sidebar.slider("Maksymalna skala (Sigma)", 2.0, 20.0, 4.0, 0.5)
    scale_step = st.sidebar.slider("Krok (Sigma)", 0.1, 3.0, 0.1, 0.1)
    
    st.sidebar.subheader("Postprocessing")
    threshold = st.sidebar.slider("Próg binaryzacji (Threshold)", 0, 255, 15)
    fov_margin = st.sidebar.slider("Odcięcie marginesu FOV (piksele)", 0, 150, 15)

    base_name = os.path.splitext(selected_file)[0]
    img_path = os.path.join(IMG_DIR, selected_file)
    fov_path = os.path.join(FOV_DIR, f"{base_name}_mask.tif") 
    
    gt_path = os.path.join(GT_DIR, selected_file)
    if not os.path.exists(gt_path):
        gt_path = os.path.join(GT_DIR, f"{base_name}.tif")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(GT_DIR, f"{base_name}_manual1.tif")

    original_img = load_image(img_path, grayscale=False, max_dim=max_dim)
    fov_mask = load_image(fov_path, grayscale=True, max_dim=max_dim)
    expert_mask = load_image(gt_path, grayscale=True, max_dim=max_dim)

    if original_img is None:
        st.error("Błąd wczytywania obrazu oryginalnego.")
        return

    with st.spinner(f"Przetwarzanie obrazu w rozdzielczości {max_dim}px (może to potrwać dłuższą chwilę)..."):
        g_clahe, frangi_raw, my_mask, eroded_fov = process_fundus_image(
            original_img, fov_mask, 
            frangi_scale_range=(scale_min, scale_max), 
            frangi_scale_step=scale_step, 
            threshold=threshold,
            fov_margin=fov_margin
        )


    st.subheader("Etapy przetwarzania")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(original_img, caption=f"1. Oryginał (Przeskalowany do max {max_dim}px)", use_container_width=True)
    with col2:
        st.image(g_clahe, caption="2. Zielony kanał + CLAHE", use_container_width=True, clamp=True)
    with col3:
        st.image(frangi_raw, caption="3. Wynik Frangiego (Surowy)", use_container_width=True, clamp=True)

    st.markdown("---")
    st.subheader("Porównanie masek")
    
    col4, col5 = st.columns(2)
    with col4:
        st.image(my_mask, caption=f"Twoja wygenerowana maska (Próg: {threshold})", use_container_width=True, clamp=True)
    
    with col5:
        if expert_mask is not None:
            st.image(expert_mask, caption="Maska ekspercka (Ground Truth)", use_container_width=True, clamp=True)
        else:
            st.warning("Nie znaleziono maski eksperckiej.")

    if expert_mask is not None:
        st.markdown("---")
        st.subheader("Wyniki Ewaluacji")
        
        if my_mask.shape != expert_mask.shape:
            expert_mask = cv2.resize(expert_mask, (my_mask.shape[1], my_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        metrics = calculate_metrics(my_mask, expert_mask, fov_mask=eroded_fov)
        
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Trafność (Accuracy)", f"{metrics['Accuracy']:.4f}")
        m_col2.metric("Czułość (Sensitivity)", f"{metrics['Sensitivity']:.4f}")
        m_col3.metric("Swoistość (Specificity)", f"{metrics['Specificity']:.4f}")
        m_col4.metric("Średnia Arytm. (G-Mean)", f"{metrics['Arithmetic Mean']:.4f}")
        m_col5.metric("Średnia Geom.", f"{metrics['Geometric Mean']:.4f}")

        st.markdown("**Macierz pomyłek (Confusion Matrix):**")
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        c_col1.metric("Prawdziwie Pozytywne (TP)", metrics['TP'], help="Poprawnie rozpoznane naczynia")
        c_col2.metric("Prawdziwie Negatywne (TN)", metrics['TN'], help="Poprawnie rozpoznane tło")
        c_col3.metric("Fałszywie Pozytywne (FP)", metrics['FP'], help="Tło błędnie rozpoznane jako naczynie")
        c_col4.metric("Fałszywie Negatywne (FN)", metrics['FN'], help="Naczynie błędnie rozpoznane jako tło")

if __name__ == "__main__":
    main()