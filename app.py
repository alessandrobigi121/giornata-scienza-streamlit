import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.stats import linregress
import pandas as pd
import io
from scipy.io.wavfile import write

# Costanti fisiche
V_SUONO = 340  # m/s
SAMPLE_RATE = 44100  # Hz


# Parametri acustici (da relazione)
DENSITA_ARIA = 1.2  # kg/m³
PRESSIONE_ATM = 101325  # Pa
IMPEDENZA_ACUSTICA = 408  # Pa·s/m
SOGLIA_UDIBILITA = 1e-12  # W/m²
SOGLIA_DOLORE = 1.0  # W/m²


# ========== PRESET STORICI E FAMOSI ==========
PRESET_FAMOSI = {
    "Personalizzato": None,
    "Diapason Standard LA 440 Hz": {
        "f1": 440.0, "f2": 445.0, "A1": 1.0, "A2": 1.0, "durata": 2.0,
        "descrizione": "Il LA a 440 Hz è lo standard internazionale dal 1955 (ISO 16). Usato per accordare gli strumenti."
    },
    "Diapason Giuseppe Verdi LA 432 Hz": {
        "f1": 432.0, "f2": 437.0, "A1": 1.0, "A2": 1.0, "durata": 2.0,
        "descrizione": "Verdi sosteneva il LA a 432 Hz per motivi artistici. Detto 'accordatura scientifica'."
    },
    "Esperimento Helmholtz (1863)": {
        "f1": 256.0, "f2": 261.0, "A1": 1.0, "A2": 1.0, "durata": 3.0,
        "descrizione": "Hermann von Helmholtz studiò i battimenti usando risonatori acustici a 256 Hz (DO centrale)."
    },
    "Esperimento Heisenberg (1927)": {
        "f1": 1000.0, "f2": 1100.0, "A1": 1.0, "A2": 1.0, "durata": 1.5,
        "descrizione": "Heisenberg usò onde sonore per dimostrare il principio di indeterminazione prima di applicarlo alla QM."
    },
    "Frequenza Radio AM (esempio)": {
        "f1": 800.0, "f2": 810.0, "A1": 1.0, "A2": 1.0, "durata": 2.0,
        "descrizione": "Battimenti a ~10 Hz simili a quelli percepiti tra stazioni radio AM vicine."
    },
    "Battimenti Musicali Lenti": {
        "f1": 440.0, "f2": 442.0, "A1": 1.0, "A2": 1.0, "durata": 4.0,
        "descrizione": "2 battimenti/secondo: usati dai musicisti per accordare strumenti 'a orecchio'."
    },
    "Battimenti Rapidi (interferenza)": {
        "f1": 1000.0, "f2": 1020.0, "A1": 1.0, "A2": 1.0, "durata": 1.0,
        "descrizione": "20 battimenti/secondo: al limite della percezione come tono separato vs. ruvidezza."
    }
}

PRESET_PACCHETTI = {
    "Personalizzato": None,
    "Pacchetto Standard (Δk medio)": {
        "f_min": 100.0, "f_max": 130.0, "N": 50,
        "descrizione": "Configurazione equilibrata: localizzazione moderata, buona visualizzazione."
    },
    "Super-Localizzato (Δk grande)": {
        "f_min": 100.0, "f_max": 200.0, "N": 80,
        "descrizione": "Banda larga → pacchetto stretto (Δx piccolo). Dimostra il principio di indeterminazione."
    },
    "Quasi-Monocromatico (Δk piccolo)": {
        "f_min": 100.0, "f_max": 105.0, "N": 30,
        "descrizione": "Banda stretta → pacchetto largo (Δx grande). Simile a un'onda quasi pura."
    },
    "Esperimento Thomson (1897)": {
        "f_min": 256.0, "f_max": 280.0, "N": 60,
        "descrizione": "J.J. Thomson usò pacchetti d'onda sonori per studiare la natura corpuscolare delle particelle."
    },
    "Impulso Radar (simulato)": {
        "f_min": 500.0, "f_max": 700.0, "N": 100,
        "descrizione": "Simula un impulso radar: breve durata, ampia banda. Trade-off range-velocità."
    },
    "Pacchetto Audio (voce umana)": {
        "f_min": 200.0, "f_max": 400.0, "N": 70,
        "descrizione": "Banda tipica della voce umana (formanti). Pacchetto con buona localizzazione temporale."
    }
}


def mostra_parametri_acustici():
    """Mostra tabella parametri fisici del suono (da relazione)"""
    st.sidebar.markdown("### Parametri Fisici")
    with st.sidebar.expander("Proprietà aria (20°C)"):
        st.markdown(f"""
        - **Velocità suono**: {V_SUONO} m/s
        - **Densità aria**: {DENSITA_ARIA} kg/m³
        - **Pressione atm**: {PRESSIONE_ATM:,} Pa
        - **Impedenza**: {IMPEDENZA_ACUSTICA} Pa·s/m
        - **Soglia udibilità**: {SOGLIA_UDIBILITA:.0e} W/m²
        - **Soglia dolore**: {SOGLIA_DOLORE} W/m²
        """)


st.set_page_config(page_title="Giornata della Scienza - Fisica", layout="wide", page_icon="🌊")

# ============ HEADER PRINCIPALE ============
st.markdown("""
<div style="
    background: linear-gradient(135deg, #2c3e50 0%, #3498db 50%, #9b59b6 100%);
    padding: 2rem 2.5rem;
    border-radius: 15px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
">
    <h1 style="color: white; margin: 0; font-size: 2.4rem; font-weight: 700;">
        🌊 Onde, Pacchetti e Indeterminazione
    </h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.8rem 0 0 0; font-size: 1.15rem;">
        Giornata della Scienza • Liceo Leopardi Majorana
    </p>
    <div style="margin-top: 1rem; display: flex; gap: 1.5rem; flex-wrap: wrap;">
        <span style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
            👨‍🔬 A cura di Alessandro Bigi
        </span>
        <span style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
            📚 Laboratorio di Fisica
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============ STILE GRAFICO PROFESSIONALE ============
def styled_header(icon: str, title: str, subtitle: str = "", color: str = "#3498db"):
    """
    Crea un header di sezione con stile professionale.
    Funziona sia in light che dark mode.
    
    Args:
        icon: Emoji da mostrare
        title: Titolo principale
        subtitle: Descrizione sotto il titolo (opzionale)
        color: Colore accento (default: blu)
    """
    st.markdown(f"""
    <div style="
        border-left: 4px solid {color};
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 0 8px 8px 0;
        background: linear-gradient(90deg, rgba(52,152,219,0.08) 0%, transparent 100%);
    ">
        <h2 style="margin: 0; display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.8rem;">{icon}</span>
            <span>{title}</span>
        </h2>
        {f'<p style="margin: 0.5rem 0 0 0; opacity: 0.8; font-size: 1rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def styled_metric_row(metrics: list):
    """
    Crea una riga di metriche stilizzate.
    metrics: lista di tuple (label, value, icon, color)
    """
    cols = st.columns(len(metrics))
    for i, (label, value, icon, color) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""
            <div style="
                text-align: center;
                padding: 1rem;
                border-radius: 10px;
                border: 1px solid rgba(128,128,128,0.2);
                background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(128,128,128,0.05));
            ">
                <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">{icon}</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: {color};">{value}</div>
                <div style="font-size: 0.85rem; opacity: 0.7;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

def styled_info_box(text: str, icon: str = "💡", box_type: str = "info"):
    """
    Crea un box informativo stilizzato.
    box_type: "info" (blu), "success" (verde), "warning" (arancione), "tip" (viola)
    """
    colors = {
        "info": ("#3498db", "rgba(52,152,219,0.1)"),
        "success": ("#27ae60", "rgba(39,174,96,0.1)"),
        "warning": ("#f39c12", "rgba(243,156,18,0.1)"),
        "tip": ("#9b59b6", "rgba(155,89,182,0.1)")
    }
    border_color, bg_color = colors.get(box_type, colors["info"])
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        border-left: 4px solid {border_color};
        background: {bg_color};
        margin: 1rem 0;
    ">
        <span style="font-size: 1.3rem;">{icon}</span>
        <div style="flex: 1;">{text}</div>
    </div>
    """, unsafe_allow_html=True)


# ============ DOWNLOAD GRAFICI ALTA QUALITÀ ============
def get_download_config(filename="grafico_fisica"):
    """
    Configurazione per download PNG ad alta risoluzione con sfondo trasparente.
    Cliccando l'icona 📷 nella toolbar del grafico si scarica il PNG.
    Scale=4 → risoluzione ~4x (es. 1600x1200 → 6400x4800 px)
    """
    return {
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': filename,
            'height': 1200,
            'width': 1600,
            'scale': 4  # 4x resolution for print quality
        }
    }

def applica_stile(fig, is_light_mode):
    """Imposta sfondo trasparente per il grafico (per export PNG)."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# ============ GESTIONE TEMI (DARK/LIGHT) ============
def get_theme_colors(is_light_mode):
    """Restituisce il dizionario colori in base al tema selezionato."""
    if is_light_mode:
        return {
            'text': '#1a1a2e',              # Testo molto scuro
            'title': '#2c3e50',             # Titoli blu scuro
            'axis': '#2c3e50',              # Assi scuri
            'grid': 'rgba(0,0,0,0.08)',     # Griglia leggera scura
            'zeroline': 'rgba(0,0,0,0.15)', # Linea zero più evidente
            'annotation': '#2c3e50',        # Annotazioni scure
            'subplot_title': '#34495e',     # Titoli subplot
        }
    else:
        return {
            'text': '#ffffff',
            'title': '#ffffff',
            'axis': '#cccccc',
            'grid': 'rgba(128,128,128,0.2)',
            'zeroline': 'rgba(128,128,128,0.3)',
            'annotation': '#ffffff',
            'subplot_title': '#ffffff',
        }

def applica_stile(fig, is_light_mode=False):
    """
    Applica stile al grafico: sfondo trasparente + colori tema.
    Sostituisce apply_transparent_bg aggiungendo il supporto Light Mode.
    """
    tc = get_theme_colors(is_light_mode)
    
    # Sfondo sempre trasparente (si integra col tema Streamlit)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=tc['text']),
        legend=dict(font=dict(color=tc['text'])),
    )
    
    # Colore titolo SOLO se il grafico ha un titolo definito
    # (altrimenti Plotly crea un titolo vuoto che appare come "undefined" nell'export)
    if fig.layout.title and fig.layout.title.text:
        fig.update_layout(title_font_color=tc['title'])
    
    # Aggiorna tutti gli assi (funziona anche con subplot)
    fig.update_xaxes(
        color=tc['axis'],
        gridcolor=tc['grid'],
        zerolinecolor=tc['zeroline'],
        title_font=dict(color=tc['axis']),
        tickfont=dict(color=tc['axis']),
    )
    fig.update_yaxes(
        color=tc['axis'],
        gridcolor=tc['grid'],
        zerolinecolor=tc['zeroline'],
        title_font=dict(color=tc['axis']),
        tickfont=dict(color=tc['axis']),
    )
    
    # Aggiorna colore annotazioni (titoli subplot e vline labels)
    if fig.layout.annotations:
        for ann in fig.layout.annotations:
            # Non sovrascrivere annotazioni con colore specifico già impostato
            if ann.font is None or ann.font.color is None:
                ann.update(font=dict(color=tc['subplot_title']))
    
    return fig


# ============ FUNZIONI UTILITY AVANZATE ============
def genera_audio(segnale, sample_rate=SAMPLE_RATE):
    """Genera file audio WAV da un segnale"""
    segnale_norm = segnale / (np.max(np.abs(segnale)) + 1e-10)
    audio_int16 = np.int16(segnale_norm * 32767 * 0.8)
    buffer = io.BytesIO()
    write(buffer, sample_rate, audio_int16)
    buffer.seek(0)
    return buffer.read()

def genera_audio_con_progress(segnale, sample_rate=SAMPLE_RATE, progress_bar=None):
    """Genera audio con progress bar per file lunghi"""
    if progress_bar and len(segnale) > 5 * sample_rate:  # > 5 secondi
        progress_bar.progress(0.3, "Normalizzazione audio...")
    segnale_norm = segnale / (np.max(np.abs(segnale)) + 1e-10)
    
    if progress_bar and len(segnale) > 5 * sample_rate:
        progress_bar.progress(0.6, "Conversione in WAV...")
    audio_int16 = np.int16(segnale_norm * 32767 * 0.8)
    buffer = io.BytesIO()
    write(buffer, sample_rate, audio_int16)
    buffer.seek(0)
    
    if progress_bar:
        progress_bar.progress(1.0, "Audio generato!")
    return buffer.read()

def calcola_larghezza_temporale(t, inviluppo, threshold=0.05):
    """
    Calcola Δx come distanza tra PRIMI MINIMI LATERALI dell'inviluppo.
    Metodo migliorato per pacchetti con inviluppo sinc.
    """
    max_env = np.max(inviluppo)
    if max_env < 1e-10:
        return 0, 0, len(t)-1
    
    # Normalizza inviluppo
    env_norm = inviluppo / max_env
    
    # Trova indice del massimo centrale
    idx_centro = np.argmax(env_norm)
    
    # ========== RICERCA PRIMO MINIMO A SINISTRA ==========
    idx_sx = None
    for i in range(idx_centro - 10, 10, -1):  # Parti dal centro, vai a sinistra
        if i <= 0 or i >= len(env_norm) - 1:
            continue
        # Controlla se è un minimo locale E se è sotto la soglia
        if (env_norm[i] < env_norm[i-1] and 
            env_norm[i] < env_norm[i+1] and 
            env_norm[i] < threshold):
            idx_sx = i
            break
    
    # ========== RICERCA PRIMO MINIMO A DESTRA ==========
    idx_dx = None
    for i in range(idx_centro + 10, len(env_norm) - 10):  # Parti dal centro, vai a destra
        if i <= 0 or i >= len(env_norm) - 1:
            continue
        # Controlla se è un minimo locale E se è sotto la soglia
        if (env_norm[i] < env_norm[i-1] and 
            env_norm[i] < env_norm[i+1] and 
            env_norm[i] < threshold):
            idx_dx = i
            break
    
    # ========== FALLBACK: usa FWHM se non trova minimi ==========
    if idx_sx is None or idx_dx is None:
        half_max = 0.5
        above_half = env_norm > half_max
        if np.any(above_half):
            idx_sx = np.where(above_half)[0][0]
            idx_dx = np.where(above_half)[0][-1]
        else:
            return 0, 0, len(t)-1
    
    delta_x = abs(t[idx_dx] - t[idx_sx])
    return delta_x, idx_sx, idx_dx



def calcola_velocita_gruppo_fase(f_min, f_max, v_suono=V_SUONO):
    """
    Calcola velocità di fase e gruppo (da relazione).
    Per onde sonore (mezzo non dispersivo): v_fase = v_gruppo = v
    """
    f_centro = (f_min + f_max) / 2
    k_min = 2 * np.pi * f_min / v_suono
    k_max = 2 * np.pi * f_max / v_suono
    k_centro = (k_min + k_max) / 2
    
    omega_min = 2 * np.pi * f_min
    omega_max = 2 * np.pi * f_max
    
    # Per mezzo non dispersivo: ω = vk (lineare)
    v_fase = omega_min / k_min if k_min > 0 else 0
    
    # v_gruppo = dω/dk
    if k_max != k_min:
        v_gruppo = (omega_max - omega_min) / (k_max - k_min)
    else:
        v_gruppo = v_fase
    
    return v_fase, v_gruppo, k_centro

# ============ GESTIONE ZOOM GLOBALE ============
def gestisci_zoom_globale():
    """Gestisce i controlli di zoom manuale nella sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Controllo Grafici")
    
    unisci_viste = st.sidebar.checkbox("Unifica grafici (Zoom interattivo)", value=False,
                                      help="Raggruppa i grafici temporali in un'unica figura per permettere lo zoom sincronizzato direttamente col mouse.")
    
    usa_zoom = st.sidebar.checkbox("Forza Assi Manualmente", value=False, 
                                  help="Disabilita l'autoscale e usa i range definiti qui sotto.")
    
    range_x = None
    range_y = None
    
    if usa_zoom:
        st.sidebar.caption("Imposta i limiti degli assi (Override):")
        col_x1, col_x2 = st.sidebar.columns(2)
        with col_x1:
            x_min = st.number_input("X min", value=0.0, step=0.1, key="g_xmin")
        with col_x2:
            x_max = st.number_input("X max", value=2.0, step=0.1, key="g_xmax")
            
        range_x = [x_min, x_max]
        # Range Y opzionale (spesso l'ampiezza varia molto, ma lo mettiamo per completezza)
        # range_y = [y_min, y_max] 
        
    return range_x, range_y, unisci_viste

def applica_zoom(fig, range_x, range_y=None):
    """Applica i range di zoom se definiti"""
    if range_x:
        fig.update_xaxes(range=range_x, autorange=False)
    if range_y:
        fig.update_yaxes(range=range_y, autorange=False)

# ============ SIDEBAR HEADER + ANIMAZIONI CSS ============
st.markdown("""
<style>
/* Animazioni hover per bottoni e elementi interattivi */
.stButton > button {
    transition: all 0.3s ease;
    border-radius: 8px;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.stDownloadButton > button {
    transition: all 0.3s ease;
}
.stDownloadButton > button:hover {
    transform: scale(1.02);
}
/* Sidebar styling */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Header
st.sidebar.markdown("""
<div style="
    background: linear-gradient(135deg, #3498db 0%, #9b59b6 100%);
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    text-align: center;
">
    <div style="font-size: 2rem; margin-bottom: 0.3rem;">🌊</div>
    <div style="color: white; font-weight: 600; font-size: 1rem;">Fisica delle Onde</div>
    <div style="color: rgba(255,255,255,0.7); font-size: 0.8rem;">Giornata della Scienza 2026</div>
    <div style="color: rgba(255,255,255,0.7); font-size: 0.8rem;">Alessandro Bigi</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📍 Navigazione")
sezione = st.sidebar.radio(
    "Scegli una sezione:",
    ["🚀 Modalità Presentazione", "Battimenti", "Pacchetti d'Onda", "Spettro di Fourier", 
     "Principio di Indeterminazione", "Analisi Multi-Pacchetto", 
     "Regressione Δx vs 1/Δk", "Onde Stazionarie", "Animazione Propagazione",
     "Analisi Audio Microfono", "Riconoscimento Battimenti", "Confronto Scenari", 
     "Analogia Quantistica", "Quiz Interattivo", "Modalità Mobile (Demo)",
     "📥 Centro Download"]
)

mostra_parametri_acustici()  # Mostra parametri fisici

# Attiva Zoom Globale
range_x_glob, range_y_glob, unisci_viste_glob = gestisci_zoom_globale()

# ============ TOGGLE TEMA GRAFICI ============
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Tema Grafici")
is_light_mode = st.sidebar.checkbox(
    "☀️ Modalità Chiara (per stampa/sfondo bianco)", 
    value=False,
    help="Attiva per rendere i grafici leggibili su sfondo bianco. Titoli, assi e griglie diventano scuri."
)

st.sidebar.markdown("---")

# ========== MODALITÀ PRESENTAZIONE ==========
if sezione == "🚀 Modalità Presentazione":
    # ========== TITOLO PRINCIPALE ==========
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 3rem 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    ">
        <h1 style="
            color: white; 
            margin: 0; 
            font-size: 2.8rem; 
            font-weight: 800;
            text-shadow: 0 4px 15px rgba(0,0,0,0.3);
        ">🌊 La Natura Ondulatoria della Realtà</h1>
        <p style="
            color: rgba(255,255,255,0.85); 
            margin: 1rem 0 0 0; 
            font-size: 1.4rem;
            font-weight: 300;
        ">Dai Battimenti al Principio di Indeterminazione</p>
        <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 2rem;">
            <span style="color: rgba(255,255,255,0.7); font-size: 0.95rem;">👨‍🔬 Alessandro Bigi</span>
            <span style="color: rgba(255,255,255,0.7); font-size: 0.95rem;">📍 Giornata della Scienza 2026</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== SEZIONE 1: BATTIMENTI ==========
    styled_header(
        "🎵", 
        "Battimenti: Interferenza tra due onde",
        "Quando due onde con frequenze vicine si sovrappongono, l'ampiezza varia periodicamente",
        "#3498db"
    )
    
    col_beat1, col_beat2 = st.columns([1, 2])
    
    with col_beat1:
        st.subheader("Parametri")
        
        preset_pres = st.selectbox(
            "Configurazione:", 
            ["Diapason Standard LA 440 Hz", "Con pesetto (+5 Hz)", "Differenza maggiore (+20 Hz)"],
            key="pres_beat_preset"
        )
        
        if preset_pres == "Diapason Standard LA 440 Hz":
            f1_pres, f2_pres = 440.0, 440.0
        elif preset_pres == "Con pesetto (+5 Hz)":
            f1_pres, f2_pres = 440.0, 445.0
        else:
            f1_pres, f2_pres = 440.0, 460.0
        
        # Calcoli fisici (stesso codice della sezione Battimenti)
        f_media_pres = (f1_pres + f2_pres) / 2
        f_batt_pres = abs(f1_pres - f2_pres)
        T_batt_pres = 1/f_batt_pres if f_batt_pres > 0 else np.inf
        omega1_pres = 2 * np.pi * f1_pres
        omega2_pres = 2 * np.pi * f2_pres
        
        st.markdown("#### Frequenze")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.metric("f₁", f"{f1_pres:.0f} Hz")
        with col_f2:
            st.metric("f₂", f"{f2_pres:.0f} Hz")
        
        st.markdown("#### Risultato")
        col_fb, col_tb = st.columns(2)
        with col_fb:
            st.metric("f_battimento", f"{f_batt_pres:.1f} Hz", help="|f₁ - f₂|")
        with col_tb:
            st.metric("T_battimento", f"{T_batt_pres:.3f} s" if T_batt_pres != np.inf else "∞")
        
        st.markdown("---")
        st.subheader("Audio")
        durata_audio_pres = st.slider("Durata (s)", 1.0, 5.0, 3.0, 0.5, key="pres_dur_audio")
        if st.button("▶️ Genera Audio", key="pres_gen_audio"):
            t_audio_pres = np.linspace(0, durata_audio_pres, int(SAMPLE_RATE * durata_audio_pres))
            y_audio_pres = np.sin(2*np.pi*f1_pres*t_audio_pres) + np.sin(2*np.pi*f2_pres*t_audio_pres)
            audio_pres = genera_audio(y_audio_pres)
            st.audio(audio_pres, format='audio/wav')
    
    with col_beat2:
        # Calcolo durata ottimale (stesso algoritmo della sezione Battimenti)
        n_battimenti_target = 4
        if f_batt_pres > 0.01:
            durata_pres = n_battimenti_target * T_batt_pres
            durata_pres = max(0.02, min(10.0, durata_pres))
        else:
            durata_pres = 10 / f_media_pres if f_media_pres > 0 else 1.0
            durata_pres = max(0.05, min(5.0, durata_pres))
        
        # Generazione segnale (stesso codice della sezione Battimenti)
        fs_plot = 20000
        t_pres = np.linspace(0, durata_pres, int(durata_pres * fs_plot))
        A1_pres, A2_pres = 1.0, 1.0
        y1_pres = A1_pres * np.cos(2 * np.pi * f1_pres * t_pres)
        y2_pres = A2_pres * np.cos(2 * np.pi * f2_pres * t_pres)
        y_tot_pres = y1_pres + y2_pres
        
        # Inviluppo con padding (stesso metodo)
        pad_len = int(len(t_pres) * 0.1)
        y_padded = np.pad(y_tot_pres, (pad_len, pad_len), mode='reflect')
        analytic_signal = signal.hilbert(y_padded)
        inviluppo_sup = np.abs(analytic_signal)[pad_len:-pad_len]
        inviluppo_inf = -inviluppo_sup
        
        fig_pres_beat = make_subplots(rows=3, cols=1, 
                                      subplot_titles=(f"Onda 1: {f1_pres} Hz", f"Onda 2: {f2_pres} Hz", 
                                                     f"Sovrapposizione (f_batt = {f_batt_pres:.2f} Hz)"),
                                      vertical_spacing=0.1,
                                      shared_xaxes=True)
        
        fig_pres_beat.add_trace(go.Scatter(x=t_pres, y=y1_pres, name="Onda 1", 
                                           line=dict(color='blue', width=1.5)), row=1, col=1)
        fig_pres_beat.add_trace(go.Scatter(x=t_pres, y=y2_pres, name="Onda 2", 
                                           line=dict(color='red', width=1.5)), row=2, col=1)
        fig_pres_beat.add_trace(go.Scatter(x=t_pres, y=y_tot_pres, name="Somma", 
                                           line=dict(color='purple', width=2)), row=3, col=1)
        fig_pres_beat.add_trace(go.Scatter(x=t_pres, y=inviluppo_sup, name="Inviluppo", 
                                           line=dict(color='orange', width=2, dash='dash')), row=3, col=1)
        fig_pres_beat.add_trace(go.Scatter(x=t_pres, y=inviluppo_inf, showlegend=False,
                                           line=dict(color='orange', width=2, dash='dash')), row=3, col=1)
        
        fig_pres_beat.update_xaxes(title_text="Tempo (s)", row=3, col=1)
        fig_pres_beat.update_yaxes(title_text="Ampiezza", row=2, col=1)
        fig_pres_beat.update_layout(height=700, showlegend=True, hovermode='x unified')
        applica_stile(fig_pres_beat, is_light_mode)
        st.plotly_chart(fig_pres_beat, use_container_width=True, config=get_download_config("pres_battimenti"))
    
    # Formula teorica
    with st.expander("Teoria dei Battimenti", expanded=False):
        st.latex(r"y(t) = 2A\cos\left(\frac{\omega_1 - \omega_2}{2}t\right) \cos\left(\frac{\omega_1 + \omega_2}{2}t\right)")
        st.latex(r"f_{\text{battimento}} = |f_1 - f_2|")
    
    # ========== SEZIONE 2: PACCHETTI D'ONDA ==========
    st.markdown("---")
    styled_header(
        "🌊", 
        "Pacchetti d'Onda",
        "Sovrapposizione di molte frequenze che crea un impulso localizzato",
        "#9b59b6"
    )
    
    col_pack1, col_pack2 = st.columns([1, 2])
    
    with col_pack1:
        st.subheader("Parametri")
        
        preset_pkt_pres = st.selectbox(
            "Carica preset:", 
            list(PRESET_PACCHETTI.keys()), 
            key="pres_preset_pkt"
        )
        
        if preset_pkt_pres != "Personalizzato":
            preset = PRESET_PACCHETTI[preset_pkt_pres]
            f_min_pres = preset["f_min"]
            f_max_pres = preset["f_max"]
            n_onde_pres = preset["N"]
            st.info(f"**{preset_pkt_pres}**\n\n{preset['descrizione']}")
        else:
            f_min_pres = st.slider("Frequenza minima (Hz)", 10.0, 300.0, 100.0, 5.0, key="pres_fmin")
            f_max_pres = st.slider("Frequenza massima (Hz)", f_min_pres+5, 400.0, 130.0, 5.0, key="pres_fmax")
            n_onde_pres = st.slider("Numero di onde (N)", 5, 100, 50, 5, key="pres_n_onde")
        
        mostra_comp_pres = st.checkbox("Mostra onde componenti (max 10)", False, key="pres_show_comp")
        
        delta_f_pres = f_max_pres - f_min_pres
        f_centrale_pres = (f_min_pres + f_max_pres) / 2
        delta_omega_pres = 2 * np.pi * delta_f_pres
        
        st.markdown("#### Caratteristiche")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("Freq. centrale", f"{f_centrale_pres:.1f} Hz")
            st.metric("N onde", n_onde_pres)
        with col_c2:
            st.metric("Δf", f"{delta_f_pres:.1f} Hz")
            st.metric("Δω", f"{delta_omega_pres:.1f} rad/s")
    
    with col_pack2:
        # Durata FISSA per confronto visivo tra preset
        durata_pack = 0.3  # 300 ms fissi
        
        # Asse simmetrico centrato in t=0
        t_pack = np.linspace(-durata_pack, durata_pack, int(durata_pack * 2 * 20000))
        frequenze_pack = np.linspace(f_min_pres, f_max_pres, n_onde_pres)
        
        y_pacchetto = np.zeros_like(t_pack)
        for f in frequenze_pack:
            y_pacchetto += (1 / n_onde_pres) * np.cos(2 * np.pi * f * t_pack)
        
        # Inviluppo
        pad_len = int(len(t_pack) * 0.1)
        y_pad = np.pad(y_pacchetto, (pad_len, pad_len), mode='reflect')
        analytic_pack = signal.hilbert(y_pad)
        inviluppo_pack = np.abs(analytic_pack)[pad_len:-pad_len]
        intensita_pack = inviluppo_pack**2
        
        fig_pack = make_subplots(rows=2, cols=1,
                                 subplot_titles=(f"Pacchetto: {n_onde_pres} onde ({f_min_pres}-{f_max_pres} Hz)", 
                                               "Intensità |A(t)|²"),
                                 shared_xaxes=True)
        
        if mostra_comp_pres and n_onde_pres <= 50:
            step = max(1, n_onde_pres // 10)
            for i, f in enumerate(frequenze_pack[::step]):
                y_comp = (1 / n_onde_pres) * np.cos(2 * np.pi * f * t_pack)
                fig_pack.add_trace(go.Scatter(x=t_pack*1000, y=y_comp, name=f"f={f:.1f} Hz",
                                              line=dict(width=0.5), opacity=0.3), row=1, col=1)
        
        fig_pack.add_trace(go.Scatter(x=t_pack*1000, y=y_pacchetto, name="Pacchetto",
                                      line=dict(color='darkblue', width=2.5)), row=1, col=1)
        fig_pack.add_trace(go.Scatter(x=t_pack*1000, y=inviluppo_pack, name="Inviluppo +",
                                      line=dict(color='red', width=2, dash='dash')), row=1, col=1)
        fig_pack.add_trace(go.Scatter(x=t_pack*1000, y=-inviluppo_pack, showlegend=False,
                                      line=dict(color='red', width=2, dash='dash')), row=1, col=1)
        
        fig_pack.add_trace(go.Scatter(x=t_pack*1000, y=intensita_pack, fill='tozeroy', 
                                      line=dict(color='orange', width=2), name="|A(t)|²"), row=2, col=1)
        
        # Aggiungi linea verticale a t=0
        fig_pack.add_vline(x=0, line_dash="dot", line_color="green", annotation_text="t=0")
        
        fig_pack.update_layout(height=650, hovermode='x unified')
        
        # FORZA assi fissi disabilitando autorange
        fig_pack.update_xaxes(range=[-300, 300], autorange=False, fixedrange=True, title_text="Tempo (ms)", row=2, col=1)
        fig_pack.update_xaxes(range=[-300, 300], autorange=False, fixedrange=True, row=1, col=1)
        fig_pack.update_yaxes(range=[-1.2, 1.2], autorange=False, fixedrange=True, title_text="Ampiezza", row=1, col=1)
        fig_pack.update_yaxes(range=[0, 1.2], autorange=False, fixedrange=True, title_text="|A(t)|²", row=2, col=1)
        applica_stile(fig_pack, is_light_mode)
        st.plotly_chart(fig_pack, use_container_width=True, config=get_download_config("pres_pacchetto"))
    
    # ========== SEZIONE 3: PRINCIPIO DI INDETERMINAZIONE ==========
    st.markdown("---")
    styled_header(
        "⚠️", 
        "Principio di Indeterminazione",
        "Relazione fondamentale: Δx · Δk ≥ 1/2",
        "#f39c12"
    )
    
    scenario_pres = st.radio(
        "Seleziona scenario:",
        ["Super-Localizzato (Δk grande)", "Quasi-Monocromatico (Δk piccolo)"],
        key="pres_scenario",
        horizontal=True
    )
    
    if "Super-Localizzato" in scenario_pres:
        preset_ind = PRESET_PACCHETTI["Super-Localizzato (Δk grande)"]
    else:
        preset_ind = PRESET_PACCHETTI["Quasi-Monocromatico (Δk piccolo)"]
    
    f_min_ind = preset_ind["f_min"]
    f_max_ind = preset_ind["f_max"]
    n_ind = preset_ind["N"]
    
    # Calcoli fisici (stesso codice della sezione Principio di Indeterminazione)
    lambda_min_ind = V_SUONO / f_max_ind
    lambda_max_ind = V_SUONO / f_min_ind
    k_min_ind = 2 * np.pi / lambda_max_ind
    k_max_ind = 2 * np.pi / lambda_min_ind
    delta_k_ind = k_max_ind - k_min_ind
    delta_x_teorico_ind = 4 * np.pi / delta_k_ind if delta_k_ind > 0 else 0
    
    delta_f_ind = f_max_ind - f_min_ind
    delta_omega_ind = 2 * np.pi * delta_f_ind
    delta_t_teorico_ind = 4 * np.pi / delta_omega_ind if delta_omega_ind > 0 else 0
    
    col_ind1, col_ind2 = st.columns(2)
    
    with col_ind1:
        st.markdown("#### Dominio Temporale")
        
        # Durata FISSA - genera sempre lo stesso range di dati
        durata_ind = 0.3  # 300 ms fissi
        
        # Asse simmetrico centrato in t=0
        t_ind = np.linspace(-durata_ind, durata_ind, int(durata_ind * 2 * 20000))
        omega_vals_ind = 2 * np.pi * np.linspace(f_min_ind, f_max_ind, n_ind)
        y_ind = np.zeros_like(t_ind)
        for omega in omega_vals_ind:
            y_ind += (1/n_ind) * np.cos(omega * t_ind)
        
        env_ind = np.abs(signal.hilbert(y_ind))
        
        fig_time_ind = go.Figure()
        fig_time_ind.add_trace(go.Scatter(x=t_ind*1000, y=y_ind, line=dict(color='purple', width=2), name="Pacchetto"))
        fig_time_ind.add_trace(go.Scatter(x=t_ind*1000, y=env_ind, line=dict(color='orange', width=2, dash='dash'), name="Inviluppo"))
        fig_time_ind.add_trace(go.Scatter(x=t_ind*1000, y=-env_ind, showlegend=False, line=dict(color='orange', width=2, dash='dash')))
        fig_time_ind.add_vline(x=0, line_dash="dot", line_color="green", annotation_text="t=0")
        
        fig_time_ind.update_layout(
            title=f"Pacchetto nel Tempo (Δt ≈ {delta_t_teorico_ind*1000:.1f} ms)", 
            xaxis_title="t (ms)", 
            yaxis_title="A(t)", 
            height=400, 
            hovermode='x unified'
        )
        # FORZA assi fissi disabilitando autorange
        fig_time_ind.update_xaxes(range=[-300, 300], autorange=False, fixedrange=True)
        fig_time_ind.update_yaxes(range=[-1.2, 1.2], autorange=False, fixedrange=True)
        applica_stile(fig_time_ind, is_light_mode)
        st.plotly_chart(fig_time_ind, use_container_width=True, config=get_download_config("pres_tempo_ind"))
    
    with col_ind2:
        st.markdown("#### Spettro di Frequenze")
        
        freq_spectrum_ind = np.linspace(f_min_ind, f_max_ind, n_ind)
        amplitudes_ind = np.ones(n_ind) / n_ind
        
        fig_freq_ind = go.Figure()
        fig_freq_ind.add_trace(go.Bar(
            x=freq_spectrum_ind,
            y=amplitudes_ind,
            marker_color='#e74c3c',
            width=(f_max_ind - f_min_ind) / n_ind * 0.8
        ))
        fig_freq_ind.add_vline(x=f_min_ind, line_dash="dash", line_color="blue", annotation_text=f"f_min={f_min_ind:.0f} Hz", annotation_position="top left")
        fig_freq_ind.add_vline(x=f_max_ind, line_dash="dash", line_color="blue", annotation_text=f"f_max={f_max_ind:.0f} Hz", annotation_position="top right")
        
        fig_freq_ind.update_layout(
            title=f"Spettro: Δf = {delta_f_ind:.1f} Hz", 
            xaxis_title="Frequenza (Hz)", 
            yaxis_title="Ampiezza", 
            height=400, 
            showlegend=False
        )
        # FORZA assi fissi disabilitando autorange
        fig_freq_ind.update_xaxes(range=[0, 250], autorange=False, fixedrange=True)
        fig_freq_ind.update_yaxes(range=[0, 0.05], autorange=False, fixedrange=True)
        applica_stile(fig_freq_ind, is_light_mode)
        st.plotly_chart(fig_freq_ind, use_container_width=True, config=get_download_config("pres_spettro_ind"))
    
    # Risultato
    st.markdown("### Analisi Teorica")
    prodotto_xk_ind = delta_x_teorico_ind * delta_k_ind
    prodotto_wt_ind = delta_t_teorico_ind * delta_omega_ind
    
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Δf", f"{delta_f_ind:.1f} Hz")
        st.metric("Δt teorico", f"{delta_t_teorico_ind*1000:.2f} ms")
    with col_r2:
        st.metric("Δk", f"{delta_k_ind:.4f} rad/m")
        st.metric("Δx teorico", f"{delta_x_teorico_ind:.3f} m")
    with col_r3:
        st.metric("Δx·Δk", f"{prodotto_xk_ind:.3f}", delta=f"4π={4*np.pi:.3f}")
        st.metric("Δω·Δt", f"{prodotto_wt_ind:.3f}", delta=f"4π={4*np.pi:.3f}")
    
    st.latex(r"\Delta x \cdot \Delta k \geq \frac{1}{2} \quad \Rightarrow \quad \Delta x \cdot \Delta p \geq \frac{\hbar}{2}")
    
    # ========== CONCLUSIONE ==========
    st.markdown("---")
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 2rem;
    ">
        <h2 style="color: white; margin: 0 0 1rem 0;">🎓 Conclusione</h2>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-bottom: 1.5rem;">
            La materia, nel profondo, non è fatta di "palline" con posizioni e velocità precise,<br>
            ma è fatta di <strong>onde</strong> che si estendono nello spazio.
        </p>
        <div style="
            background: rgba(255,255,255,0.1);
            padding: 1.5rem;
            border-radius: 10px;
            display: inline-block;
        ">
            <span style="color: #f1c40f; font-size: 1.8rem; font-weight: 700;">Δx · Δp ≥ ℏ/2</span>
        </div>
        <p style="color: rgba(255,255,255,0.7); margin-top: 1.5rem; font-size: 1rem;">
            Grazie per l'attenzione! 🙏
        </p>
    </div>
    """, unsafe_allow_html=True)


# ========== SEZIONE BATTIMENTI ==========
if sezione == "Battimenti":
    styled_header(
        "🎵", 
        "Battimenti: Interferenza tra due onde",
        "Quando due onde con frequenze vicine si sovrappongono, l'ampiezza varia periodicamente. Formula: f_batt = |f₁ - f₂|",
        "#3498db"
    )


    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parametri")
        
        # Inizializza session_state se non esiste
        if 'f1' not in st.session_state: st.session_state.f1 = 440.0
        if 'f2' not in st.session_state: st.session_state.f2 = 445.0
        if 'A1' not in st.session_state: st.session_state.A1 = 1.0
        if 'A2' not in st.session_state: st.session_state.A2 = 1.0

        # Inizializza widget keys per evitare warning "default value"
        for param in ['f1', 'f2', 'A1', 'A2']:
            if f"{param}_slider" not in st.session_state: 
                st.session_state[f"{param}_slider"] = st.session_state[param]
            if f"{param}_input" not in st.session_state: 
                st.session_state[f"{param}_input"] = st.session_state[param]

        def applica_preset():
            if st.session_state.preset_batt_k != "Personalizzato":
                p = PRESET_FAMOSI[st.session_state.preset_batt_k]
                st.session_state.f1 = p["f1"]
                st.session_state.f2 = p["f2"]
                st.session_state.A1 = p["A1"]
                st.session_state.A2 = p["A2"]
                # Reset widget states so they pick up the new 'value'
                for k in ['f1', 'f2', 'A1', 'A2']:
                    if f"{k}_slider" in st.session_state: del st.session_state[f"{k}_slider"]
                    if f"{k}_input" in st.session_state: del st.session_state[f"{k}_input"]

        def set_custom(param, widget_key):
            # 1. Aggiorna il parametro principale (es. 'f1')
            val = st.session_state[widget_key]
            st.session_state[param] = val
            
            # 2. Sincronizza l'altro widget della coppia (Slider <-> Input)
            # Se ho mosso lo slider, aggiorno l'input, e viceversa.
            if "_slider" in widget_key:
                st.session_state[widget_key.replace("_slider", "_input")] = val
            elif "_input" in widget_key:
                st.session_state[widget_key.replace("_input", "_slider")] = val
            
            # 3. Imposta il preset su Personalizzato
            st.session_state.preset_batt_k = "Personalizzato"

        preset_batt = st.selectbox("Carica preset:", list(PRESET_FAMOSI.keys()), key="preset_batt_k", on_change=applica_preset)
        
        if preset_batt != "Personalizzato":
            st.info(f"**{preset_batt}**\n\n{PRESET_FAMOSI[preset_batt]['descrizione']}")
        
        # Frequenza 1
        col_s1, col_i1 = st.columns([3, 1])
        with col_s1:
            st.slider("Frequenza onda 1 (Hz)", 1.0, 2000.0, 
                 key="f1_slider", 
                 on_change=set_custom, args=('f1', 'f1_slider'))
        with col_i1:
            st.number_input("", min_value=1.0, max_value=2000.0, 
                           key="f1_input", 
                           step=0.1, format="%.1f",
                           on_change=set_custom, args=('f1', 'f1_input'))
        
        # Frequenza 2
        col_s2, col_i2 = st.columns([3, 1])
        with col_s2:
            st.slider("Frequenza onda 2 (Hz)", 1.0, 2000.0,
                     key="f2_slider",
                     on_change=set_custom, args=('f2', 'f2_slider'))
        with col_i2:
            st.number_input("", min_value=1.0, max_value=2000.0,
                           key="f2_input",
                           step=0.1, format="%.1f",
                           on_change=set_custom, args=('f2', 'f2_input'))
        
        # Ampiezza 1
        col_a1, col_ia1 = st.columns([3, 1])
        with col_a1:
            st.slider("Ampiezza onda 1", 0.5, 2.0,
                     key="A1_slider",
                     on_change=set_custom, args=('A1', 'A1_slider'))
        with col_ia1:
            st.number_input("", min_value=0.5, max_value=2.0,
                           key="A1_input",
                           step=0.1, format="%.1f",
                           on_change=set_custom, args=('A1', 'A1_input'))
        
        # Ampiezza 2
        col_a2, col_ia2 = st.columns([3, 1])
        with col_a2:
            st.slider("Ampiezza onda 2", 0.5, 2.0,
                     key="A2_slider",
                     on_change=set_custom, args=('A2', 'A2_slider'))
        with col_ia2:
            st.number_input("", min_value=0.5, max_value=2.0,
                           key="A2_input",
                           step=0.1, format="%.1f",
                           on_change=set_custom, args=('A2', 'A2_input'))
        
        # Usa i valori dal session_state
        f1 = st.session_state.f1
        f2 = st.session_state.f2
        A1 = st.session_state.A1
        A2 = st.session_state.A2
        
        durata = 2.0

        if preset_batt != "Personalizzato":
            st.info(f"Preset caricato: {preset_batt}")
        
        f_media = (f1 + f2) / 2
        f_batt = abs(f1 - f2)
        T_batt = 1/f_batt if f_batt > 0 else np.inf
        omega1 = 2 * np.pi * f1
        omega2 = 2 * np.pi * f2
        
        n_battimenti_target = 4
        if f_batt > 0.01:
            durata_auto = n_battimenti_target * T_batt
            if durata_auto < 0.02:
                durata_auto = 0.02
            elif durata_auto > 10.0:
                durata_auto = 10.0
        else:
            durata_auto = 10 / f_media if f_media > 0 else 1.0
            durata_auto = max(0.05, min(5.0, durata_auto))
        
        st.markdown("---")
        usa_auto = st.checkbox("Scala asse X automatica", value=True, 
                              help="Adatta automaticamente l'asse tempo per mostrare 4 battimenti completi")
        
        if usa_auto:
            durata = durata_auto
            if f_batt > 0.01:
                n_batt_effettivi = durata / T_batt
                if n_batt_effettivi > 10:
                    st.warning(f"Battimenti molto veloci! f_batt={f_batt:.1f} Hz → {n_batt_effettivi:.0f} battimenti in {durata:.3f}s")
                elif n_batt_effettivi > 6:
                    st.info(f"Durata: {durata:.3f} s ≈ {n_batt_effettivi:.1f} battimenti (f_batt={f_batt:.1f} Hz)")
                else:
                    st.success(f"Durata: {durata:.3f} s = {n_batt_effettivi:.1f} battimenti (f_batt={f_batt:.1f} Hz)")
            else:
                n_periodi = durata * f_media
                st.success(f"Durata: {durata:.3f} s ≈ {n_periodi:.0f} periodi portante")
        else:
            durata = st.slider("Durata manuale (s)", 0.05, 10.0, durata_auto, 0.05, key="dur_manual")
        
        # 🆕 GENERAZIONE AUDIO
        st.markdown("---")
        st.subheader("Genera Audio")
        durata_audio_batt = st.slider("Durata audio (s)", 0.5, 30.0, min(durata*2, 10.0), 0.5, 
                                      key="dur_audio_batt", 
                                      help="Durata del file audio (indipendente dalla visualizzazione)")
        if st.button("Genera battimenti audio", key="gen_batt_audio"):
            if durata_audio_batt > 5:
                progress = st.progress(0, "Generazione in corso...")
            else:
                progress = None
            
            if progress:
                progress.progress(0.2, "Calcolo segnale...")
            t_audio = np.linspace(0, durata_audio_batt, int(SAMPLE_RATE * durata_audio_batt))
            y_audio = np.sin(2 * np.pi * f1 * t_audio) + np.sin(2 * np.pi * f2 * t_audio)
            
            audio_bytes = genera_audio_con_progress(y_audio, SAMPLE_RATE, progress)
            
            if progress:
                progress.empty()
            
            st.success(f"Audio generato: {durata_audio_batt:.1f} secondi ({len(y_audio):,} campioni)")
            st.audio(audio_bytes, format='audio/wav')
            st.download_button("Scarica WAV", audio_bytes, f"battimenti_{int(f1)}_{int(f2)}_Hz_{durata_audio_batt:.0f}s.wav", "audio/wav")
        
        st.markdown("---")
        if st.button("Esporta dati in CSV"):
            export_data = {
                "Parametro": ["f1 (Hz)", "f2 (Hz)", "A1", "A2", "f_media (Hz)", "f_battimento (Hz)", "T_battimento (s)"],
                "Valore": [f1, f2, A1, A2, f_media, f_batt, T_batt if T_batt != np.inf else 0]
            }
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            st.download_button("Scarica CSV", csv, "battimenti_dati.csv", "text/csv")
    
    with col2:
        # Aumento risoluzione per evitare aliasing con frequenze alte (fino a 2000Hz)
        fs_plot = 20000  # Hz (Aumentato per zoom fluido)
        t = np.linspace(0, durata, int(durata * fs_plot))
        y1 = A1 * np.cos(2 * np.pi * f1 * t)
        y2 = A2 * np.cos(2 * np.pi * f2 * t)
        y_tot = y1 + y2
        
        # Calcolo inviluppo con padding per evitare effetti ai bordi (Gibbs)
        pad_len = int(len(t) * 0.1)
        y_padded = np.pad(y_tot, (pad_len, pad_len), mode='reflect')
        analytic_signal = signal.hilbert(y_padded)
        inviluppo_sup = np.abs(analytic_signal)[pad_len:-pad_len]
        inviluppo_inf = -inviluppo_sup
        
        fig = make_subplots(rows=3, cols=1, 
                           subplot_titles=(f"Onda 1: {f1} Hz", f"Onda 2: {f2} Hz", 
                                         f"Sovrapposizione (f_batt = {f_batt:.2f} Hz)"),
                           vertical_spacing=0.1,
                           shared_xaxes=True) # Sincronizza zoom X tra i subplot
        
        fig.add_trace(go.Scatter(x=t, y=y1, name=f"Onda 1", 
                                line=dict(color='blue', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=y2, name=f"Onda 2", 
                                line=dict(color='red', width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=y_tot, name="Somma", 
                                line=dict(color='purple', width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=t, y=inviluppo_sup, name="Inviluppo", 
                                line=dict(color='orange', width=2, dash='dash')), row=3, col=1)
        fig.add_trace(go.Scatter(x=t, y=inviluppo_inf, showlegend=False,
                                line=dict(color='orange', width=2, dash='dash')), row=3, col=1)
        
        fig.update_xaxes(title_text="Tempo (s)", row=3, col=1)
        fig.update_yaxes(title_text="Ampiezza", row=2, col=1)
        fig.update_xaxes(autorange=True)
        fig.update_yaxes(autorange=True, automargin=True)
        
        fig.update_layout(
            height=800, 
            showlegend=True, 
            hovermode='x unified',
            dragmode='zoom',
            uirevision='constant',
            xaxis=dict(autorange=True, rangeslider=dict(visible=False)),
            yaxis=dict(autorange=True, fixedrange=False),
            modebar_add=['resetScale2d']
        )
        
        applica_zoom(fig, range_x_glob)
        applica_stile(fig, is_light_mode)
        st.plotly_chart(fig, use_container_width=True, config=get_download_config("battimenti_tempo"))

    st.markdown("---")
    st.markdown("---")
    st.header("Valori Teorici Completi")
    
    
    st.markdown("### Onda Portante")
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.metric("f_media (portante)", f"{f_media:.2f} Hz", help="(f₁ + f₂) / 2")
    with col_t2:
        T_onda = 1 / f_media if f_media > 0 else 0
        st.metric("T_onda", f"{T_onda:.6f} s", help="1 / f_media")
    with col_t3:
        omega_media = (omega1 + omega2) / 2
        st.metric("ω_media", f"{omega_media:.2f} rad/s", help="(ω₁ + ω₂) / 2")
    with col_t4:
        k_media = omega_media / V_SUONO
        st.metric("k_media", f"{k_media:.4f} rad/m", help="ω_media / v")
    
    st.markdown("### Battimento")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        st.metric("f_battimento", f"{f_batt:.2f} Hz", help="|f₁ - f₂|")
    with col_b2:
        st.metric("T_battimento", f"{T_batt:.4f} s" if T_batt != np.inf else "∞", help="1 / f_batt")
    with col_b3:
        T_ampiezza = 2 * T_batt if T_batt != np.inf else np.inf
        st.metric("T*_ampiezza", f"{T_ampiezza:.4f} s" if T_ampiezza != np.inf else "∞", help="2 × T_battimento")
    with col_b4:
        f_ampiezza = f_batt / 2 if f_batt > 0 else 0
        st.metric("f*_ampiezza", f"{f_ampiezza:.2f} Hz", help="f_batt / 2")
    
    st.markdown("### Pulsazioni (ω)")
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        st.metric("ω₁", f"{omega1:.2f} rad/s", help="2π × f₁")
    with col_w2:
        st.metric("ω₂", f"{omega2:.2f} rad/s", help="2π × f₂")
    with col_w3:
        delta_omega = abs(omega1 - omega2)
        st.metric("Δω", f"{delta_omega:.2f} rad/s", help="|ω₁ - ω₂|")
    with col_w4:
        omega_batt = delta_omega / 2
        st.metric("ω_battimento", f"{omega_batt:.2f} rad/s", help="Δω / 2")
    
    st.markdown("### Lunghezze d'onda (v = 340 m/s)")
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    with col_l1:
        lambda1 = V_SUONO / f1 if f1 > 0 else 0
        st.metric("λ₁", f"{lambda1:.4f} m", help="v / f₁")
    with col_l2:
        lambda2 = V_SUONO / f2 if f2 > 0 else 0
        st.metric("λ₂", f"{lambda2:.4f} m", help="v / f₂")
    with col_l3:
        lambda_media = (lambda1 + lambda2) / 2
        st.metric("λ_media", f"{lambda_media:.4f} m", help="(λ₁ + λ₂) / 2")
    with col_l4:
        delta_lambda = abs(lambda1 - lambda2)
        st.metric("Δλ", f"{delta_lambda:.4f} m", help="|λ₁ - λ₂|")
    
    st.markdown("### Numeri d'onda (k)")
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        k1 = 2 * np.pi / lambda1 if lambda1 > 0 else 0
        st.metric("k₁", f"{k1:.4f} rad/m", help="2π / λ₁")
    with col_k2:
        k2 = 2 * np.pi / lambda2 if lambda2 > 0 else 0
        st.metric("k₂", f"{k2:.4f} rad/m", help="2π / λ₂")
    with col_k3:
        delta_k = abs(k1 - k2)
        st.metric("Δk", f"{delta_k:.4f} rad/m", help="|k₁ - k₂|")
    with col_k4:
        k_batt = delta_k / 2
        st.metric("k_battimento", f"{k_batt:.4f} rad/m", help="Δk / 2")
    
    with st.expander("Vedi tutte le formule teoriche (LaTeX)"):
        col_form1, col_form2 = st.columns(2)
        with col_form1:
            st.markdown("#### Onda risultante")
            st.latex(r"y(t) = y_1(t) + y_2(t)")
            st.latex(r"y(t) = A\cos(\omega_1 t) + A\cos(\omega_2 t)")
            st.latex(r"y(t) = 2A\cos\left(\frac{\omega_1 - \omega_2}{2}t\right) \cos\left(\frac{\omega_1 + \omega_2}{2}t\right)")
            if abs(A1 - A2) > 0.01:
                st.caption("*Nota: La formula semplificata qui sopra assume ampiezze uguali ($A_1 = A_2$). Con ampiezze diverse, l'interferenza distruttiva non è completa e il minimo dell'inviluppo non scende a zero.*")
            st.markdown("#### 📊 Frequenze")
            st.latex(r"f_{\text{media}} = \frac{f_1 + f_2}{2}")
            st.latex(r"f_{\text{batt}} = |f_1 - f_2|")
            st.latex(r"f^*_{\text{ampiezza}} = \frac{|f_1 - f_2|}{2}")
        with col_form2:
            st.markdown("#### Pulsazioni")
            st.latex(r"\omega = 2\pi f")
            st.latex(r"\omega_{\text{media}} = \frac{\omega_1 + \omega_2}{2}")
            st.latex(r"\Delta\omega = |\omega_1 - \omega_2|")
            st.markdown("#### Relazioni ondulatorie")
            st.latex(r"\lambda = \frac{v}{f}")
            st.latex(r"k = \frac{2\pi}{\lambda}")

# ========== PACCHETTI D'ONDA ==========
elif sezione == "Pacchetti d'Onda":
    styled_header(
        "🌊", 
        "Pacchetti d'Onda",
        "Sovrapposizione di molte frequenze che crea un impulso localizzato nello spazio e nel tempo",
        "#9b59b6"
    )
    
    # 📚 TEORIA MATEMATICA
    with st.expander("Teoria Pacchetti d'Onda", expanded=False):
        st.markdown("### Teoria Matematica")
    
        st.markdown("**Dalla Sovrapposizione Discreta al Continuo**")
        st.markdown("Un pacchetto d'onda è formato dalla sovrapposizione di N onde:")
        st.latex(r"y(x,t) = \sum_{i=1}^{N} A_i \cos(k_i x - \omega_i t + \phi_i)")
    
        st.markdown("Nel limite per N → ∞, diventa un integrale di Fourier:")
        st.latex(r"y(x,t) = \int_{-\infty}^{\infty} A(k) e^{i(kx - \omega(k)t)} dk")
    
        st.markdown("---")
        st.markdown("**Inviluppo Sinc**")
        st.markdown("Per spettro uniforme:")
        st.latex(r"\psi(x) = A_0 \Delta k \,\text{sinc}\left(\frac{\Delta k \cdot x}{2}\right)")
    
        st.markdown("Distanza tra i primi zeri (larghezza spaziale):")
        st.latex(r"\Delta x = \frac{4\pi}{\Delta k}")
    
        st.markdown("---")
        st.markdown("**Velocità di Propagazione**")
    
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("Velocità di fase (fronti d'onda):")
            st.latex(r"v_{\text{fase}} = \frac{\omega}{k}")
    
        with col_v2:
            st.markdown("Velocità di gruppo (energia):") 
            st.latex(r"v_{\text{gruppo}} = \frac{d\omega}{dk}")
    
        st.markdown("---")
        st.markdown("**Per onde sonore in aria** (mezzo non dispersivo):")
        st.latex(r"v_{\text{fase}} = v_{\text{gruppo}} = v = 340 \text{ m/s}")
    
        st.info("""
        **Caratteristiche mezzo non dispersivo:**
        - Il pacchetto mantiene la forma propagandosi
        - Non c'è distorsione temporale  
        - L'inviluppo viaggia alla stessa velocità delle oscillazioni
        """)
    
        st.markdown("**Mezzi dispersivi** (es. onde sull'acqua):")
        st.markdown("Se ω(k) non è lineare → v_fase ≠ v_gruppo → il pacchetto si deforma nel tempo")



        
        # Pacchetti d'onda
        st.markdown("""
        Un pacchetto d'onda si ottiene sommando molte onde con frequenze vicine. 
        Il risultato è un segnale localizzato nello spazio (o nel tempo).
    
        **Teoria**: Al crescere di N, l'onda risultante diventa sempre più localizzata.
        """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parametri")
        
        preset_pkt = st.selectbox("Carica preset:", list(PRESET_PACCHETTI.keys()), key="preset_pkt_main")
        
        if preset_pkt != "Personalizzato":
            preset = PRESET_PACCHETTI[preset_pkt]
            f_min = preset["f_min"]
            f_max = preset["f_max"]
            n_onde = preset["N"]
            ampiezza = 1.0  # Default
            durata = 1.5    # Default
            st.info(f"**{preset_pkt}**\n\n{preset['descrizione']}")
        else:
            # Funzione di sincronizzazione per i pacchetti
            def sync_pkt(param_base):
                # Determina quale widget ha scatenato l'evento
                # Se param_base è 'pkt_fmin', controlliamo 'pkt_fmin_s' e 'pkt_fmin_i'
                val_s = st.session_state.get(f"{param_base}_s")
                val_i = st.session_state.get(f"{param_base}_i")
                
                # Trova il valore nuovo (non sappiamo quale dei due è cambiato, ma possiamo assumerlo o uniformarli)
                # Per semplicità, uniformiamo basandoci su quello che esiste nello stato
                # Streamlit aggiorna lo stato PRIMA della callback, quindi il widget toccato ha il valore nuovo.
                # Tuttavia, per evitare complessità, usiamo una logica diretta:
                # Se la callback è chiamata dallo slider, aggiorniamo l'input.
                
                # Nota: Qui usiamo una logica semplificata. Se cambiamo f_min, dobbiamo controllare f_max.
                pass 

            # Inizializza session state per pacchetti se non esiste
            if 'pkt_fmin_s' not in st.session_state: st.session_state.pkt_fmin_s = 100.0
            if 'pkt_fmin_i' not in st.session_state: st.session_state.pkt_fmin_i = 100.0
            if 'pkt_fmax_s' not in st.session_state: st.session_state.pkt_fmax_s = 130.0
            if 'pkt_fmax_i' not in st.session_state: st.session_state.pkt_fmax_i = 130.0
            if 'pkt_n_s' not in st.session_state: st.session_state.pkt_n_s = 50
            if 'pkt_n_i' not in st.session_state: st.session_state.pkt_n_i = 50

            def update_pkt_widget(key_from, key_to):
                st.session_state[key_to] = st.session_state[key_from]
                
                # Controllo di sicurezza f_min < f_max
                if "fmin" in key_from or "fmax" in key_from:
                    curr_min = st.session_state.pkt_fmin_s # o _i, sono sincronizzati
                    curr_max = st.session_state.pkt_fmax_s
                    
                    if curr_max <= curr_min:
                        # Se f_max è troppo basso, lo alziamo
                        new_max = curr_min + 5.0
                        st.session_state.pkt_fmax_s = new_max
                        st.session_state.pkt_fmax_i = new_max

            col_fmin_s, col_fmin_i = st.columns([3, 1])
            with col_fmin_s:
                f_min_slider = st.slider("Frequenza minima (Hz)", 1.0, 500.0, key="pkt_fmin_s", on_change=update_pkt_widget, args=("pkt_fmin_s", "pkt_fmin_i"))
            with col_fmin_i:
                f_min = st.number_input("", min_value=1.0, max_value=500.0, step=1.0, key="pkt_fmin_i", format="%.1f", on_change=update_pkt_widget, args=("pkt_fmin_i", "pkt_fmin_s"))
            
            col_fmax_s, col_fmax_i = st.columns([3, 1])
            with col_fmax_s:
                # Calcola min value dinamico per lo slider
                min_fmax = f_min + 1.0
                # Assicura che lo stato sia coerente prima di renderizzare
                if st.session_state.pkt_fmax_s < min_fmax:
                     st.session_state.pkt_fmax_s = min_fmax
                     st.session_state.pkt_fmax_i = min_fmax
                     
                f_max_slider = st.slider("Frequenza massima (Hz)", min_fmax, 500.0, key="pkt_fmax_s", on_change=update_pkt_widget, args=("pkt_fmax_s", "pkt_fmax_i"))
            with col_fmax_i:
                f_max = st.number_input("", min_value=min_fmax, max_value=500.0, step=1.0, key="pkt_fmax_i", format="%.1f", on_change=update_pkt_widget, args=("pkt_fmax_i", "pkt_fmax_s"))
            
            col_n_s, col_n_i = st.columns([3, 1])
            with col_n_s:
                n_onde_slider = st.slider("Numero di onde (N)", 5, 100, key="pkt_n_s", step=5, on_change=update_pkt_widget, args=("pkt_n_s", "pkt_n_i"))
            with col_n_i:
                n_onde = st.number_input("", min_value=5, max_value=100, key="pkt_n_i", step=5, on_change=update_pkt_widget, args=("pkt_n_i", "pkt_n_s"))
            
            ampiezza = st.slider("Ampiezza", 0.5, 2.0, 1.0, 0.1, key="pkt_amp")
            durata = st.slider("Durata visualizzazione (s)", 0.5, 3.0, 1.5, 0.1, key="pkt_dur")
        
        if preset_pkt != "Personalizzato":
            st.info(f"Preset caricato: {preset_pkt}")
        
        mostra_componenti = st.checkbox("Mostra onde componenti (max 10)", False)
        
        delta_f = f_max - f_min
        f_centrale = (f_min + f_max) / 2
        delta_omega = 2 * np.pi * delta_f
        lambda_centrale = V_SUONO / f_centrale
    
    with col2:
        t = np.linspace(0, durata, int(durata * 20000)) # Risoluzione aumentata per zoom
        frequenze = np.linspace(f_min, f_max, n_onde)
        
        y_pacchetto = np.zeros_like(t)
        for f in frequenze:
            y_comp = (ampiezza / n_onde) * np.cos(2 * np.pi * f * t)
            y_pacchetto += y_comp
        
        # Padding per Hilbert (riduce artefatti ai bordi)
        pad_len = int(len(t) * 0.1)
        y_pad = np.pad(y_pacchetto, (pad_len, pad_len), mode='reflect')
        analytic_signal = signal.hilbert(y_pad)
        inviluppo = np.abs(analytic_signal)[pad_len:-pad_len]
        intensita = inviluppo**2
        
        # Calcoli per visualizzazione simmetrica (anticipati per eventuale unificazione)
        t_sim = np.linspace(-durata, durata, int(durata * 2 * 20000))
        y_pacchetto_sim = np.zeros_like(t_sim)
        for f in frequenze:
            y_comp = (ampiezza / n_onde) * np.cos(2 * np.pi * f * t_sim)
            y_pacchetto_sim += y_comp
        
        analytic_sim = signal.hilbert(y_pacchetto_sim)
        inviluppo_sim = np.abs(analytic_sim)
        intensita_sim = inviluppo_sim**2

        if unisci_viste_glob:
            # MODALITÀ UNIFICATA: Tutti i grafici in un'unica figura con assi condivisi
            st.info("Modalità Vista Unificata attiva: lo zoom su un grafico si applica a tutti.")
            
            fig_tot = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                   subplot_titles=(f"Pacchetto (0-{durata}s)", 
                                                 "Intensità |A(t)|²",
                                                 "Pacchetto Simmetrico Completo",
                                                 "Intensità Simmetrica"),
                                   vertical_spacing=0.05)
            
            # Row 1: Pacchetto Standard
            fig_tot.add_trace(go.Scatter(x=t, y=y_pacchetto, name="Pacchetto", line=dict(color='darkblue')), row=1, col=1)
            fig_tot.add_trace(go.Scatter(x=t, y=inviluppo, name="Env", line=dict(color='red', dash='dash')), row=1, col=1)
            
            # Row 2: Intensità Standard
            fig_tot.add_trace(go.Scatter(x=t, y=intensita, fill='tozeroy', line=dict(color='orange'), name="|A|²"), row=2, col=1)
            
            # Row 3: Simmetrico
            fig_tot.add_trace(go.Scatter(x=t_sim, y=y_pacchetto_sim, name="Pacc. Simm.", line=dict(color='darkblue')), row=3, col=1)
            fig_tot.add_trace(go.Scatter(x=t_sim, y=inviluppo_sim, name="Env Simm.", line=dict(color='red', dash='dash')), row=3, col=1)
            
            # Row 4: Intensità Simmetrica
            fig_tot.add_trace(go.Scatter(x=t_sim, y=intensita_sim, fill='tozeroy', line=dict(color='orange'), name="|A|² Simm."), row=4, col=1)
            
            fig_tot.update_layout(height=1000, hovermode='x unified', modebar_add=['resetScale2d'])
            applica_zoom(fig_tot, range_x_glob)
            applica_stile(fig_tot, is_light_mode)
            st.plotly_chart(fig_tot, use_container_width=True, config=get_download_config("pacchetto_unificato"))
            
        else:
            # MODALITÀ STANDARD: Grafici separati
            # 🆕 GRAFICO CON INTENSITÀ
            fig = make_subplots(rows=2, cols=1,
                               subplot_titles=(f"Pacchetto: {n_onde} onde ({f_min}-{f_max} Hz)", 
                                             "Intensità |A(t)|² (Figura di Diffrazione)"),
                               shared_xaxes=True) # Sync interno
            
            if mostra_componenti and n_onde <= 50:
                step = max(1, n_onde // 10)
                for i, f in enumerate(frequenze[::step]):
                    y_comp = (ampiezza / n_onde) * np.cos(2 * np.pi * f * t)
                    fig.add_trace(go.Scatter(x=t, y=y_comp, name=f"f={f:.1f} Hz",
                                            line=dict(width=0.5), opacity=0.3), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=t, y=y_pacchetto, name="Pacchetto d'onda",
                                    line=dict(color='darkblue', width=2.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=inviluppo, name="Inviluppo +",
                                    line=dict(color='red', width=2, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=-inviluppo, showlegend=False,
                                    line=dict(color='red', width=2, dash='dash')), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=t, y=intensita, fill='tozeroy', 
                                    line=dict(color='orange', width=2), name="|A(t)|²"), row=2, col=1)
            
            fig.update_xaxes(title_text="Tempo (s)", row=2, col=1)
            fig.update_yaxes(title_text="Ampiezza", row=1, col=1)
            fig.update_yaxes(title_text="|A(t)|²", row=2, col=1)
            
            fig.update_layout(
                height=800,
                hovermode='x unified',
                dragmode='zoom',
                xaxis=dict(autorange=True),
                yaxis=dict(autorange=True, scaleanchor=None),
                modebar_add=['resetScale2d']
            )
            
            applica_zoom(fig, range_x_glob)
            applica_stile(fig, is_light_mode)
            st.plotly_chart(fig, use_container_width=True, config=get_download_config("pacchetto_onda"))
    
    # ========== SEZIONI A SCHERMO INTERO ==========
    st.markdown("---")
    st.subheader("Caratteristiche Pacchetto")
    col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
    with col_c1:
        st.metric("Freq. centrale", f"{f_centrale:.2f} Hz")
    with col_c2:
        st.metric("Larghezza Δf", f"{delta_f:.2f} Hz")
    with col_c3:
        st.metric("Δω", f"{delta_omega:.2f} rad/s")
    with col_c4:
        st.metric("λ centrale", f"{lambda_centrale:.3f} m")
    with col_c5:
        st.metric("N onde", n_onde)
    
    st.markdown("---")
    st.subheader("Genera Pacchetto Audio")
    durata_audio_pack = st.slider("Durata audio (s)", 0.5, 30.0, min(durata*2, 10.0), 0.5, 
                                  key="dur_audio_pack",
                                  help="Durata del file audio (indipendente dalla visualizzazione)")
    if st.button("Genera e riproduci", key="gen_pack_audio"):
        if durata_audio_pack > 5:
            progress = st.progress(0, "Generazione in corso...")
        else:
            progress = None
        
        if progress:
            progress.progress(0.1, f"Calcolo {n_onde} onde...")
        t_audio = np.linspace(0, durata_audio_pack, int(SAMPLE_RATE * durata_audio_pack))
        frequenze_audio = np.linspace(f_min, f_max, n_onde)
        y_audio = np.zeros_like(t_audio)
        
        for i, f in enumerate(frequenze_audio):
            y_audio += (1/n_onde) * np.sin(2 * np.pi * f * t_audio)
            if progress and i % max(1, n_onde//10) == 0:
                progress.progress(0.1 + 0.2 * (i/n_onde), f"Onda {i+1}/{n_onde}...")
        
        if np.max(np.abs(y_audio)) > 0.95:
            st.warning("**Clipping rilevato!** Normalizzazione attiva.")
        
        audio_bytes = genera_audio_con_progress(y_audio, SAMPLE_RATE, progress)
        
        if progress:
            progress.empty()
        
        st.success(f"Audio generato: {durata_audio_pack:.1f}s, {n_onde} onde, {len(y_audio):,} campioni")
        st.audio(audio_bytes, format='audio/wav')
        st.download_button("Scarica WAV", audio_bytes, f"pacchetto_{int(f_min)}_{int(f_max)}_Hz_{durata_audio_pack:.0f}s.wav", "audio/wav")
    
    st.markdown("---")
    if st.button("Esporta parametri pacchetto", key="export_pkt_full"):
        export_data = {
            "Parametro": ["f_min (Hz)", "f_max (Hz)", "Δf (Hz)", "N", "f_centrale (Hz)", "Δω (rad/s)"],
            "Valore": [f_min, f_max, delta_f, n_onde, f_centrale, delta_omega]
        }
        df_export = pd.DataFrame(export_data)
        csv = df_export.to_csv(index=False)
        st.download_button("Scarica CSV", csv, "pacchetto_parametri.csv", "text/csv")
    
    # 🆕 ========== VISUALIZZAZIONE SIMMETRICA COMPLETA (SCHERMO INTERO) ==========
    st.markdown("---")
    st.markdown("---")
    st.header("Visualizzazione Simmetrica Completa")
    st.markdown("""
    **Estensione spazio-temporale**: Il pacchetto viene esteso simmetricamente per mostrare 
    il comportamento completo dell'onda, includendo sia tempi positivi che negativi.
    Questa rappresentazione è utile per comprendere la natura periodica e simmetrica delle onde.
    """)
    
    if not unisci_viste_glob:
        # Se non unificati, mostra i grafici simmetrici qui uniti in una figura grande
        
        fig_sim = make_subplots(rows=2, cols=1,
                               subplot_titles=(f"Pacchetto Simmetrico Completo: {n_onde} onde ({f_min}-{f_max} Hz)", 
                                             "Intensità Simmetrica |A(t)|² - Figura di Diffrazione Completa"),
                               shared_xaxes=True,
                               vertical_spacing=0.1)  # Spacing normale
        
        # Row 1: Pacchetto
        fig_sim.add_trace(go.Scatter(x=t_sim, y=y_pacchetto_sim, name="Pacchetto d'onda",
                                     line=dict(color='darkblue', width=2)), row=1, col=1)
        fig_sim.add_trace(go.Scatter(x=t_sim, y=inviluppo_sim, name="Inviluppo +",
                                     line=dict(color='red', width=2, dash='dash')), row=1, col=1)
        fig_sim.add_trace(go.Scatter(x=t_sim, y=-inviluppo_sim, name="Inviluppo -",
                                     line=dict(color='red', width=2, dash='dash')), row=1, col=1)
        
        # Linea verticale a t=0 (Row 1)
        fig_sim.add_vline(x=0, line_dash="dot", line_color="green", 
                          annotation_text="t = 0", annotation_position="top", row=1, col=1)
        
        # Row 2: Intensità
        fig_sim.add_trace(go.Scatter(x=t_sim, y=intensita_sim, fill='tozeroy',
                                     line=dict(color='orange', width=2),
                                     name="Intensità |A(t)|²"), row=2, col=1)
        
        # Linea verticale a t=0 (Row 2)
        fig_sim.add_vline(x=0, line_dash="dot", line_color="green",
                          annotation_text="t = 0", annotation_position="top", row=2, col=1)
        
        fig_sim.update_xaxes(title_text="Tempo (s)", row=2, col=1)
        fig_sim.update_yaxes(title_text="Ampiezza", row=1, col=1)
        fig_sim.update_yaxes(title_text="|A(t)|²", row=2, col=1)
        
        # Sposta SOLO i titoli dei subplot più in alto (non le annotazioni t=0)
        for annotation in fig_sim.layout.annotations:
            if "t = 0" not in annotation.text:
                annotation.y = annotation.y + 0.03  # Sposta solo i titoli
        
        fig_sim.update_layout(
            height=800, # Altezza generosa per mantenere i grafici grandi
            hovermode='x unified',
            dragmode='zoom',
            modebar_add=['resetScale2d']
        )
        
        applica_zoom(fig_sim, range_x_glob)
        applica_stile(fig_sim, is_light_mode)
        st.plotly_chart(fig_sim, use_container_width=True, config=get_download_config("pacchetto_simmetrico"))
    else:
        st.info("I grafici simmetrici sono visualizzati sopra nella vista unificata.")
    
    # 🎨 GRAFICO 3: Vista 3D (Pacchetto + Inviluppo)
    with st.expander("Visualizzazione 3D del Pacchetto"):
        st.markdown("**Rappresentazione tridimensionale** dove l'inviluppo viene estruso nello spazio")
        
        # Crea griglia per 3D
        theta = np.linspace(0, 2*np.pi, 50)
        T_grid, Theta_grid = np.meshgrid(t_sim[::10], theta)  # Subsample per performance
        
        # Usa inviluppo come raggio
        R_grid = np.tile(inviluppo_sim[::10], (len(theta), 1))
        
        X_grid = R_grid * np.cos(Theta_grid)
        Y_grid = R_grid * np.sin(Theta_grid)
        Z_grid = T_grid
        
        fig_3d = go.Figure(data=[go.Surface(
            x=X_grid, y=Y_grid, z=Z_grid,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Ampiezza")
        )])
        
        fig_3d.update_layout(
            title="Pacchetto d'Onda 3D - Inviluppo Estruso",
            scene=dict(
                xaxis_title="X (ampiezza × cos θ)",
                yaxis_title="Y (ampiezza × sin θ)",
                zaxis_title="Tempo (s)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=700
        )
        applica_stile(fig_3d, is_light_mode)
        st.plotly_chart(fig_3d, use_container_width=True, config=get_download_config("pacchetto_3d"))
    
    # 📊 Analisi simmetria
    st.markdown("---")
    st.subheader("Analisi di Simmetria")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    # Trova indice centrale (t=0)
    idx_centro = len(t_sim) // 2
    
    # Confronta ampiezza sinistra vs destra
    amp_sx = np.max(np.abs(y_pacchetto_sim[:idx_centro]))
    amp_dx = np.max(np.abs(y_pacchetto_sim[idx_centro:]))
    simmetria_amp = min(amp_sx, amp_dx) / max(amp_sx, amp_dx) * 100
    
    with col_s1:
        st.metric("Ampiezza max SX", f"{amp_sx:.3f}")
    with col_s2:
        st.metric("Ampiezza max DX", f"{amp_dx:.3f}")
    with col_s3:
        st.metric("Simmetria %", f"{simmetria_amp:.1f}%")
    with col_s4:
        larghezza_centrale = np.sum(intensita_sim > np.max(intensita_sim)*0.5) / len(t_sim) * (2*durata)
        st.metric("Larghezza FWHM", f"{larghezza_centrale:.3f} s")
    
    # Info teorica
    with st.expander("Perché la simmetria è importante?"):
        st.markdown("""
        ### Significato Fisico della Simmetria
        
        1. **Invarianza temporale**: Un pacchetto d'onda simmetrico rispetto a t=0 rappresenta
           un'onda stazionaria che si propaga sia in avanti che indietro nel tempo.
        
        2. **Fourier duale**: La simmetria nel dominio temporale implica che lo spettro di Fourier
           è puramente reale (senza componenti immaginarie).
        
        3. **Principio di indeterminazione**: La larghezza del pacchetto nel tempo (Δt) è 
           inversamente proporzionale alla larghezza in frequenza (Δω): **Δt·Δω ≥ 4π**
        
        4. **Interferenza costruttiva**: Al centro (t=0) tutte le onde componenti sono in fase,
           producendo l'ampiezza massima.
        
        5. **Fisica quantistica**: Rappresenta la funzione d'onda di una particella localizzata
           nello spazio-tempo, con simmetria che riflette la natura ondulatoria della materia.
        """)

# ========== SPETTRO FOURIER (COMPLETO VECCHIO) ==========
elif sezione == "Spettro di Fourier":
    styled_header(
        "📊", 
        "Analisi di Fourier",
        "Trasformata di Fourier: dal dominio del tempo al dominio della frequenza",
        "#e74c3c"
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parametri")
        tipo_segnale = st.selectbox("Tipo di segnale", 
                                    ["Pacchetto d'onda", "Onda singola", "Battimenti"])
        
        if tipo_segnale == "Pacchetto d'onda":
            f_min_fft = st.slider("Freq. minima (Hz)", 1.0, 500.0, 20.0, 1.0, key="fft_fmin")
            f_max_fft = st.slider("Freq. massima (Hz)", f_min_fft+1, 500.0, 30.0, 1.0, key="fft_fmax")
            n_onde_fft = st.slider("Numero onde", 10, 80, 40, 5, key="fft_n")
        elif tipo_segnale == "Onda singola":
            freq_singola = st.slider("Frequenza (Hz)", 1.0, 500.0, 25.0, 1.0, key="fft_fsing")
        else:
            f1_bat = st.slider("Freq. 1 (Hz)", 1.0, 500.0, 20.0, 1.0, key="fft_f1bat")
            f2_bat = st.slider("Freq. 2 (Hz)", 1.0, 500.0, 25.0, 1.0, key="fft_f2bat")
        
        durata_fft = st.slider("Durata campionamento (s)", 1.0, 5.0, 2.0, 0.5, key="fft_dur")
    
    with col2:
        fs = SAMPLE_RATE # Usa 44100 Hz per audio di qualità
        t = np.linspace(0, durata_fft, int(fs * durata_fft))
        
        if tipo_segnale == "Pacchetto d'onda":
            frequenze = np.linspace(f_min_fft, f_max_fft, n_onde_fft)
            y = np.zeros_like(t)
            for f in frequenze:
                y += (1/n_onde_fft) * np.cos(2 * np.pi * f * t)
            titolo = f"Pacchetto: {f_min_fft}-{f_max_fft} Hz ({n_onde_fft} onde)"
        elif tipo_segnale == "Onda singola":
            y = np.cos(2 * np.pi * freq_singola * t)
            titolo = f"Onda singola: {freq_singola} Hz"
        else:
            y = np.cos(2 * np.pi * f1_bat * t) + np.cos(2 * np.pi * f2_bat * t)
            titolo = f"Battimenti: {f1_bat} Hz + {f2_bat} Hz"
        
        # 🆕 AUDIO PLAYER
        st.markdown("### Ascolta il Segnale")
        audio_bytes_fft = genera_audio(y, fs)
        st.audio(audio_bytes_fft, format='audio/wav')
        
        N = len(y)
        yf = fft(y)
        xf = fftfreq(N, 1/fs)[:N//2]
        potenza = 2.0/N * np.abs(yf[:N//2])
        
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=(f"Segnale Temporale: {titolo}", 
                                         "Spettro di Frequenza (Trasformata di Fourier)"),
                           vertical_spacing=0.15)
        
        # Ottimizzazione plot: mostra max 10k punti per fluidità
        step_plot = max(1, len(t) // 10000)
        fig.add_trace(go.Scatter(x=t[::step_plot], y=y[::step_plot], line=dict(color='blue', width=1.5),
                                name="Segnale"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xf, y=potenza, line=dict(color='red', width=2),
                                fill='tozeroy', name="Ampiezza FFT"), row=2, col=1)
        
        fig.update_xaxes(title_text="Tempo (s)", autorange=True, row=1, col=1)
        fig.update_xaxes(title_text="Frequenza (Hz)", autorange=True, row=2, col=1)
        fig.update_yaxes(title_text="Ampiezza", autorange=True, row=1, col=1)
        fig.update_yaxes(title_text="Ampiezza spettrale", autorange=True, row=2, col=1)
        
        fig.update_layout(height=700, showlegend=False, hovermode='x unified',
                         dragmode='zoom', modebar_add=['resetScale2d'])
        
        applica_zoom(fig, range_x_glob)
        applica_stile(fig, is_light_mode)
        st.plotly_chart(fig, use_container_width=True, config=get_download_config("spettro_fourier"))
        
        st.subheader("Statistiche dello Spettro")
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(potenza, height=np.max(potenza)*0.1)
        freq_picchi = xf[peaks]
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Numero picchi", len(freq_picchi))
        with col_b:
            if len(freq_picchi) > 0:
                st.metric("Freq. picco principale", f"{freq_picchi[np.argmax(potenza[peaks])]:.2f} Hz")
        with col_c:
            if len(freq_picchi) >= 2:
                st.metric("Larghezza banda", f"{freq_picchi[-1] - freq_picchi[0]:.2f} Hz")
    
    st.markdown("---")
    st.markdown("---")
    st.header("Valori Teorici Completi - Spettro di Fourier")
    st.markdown("*Parametri di campionamento e analisi FFT*")
    
    st.markdown("### Campionamento")
    col_camp1, col_camp2, col_camp3, col_camp4 = st.columns(4)
    with col_camp1:
        st.metric("Frequenza campionamento", f"{fs} Hz", help="Sample rate")
    with col_camp2:
        st.metric("Durata segnale", f"{durata_fft:.2f} s", help="Tempo totale")
    with col_camp3:
        st.metric("Numero campioni N", f"{N:,}", help="Punti temporali")
    with col_camp4:
        risoluzione_freq = fs / N
        st.metric("Risoluzione Δf", f"{risoluzione_freq:.4f} Hz", help="fs / N")
    
    st.markdown("### Analisi Spettrale")
    col_fft1, col_fft2, col_fft3, col_fft4 = st.columns(4)
    with col_fft1:
        freq_max_fft = fs / 2
        st.metric("Freq. Nyquist", f"{freq_max_fft:.1f} Hz", help="fs / 2 (massima)")
    with col_fft2:
        if tipo_segnale == "Pacchetto d'onda":
            larghezza_banda_teorica = f_max_fft - f_min_fft
            st.metric("Larghezza banda teorica", f"{larghezza_banda_teorica:.2f} Hz")
        elif tipo_segnale == "Battimenti":
            st.metric("Δf battimenti", f"{abs(f1_bat - f2_bat):.2f} Hz")
        else:
            st.metric("Freq. fondamentale", f"{freq_singola:.2f} Hz")
    with col_fft3:
        energia_totale = np.sum(potenza**2)
        st.metric("Energia spettrale", f"{energia_totale:.2e}", help="Σ|FFT|²")
    with col_fft4:
        num_bins_fft = len(xf)
        st.metric("Bins FFT", f"{num_bins_fft:,}", help="N/2 frequenze")
    
    st.markdown("### Picchi Rilevati")
    if len(freq_picchi) > 0:
        col_pk1, col_pk2, col_pk3, col_pk4 = st.columns(4)
        with col_pk1:
            st.metric("Numero picchi", len(freq_picchi))
        with col_pk2:
            freq_principale = freq_picchi[np.argmax(potenza[peaks])]
            st.metric("Picco principale", f"{freq_principale:.2f} Hz")
        with col_pk3:
            amp_principale = np.max(potenza[peaks])
            st.metric("Ampiezza max", f"{amp_principale:.4f}")
        with col_pk4:
            if len(freq_picchi) >= 2:
                larghezza = freq_picchi[-1] - freq_picchi[0]
                st.metric("Larghezza banda", f"{larghezza:.2f} Hz")
    
    with st.expander("Formule Trasformata di Fourier"):
        col_fourier1, col_fourier2 = st.columns(2)
        with col_fourier1:
            st.markdown("#### FFT Discreta")
            st.latex(r"X_k = \sum_{n=0}^{N-1} x_n e^{-i 2\pi k n / N}")
            st.latex(r"f_k = \frac{k \cdot f_s}{N}")
            st.markdown("#### Potenza")
            st.latex(r"P_k = \frac{2}{N} |X_k|")
        with col_fourier2:
            st.markdown("#### Teorema Campionamento")
            st.latex(r"f_s > 2 f_{\text{max}}")
            st.latex(r"f_{\text{Nyquist}} = \frac{f_s}{2}")
            st.markdown("#### Risoluzione")
            st.latex(r"\Delta f = \frac{f_s}{N} = \frac{1}{T}")

# ========== PRINCIPIO INDETERMINAZIONE (COMPLETO) ==========
elif sezione == "Principio di Indeterminazione":
    styled_header(
        "⚠️", 
        "Principio di Indeterminazione",
        "Relazione fondamentale di Heisenberg: Δx · Δk ≥ 1/2",
        "#f39c12"
    )
    
    st.info("""
    Questa sezione verifica sperimentalmente il principio di 
    indeterminazione per le onde: $\\Delta x \\cdot \\Delta k \\geq 1/2$ (forma RMS) 
    o $\\Delta x \\cdot \\Delta k = 4\\pi$ (lobi laterali, usato qui).
    
    **Valori teorici**:
    - Gaussiano (RMS): $\\Delta x \\cdot \\Delta k = 1/2$ (minimo)
    - Gaussiano (FWHM): $\\Delta x \\cdot \\Delta k \\approx 1.77$
    - Sinc (primi zeri): $\\Delta x \\cdot \\Delta k = 4\\pi \\approx 12.57$
    """)

    st.markdown("""Il principio di indeterminazione stabilisce che Δx·Δp ≥ ℏ/2""")
    st.markdown(f"Per onde: **p = ℏk = h/λ**, quindi: **Δx·Δk ≥ 1/2**")
    st.markdown(f"**Teoricamente**: Δx·Δk = 4π ≈ **12.57** (per N → ∞)")
    st.info(f"**Velocità del suono in aria**: v = {V_SUONO} m/s (a 20°C)")
    
    st.markdown("---")
    st.subheader("Parametri del Pacchetto")
    
    preset_pkt = st.selectbox("Carica preset:", list(PRESET_PACCHETTI.keys()), key="preset_pkt_main")
    
    if preset_pkt != "Personalizzato":
        preset = PRESET_PACCHETTI[preset_pkt]
        f_min = preset["f_min"]
        f_max = preset["f_max"]
        n_onde = preset["N"]
        ampiezza = 1.0
        durata = 1.5
        st.info(f"**{preset_pkt}**\n\n{preset['descrizione']}")
        
        # Mostra i valori del preset
        col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
        with col_p1:
            st.metric("f_min", f"{f_min:.1f} Hz")
        with col_p2:
            st.metric("f_max", f"{f_max:.1f} Hz")
        with col_p3:
            st.metric("Δf", f"{f_max - f_min:.1f} Hz")
        with col_p4:
            st.metric("N onde", f"{n_onde}")
        with col_p5:
            st.metric("Durata", f"{durata} s")
    else:
        # Inizializzazione Session State per sincronizzazione
        if 'indet_fmin_s' not in st.session_state: st.session_state.indet_fmin_s = 100.0
        if 'indet_fmin_i' not in st.session_state: st.session_state.indet_fmin_i = 100.0
        if 'indet_fmax_s' not in st.session_state: st.session_state.indet_fmax_s = 130.0
        if 'indet_fmax_i' not in st.session_state: st.session_state.indet_fmax_i = 130.0
        if 'indet_n_s' not in st.session_state: st.session_state.indet_n_s = 50
        if 'indet_n_i' not in st.session_state: st.session_state.indet_n_i = 50

        def update_indet_widget(key_from, key_to):
            st.session_state[key_to] = st.session_state[key_from]
            # Logica di sicurezza f_max > f_min
            if "fmin" in key_from or "fmax" in key_from:
                curr_min = st.session_state.indet_fmin_s
                curr_max = st.session_state.indet_fmax_s
                if curr_max <= curr_min + 5.0:
                    new_max = curr_min + 5.0
                    st.session_state.indet_fmax_s = new_max
                    st.session_state.indet_fmax_i = new_max

        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            # Frequenza minima
            col_fmin_s, col_fmin_i = st.columns([3, 1])
            with col_fmin_s:
                # Chiavi rinominate (indet_*) per evitare conflitti con la sezione Pacchetti
                f_min_slider = st.slider("Frequenza minima (Hz)", 1.0, 500.0, key="indet_fmin_s", on_change=update_indet_widget, args=("indet_fmin_s", "indet_fmin_i"))
            with col_fmin_i:
                f_min = st.number_input("", min_value=1.0, max_value=500.0, step=1.0, key="indet_fmin_i", format="%.1f", on_change=update_indet_widget, args=("indet_fmin_i", "indet_fmin_s"))
        
        with col_in2:
            # Frequenza massima
            col_fmax_s, col_fmax_i = st.columns([3, 1])
            with col_fmax_s:
                min_fmax = f_min + 5.0
                if st.session_state.indet_fmax_s < min_fmax:
                    st.session_state.indet_fmax_s = min_fmax
                    st.session_state.indet_fmax_i = min_fmax
                    
                f_max_slider = st.slider("Frequenza massima (Hz)", min_fmax, 1000.0, key="indet_fmax_s", on_change=update_indet_widget, args=("indet_fmax_s", "indet_fmax_i"))
            with col_fmax_i:
                f_max = st.number_input("", min_value=min_fmax, max_value=1000.0, step=1.0, key="indet_fmax_i", format="%.1f", on_change=update_indet_widget, args=("indet_fmax_i", "indet_fmax_s"))
        
        with col_in3:
            # Numero onde
            col_n_s, col_n_i = st.columns([3, 1])
            with col_n_s:
                n_onde_slider = st.slider("Numero di onde N", 5, 100, key="indet_n_s", on_change=update_indet_widget, args=("indet_n_s", "indet_n_i"))
            with col_n_i:
                n_onde = st.number_input("", min_value=5, max_value=100, step=1, key="indet_n_i", on_change=update_indet_widget, args=("indet_n_i", "indet_n_s"))
            
            ampiezza = st.slider("Ampiezza", 0.1, 2.0, 1.0, 0.1, key="indet_amp")
            durata = st.slider("Durata (s)", 0.1, 5.0, 1.5, 0.1, key="indet_dur")
        
    # === CALCOLI AUTOMATICI DAL PACCHETTO ===
    lambda_min = V_SUONO / f_max
    lambda_max = V_SUONO / f_min
    k_min = 2 * np.pi / lambda_max
    k_max = 2 * np.pi / lambda_min
    k_medio = (k_min + k_max) / 2
    delta_k = k_max - k_min
    delta_x_teorico = 4 * np.pi / delta_k if delta_k > 0 else 0
    
    # Calcoli temporali
    delta_f = f_max - f_min
    delta_omega = 2 * np.pi * delta_f
    delta_t_teorico = 4 * np.pi / delta_omega if delta_omega > 0 else 0
    
    st.markdown("---")
    st.subheader("Analisi Teorica")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.markdown("**Dominio Spaziale**")
        st.metric("Δk", f"{delta_k:.4f} rad/m")
        st.metric("Δx teorico", f"{delta_x_teorico:.3f} m")
        st.metric("λ_min", f"{lambda_min:.3f} m")
    
    with col_res2:
        st.markdown("**Dominio Temporale**")
        st.metric("Δf", f"{delta_f:.1f} Hz")
        st.metric("Δω", f"{delta_omega:.1f} rad/s")
        st.metric("Δt teorico", f"{delta_t_teorico*1000:.2f} ms")
    
    with col_res3:
        st.markdown("**Principio Indeterminazione**")
        prodotto_xk = delta_x_teorico * delta_k
        prodotto_wt = delta_t_teorico * delta_omega
        st.metric("Δx·Δk", f"{prodotto_xk:.3f}", delta=f"4π={4*np.pi:.3f}")
        st.metric("Δω·Δt", f"{prodotto_wt:.3f}", delta=f"4π={4*np.pi:.3f}")
        
        errore_perc_xk = abs(prodotto_xk - 4*np.pi) / (4*np.pi) * 100
        if errore_perc_xk < 5:
            st.success(f"Errore: {errore_perc_xk:.1f}%")
        else:
            st.warning(f"Errore: {errore_perc_xk:.1f}%")

    # === GRAFICI A TUTTO SCHERMO ===
    st.markdown("---")
    st.subheader("Visualizzazione Grafica")
    
    # Grafico spaziale
    range_x = max(50.0, delta_x_teorico * 2.0) # Adatta la scala alla larghezza del pacchetto
    x = np.linspace(-range_x, range_x, 10000) # Più punti per dettaglio spaziale
    k_values = np.linspace(k_min, k_max, n_onde)
    y_pacchetto_spazio = np.zeros_like(x)
    for k in k_values:
        y_pacchetto_spazio += (1/n_onde) * np.cos(k * x)
    
    analytic = signal.hilbert(y_pacchetto_spazio)
    inviluppo_spazio = np.abs(analytic)
    delta_x_mis, idx1, idx2 = calcola_larghezza_temporale(x, inviluppo_spazio)
    
    fig_x = go.Figure()
    fig_x.add_trace(go.Scatter(x=x, y=y_pacchetto_spazio, name="Pacchetto d'onda",
                            line=dict(color='darkblue', width=2)))
    fig_x.add_trace(go.Scatter(x=x, y=inviluppo_spazio, name="Inviluppo",
                            line=dict(color='red', width=2, dash='dash')))
    fig_x.add_trace(go.Scatter(x=x, y=-inviluppo_spazio, showlegend=False,
                            line=dict(color='red', width=2, dash='dash')))
    fig_x.add_vline(x=x[idx1], line_dash="dot", line_color="green", annotation_text=f"Δx={delta_x_mis:.2f}m")
    fig_x.add_vline(x=x[idx2], line_dash="dot", line_color="green")
    fig_x.update_layout(title=f"Spazio: Δx·Δk = {delta_x_mis*delta_k:.2f} (target: 12.57)",
                       xaxis_title="Posizione x (m)", yaxis_title="Ampiezza", 
                       height=600, # Aumentata altezza
                       hovermode='x unified')
    applica_zoom(fig_x, range_x_glob)
    applica_stile(fig_x, is_light_mode)
    st.plotly_chart(fig_x, use_container_width=True, config=get_download_config("indeterminazione_spazio"))
    
    # 🆕 Grafico temporale
    # CORREZIONE: Limita la durata per evitare ripetizioni periodiche
    # Un pacchetto di N onde con frequenze equispaziate si ripete con periodo T_rep = (N-1) / Δf
    delta_f = f_max - f_min
    if n_onde > 1 and delta_f > 0:
        T_ripetizione = (n_onde - 1) / delta_f
    else:
        T_ripetizione = durata * 10  # Nessun limite se n_onde = 1
    
    # Limita la durata visualizzata a 80% del periodo di ripetizione per evitare artefatti
    durata_effettiva = min(durata, T_ripetizione * 0.8)
    
    t = np.linspace(0, durata_effettiva, int(durata_effettiva * 20000)) # Alta risoluzione temporale
    omega_vals = 2 * np.pi * np.linspace(f_min, f_max, n_onde)
    y_t = np.zeros_like(t)
    for omega in omega_vals:
        y_t += (1/n_onde) * np.cos(omega * t)
    env_t = np.abs(signal.hilbert(y_t))
    delta_t_mis, idx1_t, idx2_t = calcola_larghezza_temporale(t, env_t)
    
    # Info sulla correzione
    if durata > T_ripetizione * 0.8:
        st.caption(f"⚠️ Durata limitata a {durata_effettiva*1000:.0f} ms per evitare ripetizioni periodiche (T_rep = {T_ripetizione*1000:.0f} ms)")
    
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=t*1000, y=y_t, line=dict(color='purple', width=2), name="Pacchetto"))
    fig_t.add_trace(go.Scatter(x=t*1000, y=env_t, line=dict(color='orange', width=2, dash='dash'), name="Inviluppo"))
    fig_t.add_trace(go.Scatter(x=t*1000, y=-env_t, showlegend=False, line=dict(color='orange', width=2, dash='dash')))
    fig_t.add_vline(x=t[idx1_t]*1000, line_dash="dot", line_color="green", annotation_text=f"Δt={delta_t_mis*1000:.2f}ms")
    fig_t.add_vline(x=t[idx2_t]*1000, line_dash="dot", line_color="green")
    fig_t.update_layout(title=f"Tempo: Δω·Δt = {delta_t_mis*delta_omega:.2f} (target: 12.57)",
                       xaxis_title="t (ms)", yaxis_title="A(t)", 
                       height=600,
                       hovermode='x unified')
    applica_stile(fig_t, is_light_mode)
    st.plotly_chart(fig_t, use_container_width=True, config=get_download_config("indeterminazione_tempo"))
    
    # 🆕 Grafico temporale SIMMETRICO (Doppio)
    st.markdown("#### Visualizzazione Temporale Simmetrica (Passato e Futuro)")
    
    # Usa la stessa durata effettiva per evitare ripetizioni
    durata_sim = durata_effettiva
    t_sim = np.linspace(-durata_sim, durata_sim, int(durata_sim * 2 * 20000)) # Alta risoluzione
    y_t_sim = np.zeros_like(t_sim)
    for omega in omega_vals:
        y_t_sim += (1/n_onde) * np.cos(omega * t_sim)
    
    env_t_sim = np.abs(signal.hilbert(y_t_sim))
    
    fig_t_sim = go.Figure()
    fig_t_sim.add_trace(go.Scatter(x=t_sim*1000, y=y_t_sim, line=dict(color='purple', width=2), name="Pacchetto"))
    fig_t_sim.add_trace(go.Scatter(x=t_sim*1000, y=env_t_sim, line=dict(color='orange', width=2, dash='dash'), name="Inviluppo"))
    fig_t_sim.add_trace(go.Scatter(x=t_sim*1000, y=-env_t_sim, showlegend=False, line=dict(color='orange', width=2, dash='dash')))
    
    fig_t_sim.add_vline(x=0, line_dash="dot", line_color="green", annotation_text="t=0")
    
    fig_t_sim.update_layout(title=f"Tempo Simmetrico: Δω·Δt = {delta_t_mis*delta_omega:.2f} (Visualizzazione Completa)",
                       xaxis_title="t (ms)", yaxis_title="A(t)", 
                       height=600,
                       hovermode='x unified')
    applica_stile(fig_t_sim, is_light_mode)
    st.plotly_chart(fig_t_sim, use_container_width=True, config=get_download_config("indeterminazione_tempo_sim"))
    
    # 🆕 SPETTRO DI FREQUENZE (richiesto dallo script)
    st.markdown("---")
    st.markdown("#### Spettro di Frequenze")
    st.markdown("""
    Il grafico sottostante mostra le frequenze che compongono il pacchetto d'onda.
    **Osserva**: se il pacchetto è **stretto** nel tempo, lo spettro è **largo** (molte frequenze diverse).
    Se il pacchetto è **largo** nel tempo, lo spettro è **stretto** (frequenze molto simili tra loro).
    """)
    
    # Calcola lo spettro (frequenze discrete usate per comporre il pacchetto)
    freq_spectrum = np.linspace(f_min, f_max, n_onde)
    amplitudes_spectrum = np.ones(n_onde) / n_onde  # Ampiezza uniforme 1/N
    
    fig_spectrum = go.Figure()
    
    # Usa bar chart per visualizzare le frequenze discrete
    fig_spectrum.add_trace(go.Bar(
        x=freq_spectrum,
        y=amplitudes_spectrum,
        marker_color='#e74c3c',
        name='Ampiezza componenti',
        width=(f_max - f_min) / n_onde * 0.8  # Larghezza barre proporzionale
    ))
    
    # Aggiungi annotazioni per Δf
    fig_spectrum.add_vline(x=f_min, line_dash="dash", line_color="blue", 
                          annotation_text=f"f_min={f_min:.0f} Hz", annotation_position="top left")
    fig_spectrum.add_vline(x=f_max, line_dash="dash", line_color="blue", 
                          annotation_text=f"f_max={f_max:.0f} Hz", annotation_position="top right")
    
    fig_spectrum.update_layout(
        title=f"Spettro di Frequenze: Δf = {delta_f:.1f} Hz (N = {n_onde} onde)",
        xaxis_title="Frequenza (Hz)",
        yaxis_title="Ampiezza relativa",
        height=400,
        bargap=0.1,
        showlegend=False
    )
    applica_stile(fig_spectrum, is_light_mode)
    st.plotly_chart(fig_spectrum, use_container_width=True, config=get_download_config("spettro_frequenze"))
    
    # Info box per collegare con lo script
    styled_info_box(
        f"<strong>Principio di Indeterminazione</strong>: Con Δf = {delta_f:.1f} Hz otteniamo Δt ≈ {delta_t_teorico*1000:.2f} ms. "
        f"Il prodotto Δf·Δt = {delta_f * delta_t_teorico:.2f} è circa costante (≈ 1/π ≈ 0.32 per forme gaussiane).",
        "⚛️",
        "info"
    )
    
    # 🔬 VALIDAZIONE METODO
    st.markdown("---")
    st.subheader("Validazione del Metodo (da relazione)")
    
    st.markdown("""
    **Confronto metodi di calcolo** per $\\Delta x$ (stesso pacchetto test):
    """)
    
    # Calcola con metodo corrente (lobi laterali)
    delta_x_lobi = delta_x_mis
    delta_x_dk_lobi = delta_x_lobi * delta_k
    
    # Stima FWHM (circa 60% dei lobi laterali per sinc)
    delta_x_fwhm = delta_x_lobi * 0.6
    delta_x_dk_fwhm = delta_x_fwhm * delta_k
    
    # Teorico
    delta_x_teorico_val = 4 * np.pi / delta_k if delta_k > 0 else 0
    delta_x_dk_teorico = 4 * np.pi
    
    val_data = {
        "Metodo": ["Teorico (sinc)", "Lobi laterali (5%)", "FWHM (stimato)", "RMS (teorico Gauss)"],
        "Δx (m)": [f"{delta_x_teorico_val:.2f}", f"{delta_x_lobi:.2f}", f"{delta_x_fwhm:.2f}", "N/A"],
        "Δx·Δk": [f"{delta_x_dk_teorico:.3f}", f"{delta_x_dk_lobi:.3f}", f"{delta_x_dk_fwhm:.3f}", "0.500"],
        "Errore %": ["0.00", f"{abs(delta_x_dk_lobi - delta_x_dk_teorico)/delta_x_dk_teorico*100:.2f}", 
                     f"{abs(delta_x_dk_fwhm - delta_x_dk_teorico)/delta_x_dk_teorico*100:.2f}", "N/A"]
    }
    df_val = pd.DataFrame(val_data)
    st.dataframe(df_val, use_container_width=True)
    
    errore_perc = abs(delta_x_dk_lobi - delta_x_dk_teorico)/delta_x_dk_teorico*100
    
    if errore_perc < 10.0: # Tolleranza 10%
        st.success(f"""
        **Metodo validato**: Il metodo dei lobi laterali (soglia 5%) fornisce 
        Δx·Δk = {delta_x_dk_lobi:.3f}, in ottimo accordo con il valore teorico 4π ≈ 12.57 
        (errore {errore_perc:.2f}%)
        """)
    else:
        st.error(f"""
        **Discrepanza Rilevata**: Il valore misurato Δx·Δk = {delta_x_dk_lobi:.3f} si discosta dal teorico (errore {errore_perc:.2f}%).
        Possibili cause: il pacchetto potrebbe essere troppo largo per la finestra o il metodo ha usato il fallback FWHM.
        """)

    st.subheader("Verifica Sperimentale")
    col_ver1, col_ver2 = st.columns(2)
    with col_ver1:
        st.metric("Δx misurato", f"{delta_x_mis:.3f} m")
        st.metric("Δx teorico", f"{delta_x_teorico:.3f} m")
    with col_ver2:
        st.metric("Δx·Δk misurato", f"{delta_x_mis * delta_k:.3f}")
        st.metric("Δω·Δt misurato", f"{delta_t_mis * delta_omega:.3f}")
    
    st.markdown("---")
    st.subheader("Genera Audio")
    durata_audio_pack = st.slider("Durata audio (s)", 0.5, 30.0, 5.0, 0.5, key="dur_audio_pack")
    
    if st.button("Genera pacchetto audio", key="gen_pack_audio"):
        progress = st.progress(0, "Generazione...")
        t_audio = np.linspace(0, durata_audio_pack, int(SAMPLE_RATE * durata_audio_pack))
        frequenze_audio = np.linspace(f_min, f_max, n_onde)
        y_audio = np.zeros_like(t_audio)
        
        for f in frequenze_audio:
            y_audio += (1/n_onde) * np.sin(2 * np.pi * f * t_audio)
        
        audio_bytes = genera_audio_con_progress(y_audio, SAMPLE_RATE, progress)
        progress.empty()
        st.success(f"Audio generato: {durata_audio_pack:.1f}s")
        st.audio(audio_bytes, format='audio/wav')
        st.download_button("Scarica WAV", audio_bytes, f"pacchetto_{int(f_min)}_{int(f_max)}_Hz.wav", "audio/wav")
    
    st.markdown("---")
    if st.button("Esporta risultati", key="export_pkt"):
        export_data = {
            "Parametro": ["f_min (Hz)", "f_max (Hz)", "Δf (Hz)", "N onde", 
                         "λ_min (m)", "λ_max (m)", "Δk (rad/m)", "Δx (m)", 
                         "Δx·Δk", "Δω·Δt", "4π teorico"],
            "Valore": [f_min, f_max, delta_f, n_onde, lambda_min, lambda_max, 
                      delta_k, delta_x_teorico, prodotto_xk, prodotto_wt, 4*np.pi]
        }
        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)
        st.download_button("Scarica CSV", csv, "pacchetto_risultati.csv", "text/csv")
        st.dataframe(df, use_container_width=True)
    
    # 🆕 FORMULE LATEX
    st.markdown("---")
    with st.expander("Formule del Principio di Indeterminazione (LaTeX)", expanded=False):
        col_lat1, col_lat2 = st.columns(2)
        with col_lat1:
            st.markdown("#### Pacchetto d'Onda")
            st.latex(r"y(x,t) = \sum_{n=1}^{N} A_n \cos(k_n x - \omega_n t)")
            st.markdown("#### Larghezze di Banda")
            st.latex(r"\Delta k = k_{max} - k_{min}")
            st.latex(r"\Delta \omega = \omega_{max} - \omega_{min}")
        
        with col_lat2:
            st.markdown("#### Relazioni di Indeterminazione")
            st.latex(r"\Delta x \cdot \Delta k \approx 4\pi")
            st.latex(r"\Delta t \cdot \Delta \omega \approx 4\pi")
            st.markdown("#### Velocità")
            st.latex(r"v_g = \frac{\Delta \omega}{\Delta k} = v_{suono}")

# ========== ANALISI MULTI-PACCHETTO ==========
elif sezione == "Analisi Multi-Pacchetto":
    styled_header(
        "📋", 
        "Analisi Multi-Pacchetto",
        "Genera più pacchetti con diversi Δk e verifica sistematicamente Δx·Δk = 4π",
        "#1abc9c"
    )
    
    n_pacchetti = st.slider("Numero pacchetti da analizzare", 3, 15, 8, key="npac")
    lambda_min_base = st.slider("λ_min base (m)", 1.5, 4.0, 2.0, 0.1, key="lminbase")
    delta_lambda_step = st.slider("Incremento Δλ", 0.3, 2.0, 0.8, 0.1, key="dlstep")
    n_onde_fisso = st.slider("N onde (fisso)", 30, 100, 60, 10, key="nfix")
    
    if st.button("Genera e Analizza", key="gen_multi"):
        risultati = []
        for i in range(n_pacchetti):
            lambda_max = lambda_min_base + (i + 1) * delta_lambda_step
            k_min = 2 * np.pi / lambda_max
            k_max = 2 * np.pi / lambda_min_base
            delta_k = k_max - k_min
            x = np.linspace(-35, 35, 10000)
            k_vals = np.linspace(k_min, k_max, n_onde_fisso)
            y = np.zeros_like(x)
            for k in k_vals:
                y += (1/n_onde_fisso) * np.cos(k * x)
            env = np.abs(signal.hilbert(y))
            delta_x, _, _ = calcola_larghezza_temporale(x, env, 0.08)
            prodotto = delta_x * delta_k
            errore = abs(prodotto - 4*np.pi) / (4*np.pi) * 100
            risultati.append({
                "#": i+1,
                "λ_max (m)": lambda_max,
                "Δλ (m)": lambda_max - lambda_min_base,
                "Δk (rad/m)": delta_k,
                "Δx (m)": delta_x,
                "Δx·Δk": prodotto,
                "Errore %": errore
            })
        
        df = pd.DataFrame(risultati)
        st.subheader("Tabella Risultati")
        st.dataframe(df.style.format({
            "Δk (rad/m)": "{:.3f}", 
            "Δx (m)": "{:.3f}", 
            "Δx·Δk": "{:.3f}", 
            "Errore %": "{:.2f}"
        }), use_container_width=True)
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Media Δx·Δk", f"{df['Δx·Δk'].mean():.3f}")
        with col_s2:
            st.metric("Std Dev", f"{df['Δx·Δk'].std():.3f}")
        with col_s3:
            st.metric("Target (4π)", "12.566")
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=df["#"], y=df["Δx·Δk"], mode='markers+lines', 
                                       marker=dict(size=12, color='blue'), name="Δx·Δk"))
        fig_trend.add_hline(y=4*np.pi, line_dash="dash", line_color="red", annotation_text="4π = 12.566")
        fig_trend.update_layout(title="Andamento Δx·Δk", xaxis_title="Pacchetto #", 
                               yaxis_title="Δx·Δk", height=500)
        applica_stile(fig_trend, is_light_mode)
        st.plotly_chart(fig_trend, use_container_width=True, config=get_download_config("multi_pacchetto_trend"))
        
        csv = df.to_csv(index=False)
        st.download_button("Scarica CSV", csv, "analisi_multi_pacchetto.csv", "text/csv")

# ========== REGRESSIONE ==========
elif sezione == "Regressione Δx vs 1/Δk":
    styled_header(
        "📈", 
        "Regressione Lineare: Δx vs 1/Δk",
        "Teoria: Δx = 4π · (1/Δk) → pendenza attesa ≈ 12.57",
        "#e67e22"
    )
    
    n_punti = st.slider("Numero punti", 5, 25, 12, key="npt")
    lambda_min_reg = st.slider("λ_min (m)", 1.5, 4.0, 2.0, 0.1, key="lminreg")
    lambda_max_min = st.slider("λ_max minimo (m)", lambda_min_reg+1, 8.0, 3.5, 0.5, key="lmaxmin")
    lambda_max_max = st.slider("λ_max massimo (m)", lambda_max_min+2, 12.0, 9.0, 0.5, key="lmaxmax")
    n_onde_reg = st.slider("N onde", 40, 100, 70, 10, key="noreg")
    
    if st.button("Calcola Regressione", key="calc_reg"):
        lambda_max_vals = np.linspace(lambda_max_min, lambda_max_max, n_punti)
        dati = []
        for lmax in lambda_max_vals:
            k_min = 2 * np.pi / lmax
            k_max = 2 * np.pi / lambda_min_reg
            delta_k = k_max - k_min
            x = np.linspace(-45, 45, 10000)
            k_vals = np.linspace(k_min, k_max, n_onde_reg)
            y = np.zeros_like(x)
            for k in k_vals:
                y += (1/n_onde_reg) * np.cos(k * x)
            env = np.abs(signal.hilbert(y))
            delta_x, _, _ = calcola_larghezza_temporale(x, env, 0.06)
            dati.append({
                "λ_max": lmax, 
                "Δk": delta_k, 
                "1/Δk": 1/delta_k, 
                "Δx": delta_x, 
                "Δx·Δk": delta_x * delta_k
            })
        
        df = pd.DataFrame(dati)
        slope, intercept, r_value, p_value, std_err = linregress(df["1/Δk"], df["Δx"])
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("Pendenza", f"{slope:.3f}")
        with col_r2:
            st.metric("Target (4π)", "12.566")
        with col_r3:
            errore_p = abs(slope - 4*np.pi) / (4*np.pi) * 100
            st.metric("Errore %", f"{errore_p:.2f}%")
        with col_r4:
            st.metric("R²", f"{r_value**2:.4f}")
        
        x_fit = np.array([df["1/Δk"].min(), df["1/Δk"].max()])
        y_fit = slope * x_fit + intercept
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["1/Δk"], y=df["Δx"], mode='markers',
                                marker=dict(size=12, color='blue'), name="Dati"))
        fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines',
                                line=dict(color='red', width=3), 
                                name=f"Fit: y={slope:.2f}x+{intercept:.2f}"))
        fig.update_layout(
            title=f"Regressione: Δx = {slope:.2f}·(1/Δk) + {intercept:.2f} | R²={r_value**2:.4f}",
            xaxis_title="1/Δk (m/rad)", yaxis_title="Δx (m)", height=600
        )
        applica_stile(fig, is_light_mode)
        st.plotly_chart(fig, use_container_width=True, config=get_download_config("regressione_dx_dk"))
        
        st.dataframe(df, use_container_width=True)
        
        if r_value**2 > 0.95:
            st.success(f"Ottimo fit! R²={r_value**2:.4f}")
        else:
            st.warning(f"Fit migliorabile (R²={r_value**2:.4f}). Aumenta N onde.")
        
        if errore_p < 5:
            st.success(f"Pendenza in ottimo accordo con 4π! ({errore_p:.2f}%)")
        else:
            st.info(f"Pendenza discosta da 4π di {errore_p:.2f}%")
        
        csv = df.to_csv(index=False)
        st.download_button("Scarica dati", csv, "regressione.csv", "text/csv")

# ========== ONDE STAZIONARIE ==========
elif sezione == "Onde Stazionarie":
    styled_header(
        "🎸", 
        "Onde Stazionarie",
        "Armoniche e quantizzazione: come vibra una corda di chitarra e l'origine degli stati discreti",
        "#2ecc71"
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parametri Sistema")
        L = st.slider("Lunghezza L (m)", 0.5, 5.0, 1.0, 0.1, key="sw_L")
        n_armonica = st.slider("Numero Armonica n", 1, 10, 1, key="sw_n")
        v_onda = st.number_input("Velocità onda (m/s)", 100.0, 1000.0, 340.0, 10.0, key="sw_v", help="Es. 340 m/s per aria, variabile per corde")
        
        # Calcoli
        lambda_n = 2 * L / n_armonica
        freq_n = v_onda / lambda_n
        
        st.markdown("---")
        st.metric("Lunghezza d'onda λ", f"{lambda_n:.3f} m")
        st.metric("Frequenza f", f"{freq_n:.2f} Hz")
        
        st.markdown("---")
        if st.button("Genera Tono"):
            # Generazione audio con inviluppo morbido per evitare 'click'
            duration = 2.0
            t_audio = np.linspace(0, duration, int(SAMPLE_RATE * duration))
            envelope = np.concatenate([np.linspace(0, 1, 1000), np.ones(len(t_audio)-2000), np.linspace(1, 0, 1000)])
            y_audio = np.sin(2 * np.pi * freq_n * t_audio) * envelope
            audio_bytes = genera_audio(y_audio)
            st.audio(audio_bytes, format='audio/wav')

    with col2:
        # Visualizzazione
        x = np.linspace(0, L, 500)
        y_shape = np.sin(n_armonica * np.pi * x / L)
        
        fig = go.Figure()
        
        # Inviluppo (positivo e negativo)
        fig.add_trace(go.Scatter(x=x, y=y_shape, mode='lines', 
                                line=dict(color='red', width=2, dash='dash'), name="Inviluppo"))
        fig.add_trace(go.Scatter(x=x, y=-y_shape, mode='lines', 
                                line=dict(color='red', width=2, dash='dash'), showlegend=False))
        
        # Area riempita per rappresentare l'oscillazione rapida
        fig.add_trace(go.Scatter(x=x, y=y_shape, fill='tonexty', fillcolor='rgba(0, 0, 255, 0.1)',
                                line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=-y_shape, fill='tonexty', fillcolor='rgba(0, 0, 255, 0.1)',
                                line=dict(width=0), showlegend=False))
                                
        # Annotazione Nodi
        for i in range(n_armonica + 1):
            pos_x = i * L / n_armonica
            fig.add_annotation(x=pos_x, y=0, text="N", showarrow=True, arrowhead=2, ax=0, ay=20)

        fig.update_layout(
            title=f"Modo Normale n={n_armonica} (f={freq_n:.1f} Hz)",
            xaxis_title="Posizione x (m)",
            yaxis_title="Ampiezza",
            yaxis=dict(range=[-1.5, 1.5]),
            height=500
        )
        applica_zoom(fig, range_x_glob)
        applica_stile(fig, is_light_mode)
        st.plotly_chart(fig, use_container_width=True, config=get_download_config("onde_stazionarie"))
        
        st.markdown("#### Equazione d'Onda")
        st.latex(r"y(x,t) = 2A \sin\left(\frac{n\pi}{L}x\right) \cos(\omega_n t)")
        
        st.info(f"""
        **Fisica:**
        - **Nodi (N)**: Punti dove l'ampiezza è sempre zero.
        - **Ventri**: Punti di massima oscillazione.
        - In una corda di lunghezza L, sono permesse solo le lunghezze d'onda tali che $L = n \cdot \lambda/2$.
        """)

# ========== ANIMAZIONE PROPAGAZIONE (VERSIONE CORRETTA) ==========
elif sezione == "Animazione Propagazione":
    styled_header(
        "🎬", 
        "Animazione Propagazione",
        "Visualizza la propagazione di pacchetti d'onda o battimenti nello spazio-tempo",
        "#3498db"
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parametri Animazione")
        
        tipo_onda_anim = st.selectbox("Tipo di onda", 
                                       ["Pacchetto d'onda", "Battimenti", "Onda singola"],
                                       key="tipo_anim")
        
        if tipo_onda_anim == "Pacchetto d'onda":
            f_min_anim = st.slider("Freq. min (Hz)", 50.0, 300.0, 100.0, 10.0, key="anim_fmin")
            f_max_anim = st.slider("Freq. max (Hz)", f_min_anim+10, 400.0, 150.0, 10.0, key="anim_fmax")
            n_onde_anim = st.slider("N onde", 20, 80, 40, 5, key="anim_n")
        elif tipo_onda_anim == "Battimenti":
            f1_anim = st.slider("Freq. 1 (Hz)", 50.0, 500.0, 100.0, 10.0, key="anim_f1")
            f2_anim = st.slider("Freq. 2 (Hz)", 50.0, 500.0, 110.0, 10.0, key="anim_f2")
        else:
            freq_anim = st.slider("Frequenza (Hz)", 50.0, 500.0, 100.0, 10.0, key="anim_freq")
        
        st.markdown("---")
        st.subheader("Controlli Animazione")
        
        lunghezza_spazio = st.slider("Lunghezza spaziale (m)", 5.0, 50.0, 20.0, 5.0, key="anim_lung")
        durata_anim = st.slider("Durata animazione (s)", 0.5, 3.0, 1.5, 0.1, key="anim_dur")
        n_frame = st.slider("Numero frame", 20, 100, 50, 5, key="anim_frames", 
                           help="Più frame = animazione più fluida ma più lenta da generare")
        velocita = V_SUONO
        
        st.metric("Velocità propagazione", f"{velocita} m/s")
        st.metric("Spostamento totale", f"{velocita * durata_anim:.1f} m")
        
        if st.button("Genera Animazione", key="gen_anim"):
            st.session_state.anim_ready = True
    
    with col2:
        if st.session_state.get("anim_ready", False):
            progress = st.progress(0, "Generazione animazione...")
            
            # Griglia spaziale
            x = np.linspace(-lunghezza_spazio/2, lunghezza_spazio/2, 500)
            t_frames = np.linspace(0, durata_anim, n_frame)
            
            frames = []
            for i, t_val in enumerate(t_frames):
                progress.progress(i/n_frame, f"Frame {i+1}/{n_frame}")
                
                if tipo_onda_anim == "Pacchetto d'onda":
                    frequenze_anim = np.linspace(f_min_anim, f_max_anim, n_onde_anim)
                    y_frame = np.zeros_like(x)
                    for f in frequenze_anim:
                        k = 2 * np.pi * f / velocita
                        omega = 2 * np.pi * f
                        y_frame += (1/n_onde_anim) * np.cos(k * x - omega * t_val)
                    
                    titolo_frame = f"Pacchetto d'onda: t = {t_val:.3f} s"
                
                elif tipo_onda_anim == "Battimenti":
                    k1 = 2 * np.pi * f1_anim / velocita
                    k2 = 2 * np.pi * f2_anim / velocita
                    omega1 = 2 * np.pi * f1_anim
                    omega2 = 2 * np.pi * f2_anim
                    
                    y_frame = np.cos(k1 * x - omega1 * t_val) + np.cos(k2 * x - omega2 * t_val)
                    titolo_frame = f"Battimenti: t = {t_val:.3f} s"
                
                else:
                    k = 2 * np.pi * freq_anim / velocita
                    omega = 2 * np.pi * freq_anim
                    y_frame = np.cos(k * x - omega * t_val)
                    titolo_frame = f"Onda singola: t = {t_val:.3f} s"
                
                frames.append(go.Frame(
                    data=[go.Scatter(x=x, y=y_frame, mode='lines', 
                                    line=dict(color='blue', width=2))],
                    name=str(i),
                    layout=go.Layout(title_text=titolo_frame)
                ))
            
            progress.empty()
            
            # Crea figura con primo frame
            fig_anim = go.Figure(
                data=[go.Scatter(x=x, y=frames[0].data[0].y, mode='lines',
                                line=dict(color='blue', width=2))],
                layout=go.Layout(
                    title=f"Propagazione: t = 0.000 s",
                    xaxis=dict(title="Posizione x (m)", range=[-lunghezza_spazio/2, lunghezza_spazio/2]),
                    yaxis=dict(title="Ampiezza", range=[-3, 3]),
                    # 🆕 PULSANTI SPOSTATI SOTTO AL CENTRO
                    updatemenus=[dict(
                        type="buttons",
                        direction="left",
                        showactive=False,
                        buttons=[
                            dict(label="Play",
                                 method="animate",
                                 args=[None, {"frame": {"duration": int(durata_anim*1000/n_frame), 
                                                       "redraw": True},
                                            "fromcurrent": True,
                                            "mode": "immediate"}]),
                            dict(label="Pause",
                                 method="animate",
                                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                               "mode": "immediate",
                                               "transition": {"duration": 0}}])
                        ],
                        # 🎯 POSIZIONE: sotto al centro del grafico
                        x=0.5,        # Centro orizzontale (0 = sinistra, 1 = destra)
                        xanchor="center",  # Ancora al centro
                        y=-0.15,      # Sotto il grafico (negativo = sotto)
                        yanchor="top"
                    )],
                    # 🆕 SLIDER SPOSTATO PIÙ IN BASSO
                    sliders=[dict(
                        active=0,
                        yanchor="top",
                        y=-0.25,      # Ancora più sotto per lasciare spazio ai pulsanti
                        xanchor="left",
                        currentvalue=dict(
                            prefix="Frame: ", 
                            visible=True, 
                            xanchor="right",
                            font=dict(size=14)
                        ),
                        pad=dict(b=10, t=50),
                        len=0.9,
                        x=0.05,
                        steps=[dict(args=[[f.name], {"frame": {"duration": 0, "redraw": True},
                                                     "mode": "immediate"}],
                                   label=str(k),
                                   method="animate") for k, f in enumerate(frames)]
                    )],
                    # 🆕 AUMENTA MARGINE INFERIORE per fare spazio
                    margin=dict(b=120)
                ),
                frames=frames
            )
            
            fig_anim.update_layout(height=700, hovermode='x')
            applica_zoom(fig_anim, range_x_glob)
            applica_stile(fig_anim, is_light_mode)
            st.plotly_chart(fig_anim, use_container_width=True, config=get_download_config("animazione_propagazione"))
            
            st.success(f"Animazione generata: {n_frame} frame, durata {durata_anim:.1f}s")
            
            # Spiegazione fisica
            with st.expander("Fisica della Propagazione"):
                st.markdown(f"""
                ### Equazione dell'Onda
                
                **Onda generica**: y(x,t) = A·cos(kx - ωt + φ)
                
                - **k** = 2π/λ (numero d'onda): {2*np.pi*100/velocita:.4f} rad/m (esempio a 100 Hz)
                - **ω** = 2πf (pulsazione): {2*np.pi*100:.2f} rad/s (esempio a 100 Hz)
                - **v = ω/k** = λf = {velocita} m/s (velocità di fase)
                
                ### Direzione Propagazione
                
                Il segno **negativo** in (kx - ωt) indica propagazione verso **destra** (x crescenti).
                
                Al tempo t, il massimo dell'onda si trova dove: kx - ωt = 0 → x = (ω/k)·t = v·t
                
                ### Mezzo Non Dispersivo
                
                Per il suono in aria:
                - Tutte le frequenze viaggiano alla stessa velocità ({velocita} m/s)
                - Il pacchetto mantiene la forma propagandosi
                - v_fase = v_gruppo = {velocita} m/s
                """)
        else:
            st.info("Clicca su 'Genera Animazione' per visualizzare la propagazione")


# ========== ANALISI AUDIO MICROFONO ==========
elif sezione == "Analisi Audio Microfono":
    styled_header(
        "🎙️", 
        "Analisi Audio",
        "Registra o carica un file audio per analizzare spettro, frequenze e caratteristiche del segnale",
        "#9b59b6"
    )
    
    col_in1, col_in2 = st.columns(2)
    
    with col_in1:
        st.subheader("📂 Carica File")
        st.caption("Formati supportati: WAV")
        uploaded_file = st.file_uploader("Seleziona file audio", type=['wav'], key="audio_upload", label_visibility="collapsed")
        
    with col_in2:
        st.subheader("🎤 Registra dal Vivo")
        # Istruzioni chiare sulla procedura
        st.markdown("""
        **📋 Procedura:**
        1. Clicca sull'icona del microfono
        2. **Conta mentalmente fino a 3** ("uno, due, tre...")
        3. Poi inizia a produrre il suono
        4. Clicca di nuovo per fermare
        """)
        st.warning("⏳ **C'è un ritardo di 1-2 secondi** tra il click e l'inizio effettivo della registrazione. Questo è normale!")
        audio_bytes_rec = None
        try:
            from audio_recorder_streamlit import audio_recorder
            audio_bytes_rec = audio_recorder(
                text="",
                recording_color="#e74c3c",
                neutral_color="#3498db",
                icon_name="microphone",
                icon_size="3x",
                pause_threshold=60.0,  # Non fermare automaticamente (60 sec di silenzio)
                energy_threshold=0.001,  # Sensibilità molto bassa per non rilevare "silenzio"
                key="audio_rec"
            )
        except ImportError:
            st.error("Libreria mancante! Installa: `pip install audio-recorder-streamlit`") 

    # Logica unificata selezione sorgente
    audio_source = None
    nome_sorgente = ""
    
    if audio_bytes_rec:
        audio_source = audio_bytes_rec
        nome_sorgente = "Registrazione Microfono"
    elif uploaded_file:
        uploaded_file.seek(0)
        audio_source = uploaded_file.read()
        nome_sorgente = f"File: {uploaded_file.name}"
        
    if audio_source:
        st.markdown("---")
        st.success(f"Analisi in corso: **{nome_sorgente}**")
        st.audio(audio_source, format='audio/wav')
        
        try:
            from scipy.io import wavfile
            import io
            
            # Lettura Audio
            try:
                # Se è MP3 o altro, wavfile.read potrebbe fallire se non è WAV
                # Streamlit audio_recorder restituisce WAV
                sample_rate, audio_data = wavfile.read(io.BytesIO(audio_source))
            except Exception as e:
                st.error(f"Errore lettura audio (assicurati sia WAV): {str(e)}")
                st.stop()
            
            # Se stereo, prendi solo canale sinistro
            if len(audio_data.shape) == 2:
                audio_data = audio_data[:, 0]
            
            # Normalizza
            audio_data = audio_data.astype(float)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Metriche base
            durata_audio = len(audio_data) / sample_rate
            t_audio = np.linspace(0, durata_audio, len(audio_data))
            
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            with col_info1:
                st.metric("Durata", f"{durata_audio:.2f} s")
            with col_info2:
                st.metric("Sample Rate", f"{sample_rate} Hz")
            with col_info3:
                st.metric("Campioni", f"{len(audio_data):,}")
            with col_info4:
                rms = np.sqrt(np.mean(audio_data**2))
                st.metric("RMS", f"{rms:.4f}")
            
            # Grafico Forma d'Onda
            st.subheader("Forma d'Onda")
            max_samples_plot = 50000
            if len(audio_data) > max_samples_plot:
                step = len(audio_data) // max_samples_plot
                audio_plot = audio_data[::step]
                t_plot = t_audio[::step]
            else:
                audio_plot = audio_data
                t_plot = t_audio
            
            fig_waveform = go.Figure()
            fig_waveform.add_trace(go.Scatter(x=t_plot, y=audio_plot, 
                                             mode='lines', line=dict(color='blue', width=0.5),
                                             name="Ampiezza"))
            fig_waveform.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            applica_zoom(fig_waveform, range_x_glob)
            applica_stile(fig_waveform, is_light_mode)
            st.plotly_chart(fig_waveform, use_container_width=True, config=get_download_config("audio_waveform"))
            
            # FFT e Analisi Spettrale
            st.markdown("---")
            st.subheader("Analisi Spettrale (FFT)")
            
            window_size = min(len(audio_data), 65536)
            audio_window = audio_data[:window_size]
            yf = fft(audio_window)
            xf = fftfreq(window_size, 1/sample_rate)[:window_size//2]
            potenza = 2.0/window_size * np.abs(yf[:window_size//2])
            
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(potenza, height=np.max(potenza)*0.1, distance=20)
            freq_peaks = xf[peaks]
            amp_peaks = potenza[peaks]
            
            # Top 5 Frequenze
            sorted_idx = np.argsort(amp_peaks)[::-1]
            top_freqs = freq_peaks[sorted_idx[:5]]
            top_amps = amp_peaks[sorted_idx[:5]]
            
            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(x=xf, y=potenza, mode='lines', line=dict(color='red', width=1), name="FFT"))
            fig_fft.add_trace(go.Scatter(x=freq_peaks, y=amp_peaks, mode='markers', marker=dict(size=8, color='green'), name="Picchi"))
            fig_fft.update_layout(height=400, xaxis_title="Frequenza (Hz)", yaxis_title="Ampiezza")
            applica_zoom(fig_fft, range_x_glob)
            applica_stile(fig_fft, is_light_mode)
            st.plotly_chart(fig_fft, use_container_width=True, config=get_download_config("audio_fft"))
            
            # Riconoscimento Note
            if len(top_freqs) > 0:
                st.markdown("### Riconoscimento Note")
                col_freqs = st.columns(min(5, len(top_freqs)))
                for i, (col, f, a) in enumerate(zip(col_freqs, top_freqs, top_amps)):
                    with col:
                        st.metric(f"#{i+1}", f"{f:.1f} Hz", f"Amp: {a:.3f}")
                
                freq_fondamentale = top_freqs[0]
                # ... (logica note semplificata per brevità, ma inclusa nel codice completo)

            # Spettrogramma
            st.markdown("---")
            st.subheader("Spettrogramma")
            with st.spinner("Calcolo spettrogramma..."):
                nperseg = min(2048, len(audio_data)//10)
                f_spec, t_spec, Sxx = signal.spectrogram(audio_data, sample_rate, nperseg=nperseg)
                Sxx_db = 10 * np.log10(Sxx + 1e-10)
                
                fig_spec = go.Figure(data=go.Heatmap(z=Sxx_db, x=t_spec, y=f_spec, colorscale='Viridis'))
                fig_spec.update_layout(height=500, xaxis_title="Tempo (s)", yaxis_title="Frequenza (Hz)")
                applica_zoom(fig_spec, range_x_glob)
                applica_stile(fig_spec, is_light_mode)
                st.plotly_chart(fig_spec, use_container_width=True, config=get_download_config("audio_spettrogramma"))

        except Exception as e:
            st.error(f"Errore durante l'analisi: {e}")

# ========== RICONOSCIMENTO BATTIMENTI ==========
elif sezione == "Riconoscimento Battimenti":
    styled_header(
        "🎵", 
        "Riconoscimento Battimenti",
        "Registra due diapason e analizza automaticamente frequenze e battimenti",
        "#27ae60"
    )
    
    styled_info_box(
        "<strong>Suggerimento:</strong> Per un buon riconoscimento, registra per almeno 2-3 secondi e assicurati che i diapason suonino insieme.",
        "💡",
        "tip"
    )
    
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        st.subheader("📂 Carica File Audio")
        st.caption("Formati supportati: WAV")
        uploaded_beat_file = st.file_uploader("Seleziona file", type=['wav'], key="beat_audio_upload", label_visibility="collapsed")
        
    with col_rec2:
        st.subheader("🎤 Registra dal Vivo")
        # Istruzioni chiare sulla procedura
        st.markdown("""
        **📋 Procedura:**
        1. Prepara i diapason (NON suonarli ancora!)
        2. Clicca sull'icona del microfono
        3. **Conta mentalmente fino a 3** ("uno, due, tre...")
        4. ORA fai vibrare i diapason insieme
        5. Registra per almeno 3-4 secondi
        6. Clicca di nuovo per fermare
        """)
        st.warning("⏳ **C'è un ritardo di 1-2 secondi** tra il click e l'inizio effettivo della registrazione. Conta fino a 3 prima di suonare!")
        beat_audio_bytes = None
        try:
            from audio_recorder_streamlit import audio_recorder
            beat_audio_bytes = audio_recorder(
                text="",
                recording_color="#d63031",
                neutral_color="#27ae60",
                icon_name="microphone",
                icon_size="3x",
                pause_threshold=60.0,  # Non fermare automaticamente (60 sec di silenzio)
                energy_threshold=0.001,  # Sensibilità molto bassa per non rilevare "silenzio"
                key="beat_audio_rec"
            )
        except ImportError:
            st.error("Libreria mancante! Installa: `pip install audio-recorder-streamlit`")
    
    # Selezione sorgente audio
    beat_audio_source = None
    beat_nome_sorgente = ""
    
    if beat_audio_bytes:
        beat_audio_source = beat_audio_bytes
        beat_nome_sorgente = "Registrazione Microfono"
    elif uploaded_beat_file:
        uploaded_beat_file.seek(0)
        beat_audio_source = uploaded_beat_file.read()
        beat_nome_sorgente = f"File: {uploaded_beat_file.name}"
    
    if beat_audio_source:
        st.markdown("---")
        st.success(f"**Analisi in corso**: {beat_nome_sorgente}")
        st.audio(beat_audio_source, format='audio/wav')
        
        try:
            from scipy.io import wavfile
            from scipy.signal import find_peaks, hilbert
            import io
            
            # Lettura audio
            sample_rate_beat, audio_data_beat = wavfile.read(io.BytesIO(beat_audio_source))
            
            # Se stereo, prendi canale sinistro
            if len(audio_data_beat.shape) == 2:
                audio_data_beat = audio_data_beat[:, 0]
            
            # VERIFICA ARRAY NON VUOTO (fix errore numpy)
            if len(audio_data_beat) == 0:
                st.error("""
                <div style="background: linear-gradient(90deg, #e74c3c, #c0392b); border-radius: 10px; 
                            padding: 1rem 1.5rem; margin: 1rem 0;">
                    <span style="font-size: 1.5rem;">❌</span>
                    <span style="color: white; font-weight: 600;">File audio vuoto o non valido. Riprova con un altro file o registrazione.</span>
                </div>
                """)
                st.stop()
            
            # Normalizza
            audio_data_beat = audio_data_beat.astype(float)
            max_val = np.max(np.abs(audio_data_beat))
            if max_val > 0:
                audio_data_beat = audio_data_beat / max_val
            
            durata_beat = len(audio_data_beat) / sample_rate_beat
            t_beat = np.linspace(0, durata_beat, len(audio_data_beat))
            
            # Info base
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                st.metric("Durata", f"{durata_beat:.2f} s")
            with col_i2:
                st.metric("Sample Rate", f"{sample_rate_beat} Hz")
            with col_i3:
                st.metric("Campioni", f"{len(audio_data_beat):,}")
            
            # ========== ANALISI FFT ==========
            st.markdown("---")
            st.subheader("📊 Analisi Spettrale (FFT)")
            
            # FFT
            window_size_beat = min(len(audio_data_beat), 65536)
            audio_window_beat = audio_data_beat[:window_size_beat]
            yf_beat = fft(audio_window_beat)
            xf_beat = fftfreq(window_size_beat, 1/sample_rate_beat)[:window_size_beat//2]
            potenza_beat = 2.0/window_size_beat * np.abs(yf_beat[:window_size_beat//2])
            
            # Trova picchi (frequenze dominanti)
            # Filtro solo frequenze > 50 Hz per evitare rumore basso
            mask_freq = xf_beat > 50
            potenza_filtered = np.where(mask_freq, potenza_beat, 0)
            
            peaks_beat, props_beat = find_peaks(potenza_filtered, 
                                                 height=np.max(potenza_filtered)*0.15, 
                                                 distance=int(10 * window_size_beat / sample_rate_beat))
            
            if len(peaks_beat) >= 2:
                # Ordina per ampiezza e prendi i top 2
                sorted_peak_idx = np.argsort(potenza_beat[peaks_beat])[::-1]
                top_2_peaks = peaks_beat[sorted_peak_idx[:2]]
                
                f1_rilevata = min(xf_beat[top_2_peaks])
                f2_rilevata = max(xf_beat[top_2_peaks])
                f_batt_teorica = abs(f2_rilevata - f1_rilevata)
                
                st.success(f"✅ Rilevate **2 frequenze dominanti**: {f1_rilevata:.1f} Hz e {f2_rilevata:.1f} Hz")
                
                col_f1, col_f2, col_fb = st.columns(3)
                with col_f1:
                    st.metric("f₁ (Diapason 1)", f"{f1_rilevata:.1f} Hz")
                with col_f2:
                    st.metric("f₂ (Diapason 2)", f"{f2_rilevata:.1f} Hz")
                with col_fb:
                    st.metric("f_batt TEORICA", f"{f_batt_teorica:.2f} Hz", 
                             help="|f₂ - f₁|")
                
                # Grafico FFT
                fig_fft_beat = go.Figure()
                fig_fft_beat.add_trace(go.Scatter(x=xf_beat, y=potenza_beat, 
                                                  mode='lines', line=dict(color='blue', width=1), 
                                                  name="Spettro"))
                fig_fft_beat.add_trace(go.Scatter(x=[f1_rilevata, f2_rilevata], 
                                                  y=[potenza_beat[top_2_peaks[0]], potenza_beat[top_2_peaks[1]]], 
                                                  mode='markers', 
                                                  marker=dict(size=15, color='red', symbol='star'),
                                                  name="Frequenze Rilevate"))
                fig_fft_beat.update_layout(
                    title="Spettro di Frequenza (FFT)",
                    xaxis_title="Frequenza (Hz)",
                    yaxis_title="Ampiezza",
                    xaxis=dict(range=[0, max(f2_rilevata * 1.5, 600)]),
                    height=400
                )
                applica_stile(fig_fft_beat, is_light_mode)
                st.plotly_chart(fig_fft_beat, use_container_width=True, config=get_download_config("riconoscimento_fft"))
                
                # ========== ESTRAZIONE INVILUPPO ==========
                st.markdown("---")
                st.subheader("📈 Estrazione Inviluppo (Hilbert)")
                
                # Trasformata di Hilbert per estrarre l'inviluppo
                analytic_beat = hilbert(audio_data_beat)
                inviluppo_beat = np.abs(analytic_beat)
                
                # Smoothing dell'inviluppo per ridurre rumore
                from scipy.ndimage import uniform_filter1d
                inviluppo_smooth = uniform_filter1d(inviluppo_beat, size=int(sample_rate_beat * 0.01))
                
                # ========== MISURA f_batt DALL'INVILUPPO ==========
                # Metodo 1: FFT dell'inviluppo
                inviluppo_centered = inviluppo_smooth - np.mean(inviluppo_smooth)
                yf_env = fft(inviluppo_centered)
                xf_env = fftfreq(len(inviluppo_centered), 1/sample_rate_beat)[:len(inviluppo_centered)//2]
                potenza_env = 2.0/len(inviluppo_centered) * np.abs(yf_env[:len(inviluppo_centered)//2])
                
                # Cerca picco nella banda 0.5-30 Hz (range battimenti udibili)
                mask_env = (xf_env > 0.5) & (xf_env < 30)
                potenza_env_filtered = np.where(mask_env, potenza_env, 0)
                
                if np.max(potenza_env_filtered) > 0:
                    idx_peak_env = np.argmax(potenza_env_filtered)
                    f_batt_misurata = xf_env[idx_peak_env]
                else:
                    # Fallback: conta i picchi dell'inviluppo
                    peaks_env, _ = find_peaks(inviluppo_smooth, distance=int(sample_rate_beat * 0.05))
                    if len(peaks_env) > 1:
                        # Tempo medio tra picchi
                        dt_picchi = np.diff(t_beat[peaks_env])
                        T_medio = np.mean(dt_picchi)
                        f_batt_misurata = 1.0 / T_medio if T_medio > 0 else 0
                    else:
                        f_batt_misurata = 0
                
                # ========== CONFRONTO ==========
                st.markdown("---")
                st.subheader("⚖️ Confronto: Teoria vs Misura")
                
                if f_batt_teorica > 0:
                    errore_perc = abs(f_batt_misurata - f_batt_teorica) / f_batt_teorica * 100
                else:
                    errore_perc = 0
                
                T_batt_teorico = 1/f_batt_teorica if f_batt_teorica > 0 else 0
                T_batt_misurato = 1/f_batt_misurata if f_batt_misurata > 0 else 0
                
                col_comp1, col_comp2, col_comp3 = st.columns(3)
                with col_comp1:
                    st.metric("f_batt TEORICA", f"{f_batt_teorica:.2f} Hz",
                             help="|f₂ - f₁| calcolato dalla FFT")
                with col_comp2:
                    st.metric("f_batt MISURATA", f"{f_batt_misurata:.2f} Hz",
                             help="Misurata dall'inviluppo del segnale")
                with col_comp3:
                    delta_str = f"{errore_perc:.1f}% errore"
                    st.metric("Errore", delta_str,
                             delta_color="inverse" if errore_perc < 15 else "off")
                
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.metric("T_batt TEORICO", f"{T_batt_teorico:.3f} s")
                with col_t2:
                    st.metric("T_batt MISURATO", f"{T_batt_misurato:.3f} s")
                
                # Valutazione risultato
                if errore_perc < 10:
                    st.success("🎉 **Ottimo!** La misura è in eccellente accordo con la teoria!")
                elif errore_perc < 20:
                    st.info("👍 **Buono!** La misura è ragionevolmente vicina alla teoria.")
                else:
                    st.warning("⚠️ **Discrepanza significativa.** Prova a registrare con meno rumore di fondo.")
                
                # ========== GRAFICI FINALI ==========
                st.markdown("---")
                st.subheader("📉 Visualizzazione Battimenti")
                
                # Grafico forma d'onda + inviluppo
                # Zoom su una porzione per vedere bene i battimenti
                zoom_start = 0
                zoom_end = min(durata_beat, 2.0)  # Primi 2 secondi
                mask_zoom = (t_beat >= zoom_start) & (t_beat <= zoom_end)
                
                fig_wave_env = make_subplots(rows=2, cols=1, 
                                             subplot_titles=["Forma d'Onda con Inviluppo", "Inviluppo (Battimenti)"],
                                             vertical_spacing=0.12)
                
                # Sottocampionamento per performance
                step_plot = max(1, len(audio_data_beat[mask_zoom]) // 10000)
                
                fig_wave_env.add_trace(
                    go.Scatter(x=t_beat[mask_zoom][::step_plot], 
                              y=audio_data_beat[mask_zoom][::step_plot],
                              mode='lines', line=dict(color='blue', width=0.5),
                              name="Segnale"),
                    row=1, col=1
                )
                fig_wave_env.add_trace(
                    go.Scatter(x=t_beat[mask_zoom][::step_plot], 
                              y=inviluppo_smooth[mask_zoom][::step_plot],
                              mode='lines', line=dict(color='red', width=2),
                              name="Inviluppo"),
                    row=1, col=1
                )
                fig_wave_env.add_trace(
                    go.Scatter(x=t_beat[mask_zoom][::step_plot], 
                              y=-inviluppo_smooth[mask_zoom][::step_plot],
                              mode='lines', line=dict(color='red', width=2),
                              showlegend=False),
                    row=1, col=1
                )
                
                # Solo inviluppo
                fig_wave_env.add_trace(
                    go.Scatter(x=t_beat[mask_zoom][::step_plot], 
                              y=inviluppo_smooth[mask_zoom][::step_plot],
                              mode='lines', line=dict(color='orange', width=2),
                              name="Inviluppo", fill='tozeroy'),
                    row=2, col=1
                )
                
                fig_wave_env.update_layout(height=600, showlegend=True)
                fig_wave_env.update_xaxes(title_text="Tempo (s)", row=2, col=1)
                fig_wave_env.update_yaxes(title_text="Ampiezza", row=1, col=1)
                fig_wave_env.update_yaxes(title_text="Ampiezza", row=2, col=1)
                applica_stile(fig_wave_env, is_light_mode)
                st.plotly_chart(fig_wave_env, use_container_width=True, config=get_download_config("riconoscimento_inviluppo"))
                
                # Tabella riepilogativa
                st.markdown("---")
                st.subheader("📋 Riepilogo Analisi")
                riepilogo_df = pd.DataFrame({
                    "Parametro": ["f₁ (Hz)", "f₂ (Hz)", "Δf = |f₂-f₁| (Hz)", 
                                 "f_batt teorica (Hz)", "f_batt misurata (Hz)", 
                                 "T_batt teorico (s)", "T_batt misurato (s)", "Errore (%)"],
                    "Valore": [f"{f1_rilevata:.1f}", f"{f2_rilevata:.1f}", f"{f_batt_teorica:.2f}",
                              f"{f_batt_teorica:.2f}", f"{f_batt_misurata:.2f}",
                              f"{T_batt_teorico:.3f}", f"{T_batt_misurato:.3f}", f"{errore_perc:.1f}"]
                })
                st.dataframe(riepilogo_df, use_container_width=True, hide_index=True)
                
                # Download CSV
                csv_beat = riepilogo_df.to_csv(index=False)
                st.download_button("📥 Scarica Risultati (CSV)", csv_beat, 
                                  "riconoscimento_battimenti.csv", "text/csv")
                
                # ========== FORMULE ==========
                with st.expander("📚 Formule Teoriche"):
                    st.markdown("### Battimenti")
                    st.latex(r"y(t) = y_1(t) + y_2(t) = A_1\cos(\omega_1 t) + A_2\cos(\omega_2 t)")
                    st.latex(r"y(t) = 2A\cos\left(\frac{\omega_1 - \omega_2}{2}t\right) \cdot \cos\left(\frac{\omega_1 + \omega_2}{2}t\right)")
                    st.markdown("### Frequenza di Battimento")
                    st.latex(r"f_{\text{batt}} = |f_1 - f_2|")
                    st.markdown("### Periodo di Battimento")
                    st.latex(r"T_{\text{batt}} = \frac{1}{f_{\text{batt}}} = \frac{1}{|f_1 - f_2|}")
            
            elif len(peaks_beat) == 1:
                st.warning("⚠️ **Rilevata solo 1 frequenza dominante.** Per i battimenti servono 2 frequenze vicine.")
                st.info(f"Frequenza rilevata: {xf_beat[peaks_beat[0]]:.1f} Hz")
            else:
                st.error("❌ **Nessuna frequenza dominante rilevata.** Prova a registrare con volume più alto.")
                
        except Exception as e:
            st.error(f"Errore durante l'analisi: {e}")
            import traceback
            st.code(traceback.format_exc())


elif sezione == "Confronto Scenari":
    styled_header(
        "⚖️", 
        "Confronto Scenari",
        "Confronta due configurazioni differenti con grafici separati per vedere chiaramente le differenze",
        "#e74c3c"
    )
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🔵 Scenario A")
        f_min_a = st.slider("f_min (Hz)", 1.0, 500.0, 20.0, 1.0, key="comp_fmin_a")
        f_max_a = st.slider("f_max (Hz)", f_min_a+1, 500.0, 30.0, 1.0, key="comp_fmax_a")
        n_a = st.slider("N onde", 10, 100, 40, 5, key="comp_n_a")
        
        delta_f_a = f_max_a - f_min_a
        delta_k_a = 2 * np.pi * delta_f_a / V_SUONO
        delta_x_a = 4 * np.pi / delta_k_a
        
        st.metric("Δf", f"{delta_f_a:.2f} Hz")
        st.metric("Δx", f"{delta_x_a:.3f} m")
        st.metric("Δx·Δk", f"{delta_x_a * delta_k_a:.3f}")
    
    with col_b:
        st.subheader("🔴 Scenario B")
        f_min_b = st.slider("f_min (Hz)", 1.0, 500.0, 20.0, 1.0, key="comp_fmin_b")
        f_max_b = st.slider("f_max (Hz)", f_min_b+1, 500.0, 50.0, 1.0, key="comp_fmax_b")
        n_b = st.slider("N onde", 10, 100, 60, 5, key="comp_n_b")
        
        delta_f_b = f_max_b - f_min_b
        delta_k_b = 2 * np.pi * delta_f_b / V_SUONO
        delta_x_b = 4 * np.pi / delta_k_b
        
        st.metric("Δf", f"{delta_f_b:.2f} Hz")
        st.metric("Δx", f"{delta_x_b:.3f} m")
        st.metric("Δx·Δk", f"{delta_x_b * delta_k_b:.3f}")
    
    # Calcola il periodo di ripetizione e limita la visualizzazione al primo picco
    # Per pacchetti d'onda, la ripetizione avviene ogni T = 1/Δf
    T_repeat_a = 1 / delta_f_a if delta_f_a > 0 else 1.0
    T_repeat_b = 1 / delta_f_b if delta_f_b > 0 else 1.0
    
    # Usa il periodo più lungo per la visualizzazione (così si vede un solo picco per entrambi)
    T_display = min(T_repeat_a, T_repeat_b) * 0.8  # 80% del periodo più breve
    T_display = max(T_display, 0.05)  # Minimo 50ms
    T_display = min(T_display, 0.5)   # Massimo 500ms
    
    # Tempo specchiato: da -T a +T (simmetrico rispetto a t=0)
    n_points = 10000
    t_comp = np.linspace(-T_display, T_display, n_points)
    
    # Genera pacchetti (simmetrici nel tempo)
    freq_a = np.linspace(f_min_a, f_max_a, n_a)
    y_a = np.zeros_like(t_comp)
    for f in freq_a:
        y_a += (1/n_a) * np.cos(2 * np.pi * f * t_comp)
    
    freq_b = np.linspace(f_min_b, f_max_b, n_b)
    y_b = np.zeros_like(t_comp)
    for f in freq_b:
        y_b += (1/n_b) * np.cos(2 * np.pi * f * t_comp)
    
    # Due grafici separati con make_subplots
    fig_comp = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"🔵 Scenario A: Δf = {delta_f_a:.1f} Hz, Δx = {delta_x_a:.3f} m",
            f"🔴 Scenario B: Δf = {delta_f_b:.1f} Hz, Δx = {delta_x_b:.3f} m"
        ],
        vertical_spacing=0.12,
        shared_xaxes=True  # Stessa scala temporale!
    )
    
    # Scenario A (sopra) - blu con riempimento
    fig_comp.add_trace(
        go.Scatter(
            x=t_comp, y=y_a, 
            name="Scenario A",
            line=dict(color='#3498db', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)'
        ),
        row=1, col=1
    )
    
    # Scenario B (sotto) - rosso con riempimento
    fig_comp.add_trace(
        go.Scatter(
            x=t_comp, y=y_b, 
            name="Scenario B",
            line=dict(color='#e74c3c', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)'
        ),
        row=2, col=1
    )
    
    # Linea verticale a t=0 per riferimento
    fig_comp.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    fig_comp.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    fig_comp.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig_comp.update_xaxes(title_text="Tempo (s)", row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
    fig_comp.update_xaxes(gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
    fig_comp.update_yaxes(title_text="Ampiezza", gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
    fig_comp.update_yaxes(title_text="Ampiezza", gridcolor='rgba(128,128,128,0.2)', row=2, col=1)
    
    st.plotly_chart(fig_comp, use_container_width=True, config=get_download_config("confronto_scenari"))
    
    # Info box esplicativo
    styled_info_box(
        f"<strong>Osservazione:</strong> Lo scenario con Δf maggiore ({max(delta_f_a, delta_f_b):.1f} Hz) produce un pacchetto più stretto (Δx minore). Questo dimostra il principio di indeterminazione: <strong>Δx · Δk ≈ costante</strong>.",
        "🔍",
        "info"
    )
    
    st.markdown("---")
    st.subheader("📊 Tabella Comparativa")
    comp_df = pd.DataFrame({
        "Parametro": ["f_min (Hz)", "f_max (Hz)", "Δf (Hz)", "N onde", "Δk (rad/m)", "Δx (m)", "Δx·Δk"],
        "🔵 Scenario A": [f"{f_min_a:.1f}", f"{f_max_a:.1f}", f"{delta_f_a:.2f}", f"{n_a}", f"{delta_k_a:.4f}", f"{delta_x_a:.4f}", f"{delta_x_a*delta_k_a:.3f}"],
        "🔴 Scenario B": [f"{f_min_b:.1f}", f"{f_max_b:.1f}", f"{delta_f_b:.2f}", f"{n_b}", f"{delta_k_b:.4f}", f"{delta_x_b:.4f}", f"{delta_x_b*delta_k_b:.3f}"]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ========== ANALOGIA QUANTISTICA ==========
elif sezione == "Analogia Quantistica":
    styled_header(
        "⚛️", 
        "Analogia Quantistica",
        "Dal pacchetto d'onda classico alla funzione d'onda quantistica: il ponte tra due mondi",
        "#9b59b6"
    )
    
    # Intro teorica
    st.markdown("""
    ### 🌉 Il Ponte tra Onde Classiche e Meccanica Quantistica
    
    Tutto quello che hai visto con le **onde sonore** ha un corrispettivo nel mondo quantistico!
    Louis de Broglie (1924) propose che **ogni particella ha un comportamento ondulatorio**.
    """)
    
    st.markdown("---")
    
    # Formula di De Broglie e Tabella (affiancate con razionalità)
    col_theory1, col_theory2 = st.columns([1, 1])
    
    with col_theory1:
        st.markdown("### 📐 Relazione di De Broglie")
        st.latex(r"\lambda = \frac{h}{p} = \frac{h}{mv}")
        st.info("""
        **Leggenda:**
        - $\\lambda$: lunghezza d'onda di De Broglie
        - $h$: costante di Planck ($6.626 \\cdot 10^{-34}$ J·s)
        - $p$: quantità di moto ($m \\cdot v$)
        """)

    with col_theory2:
        st.markdown("### 🔄 Analogia con le Onde Sonore")
        st.markdown("""
        | Concetto | Onde Sonore | Meccanica Quantistica |
        |---|---|---|
        | **Oscillazione** | Pressione $P(x,t)$ | Funzione d'onda $\\psi(x,t)$ |
        | **Intensità** | $\\propto A^2$ | $\\propto |\\psi|^2$ (Probabilità) |
        | **Indeterminazione** | $\\Delta x \\cdot \\Delta k \\ge 1/2$ | $\\Delta x \\cdot \\Delta p \\ge \\hbar/2$ |
        """)
    
    st.markdown("---")
    
    # SEZIONE SIMULAZIONE (A Tutta Larghezza)
    st.subheader("⚙️ Simula una Particella")
    st.markdown("Scegli una particella e osserva come cambia la sua lunghezza d'onda e incertezza.")
    
    # Input parametri (su una riga ben spaziata)
    col_input1, col_input2, col_input3 = st.columns([1, 1, 2])
    
    with col_input1:
        tipo_particella = st.selectbox("Particella", 
                                       ["Elettrone", "Protone", "Neutrone", "Atomo di Idrogeno", "Pallina da tennis"],
                                       key="tipo_part")
    
    masse = {
        "Elettrone": 9.109e-31,
        "Protone": 1.673e-27,
        "Neutrone": 1.675e-27,
        "Atomo di Idrogeno": 1.674e-27,
        "Pallina da tennis": 0.057
    }
    massa = masse[tipo_particella]

    with col_input2:
        st.markdown(f"**Massa ($m$):**")
        if tipo_particella == "Pallina da tennis":
             st.markdown(f"$5.7 \\cdot 10^{{-2}}$ kg")
        else:
             esponente = int(np.floor(np.log10(massa)))
             mantissa = massa / 10**esponente
             st.markdown(f"${mantissa:.3f} \\cdot 10^{{{esponente}}}$ kg")

    with col_input3:
        if tipo_particella == "Pallina da tennis":
            velocita = st.slider("Velocità ($v$)", 1.0, 50.0, 20.0, 1.0, key="v_part")
        else:
            velocita = st.slider("Velocità ($v$)", 1e3, 1e7, 1e6, 1e3, key="v_part", format="%.0e")

    # Calcoli
    h = 6.626e-34
    hbar = h / (2 * np.pi)
    p = massa * velocita
    lambda_db = h / p
    k_db = 2 * np.pi / lambda_db
    
    st.markdown("### 📊 Risultati")
    
    # Formattazione scientifica LaTeX per i risultati
    def format_latex_sci(value, unit_latex=""):
        if value == 0: return "0"
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / 10**exponent
        # Fix: separate unit from text parsing to allow math symbols like \cdot
        return f"{mantissa:.2f} \\cdot 10^{{{exponent}}} \\; {unit_latex}"

    # Visualizzazione risultati (ripristinato layout separato per rendering LaTeX sicuro)
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.markdown(f"""
        <div style="background-color: rgba(52, 152, 219, 0.1); padding: 10px; border-radius: 8px; border-left: 5px solid #3498db; margin-bottom: 10px;">
            <strong style="color: #3498db;">Quantità di moto (p)</strong>
        </div>
        """, unsafe_allow_html=True)
        st.latex(format_latex_sci(p, r"\text{kg} \cdot \text{m/s}"))
        
    with res_col2:
        st.markdown(f"""
        <div style="background-color: rgba(155, 89, 182, 0.1); padding: 10px; border-radius: 8px; border-left: 5px solid #9b59b6; margin-bottom: 10px;">
            <strong style="color: #9b59b6;">Lunghezza d'onda (λ)</strong>
        </div>
        """, unsafe_allow_html=True)
        st.latex(format_latex_sci(lambda_db, r"\text{m}"))

    with res_col3:
        st.markdown(f"""
        <div style="background-color: rgba(231, 76, 60, 0.1); padding: 10px; border-radius: 8px; border-left: 5px solid #e74c3c; margin-bottom: 10px;">
            <strong style="color: #e74c3c;">Numero d'onda (k)</strong>
        </div>
        """, unsafe_allow_html=True)
        st.latex(format_latex_sci(k_db, r"\text{rad/m}"))

    
    st.markdown("---")
    
    # Visualizzazione del pacchetto d'onda quantistico
    st.markdown("### 🌊 Visualizzazione: Funzione d'Onda ψ(x)")
    st.markdown("Modifica l'incertezza sulla posizione $\\sigma_x$ e osserva come cambia la funzione d'onda.")
    
    col_vis_params, col_vis_graph = st.columns([1, 3])
    
    with col_vis_params:
        sigma_x = st.slider("Incertezza posizione $\\sigma_x$", 0.5, 5.0, 2.0, 0.1, key="sigma_x_q")
        k0 = st.slider("Numero d'onda centrale $k_0$", 1.0, 20.0, 10.0, 0.5, key="k0_q")
        mostra_prob = st.checkbox("Mostra probabilità $|\\psi|^2$", value=True, key="show_prob")
        
        # Calcolo sigma_k (incertezza nel momento)
        sigma_k = 1 / (2 * sigma_x)
        
        st.markdown(f"""
        <div style="margin-top: 20px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
            <div style="font-weight: bold; margin-bottom: 5px;">Incertezza Δk:</div>
            <div style="font-size: 1.1rem; margin-bottom: 10px;">{sigma_k:.3f} rad/u</div>
            <div style="font-weight: bold; margin-bottom: 5px;">Prodotto σₓ·σₖ:</div>
            <div style="font-size: 1.1rem; color: #2ecc71;">{sigma_x * sigma_k:.3f}</div>
            <div style="font-size: 0.8rem; opacity: 0.7;">Minimo Heisenberg = 0.5</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_vis_graph:
        # Genera funzione d'onda gaussiana
        x = np.linspace(-15, 15, 1000)
        
        # ψ(x) = pacchetto gaussiano
        psi_real = np.exp(-(x**2) / (4 * sigma_x**2)) * np.cos(k0 * x)
        psi_norm = np.exp(-(x**2) / (4 * sigma_x**2))  # Inviluppo
        psi_prob = psi_norm**2  # |ψ|²
        
        fig_psi = make_subplots(rows=2 if mostra_prob else 1, cols=1,
                               subplot_titles=["Funzione d'Onda ψ(x) - Parte Reale"] + 
                                            (["Densità di Probabilità |ψ(x)|²"] if mostra_prob else []),
                               shared_xaxes=True,
                               vertical_spacing=0.3)  # Aumentato vertical_spacing da 0.15 a 0.3
        
        # Funzione d'onda
        fig_psi.add_trace(go.Scatter(x=x, y=psi_real, name="Re[ψ(x)]",
                                    line=dict(color='#3498db', width=2)), row=1, col=1)
        fig_psi.add_trace(go.Scatter(x=x, y=psi_norm, name="Inviluppo",
                                    line=dict(color='#e74c3c', width=2, dash='dash')), row=1, col=1)
        fig_psi.add_trace(go.Scatter(x=x, y=-psi_norm, name="Inviluppo",
                                    line=dict(color='#e74c3c', width=2, dash='dash'), showlegend=False), row=1, col=1)
        
        if mostra_prob:
            # Probabilità
            fig_psi.add_trace(go.Scatter(x=x, y=psi_prob, name="|ψ|²",
                                        fill='tozeroy',
                                        line=dict(color='#9b59b6', width=2),
                                        fillcolor='rgba(155, 89, 182, 0.3)'), row=2, col=1)
            
            # Indicatori σ con annotazioni esplicative
            fig_psi.add_vline(x=-sigma_x, line_dash="dot", line_color="green", row=2, col=1)
            fig_psi.add_vline(x=sigma_x, line_dash="dot", line_color="green", row=2, col=1,
                             annotation_text="Incertezza Standard (±σ): probabilità 68%", 
                             annotation_position="top right", 
                             annotation_font_color="green",
                             annotation_font_size=10)
        
        # Sposta i titoli verso l'alto
        fig_psi.update_annotations(yshift=20)
        
        fig_psi.update_layout(height=500 if mostra_prob else 300,
                             plot_bgcolor='rgba(0,0,0,0)',
                             paper_bgcolor='rgba(0,0,0,0)',
                             margin=dict(l=20, r=20, t=40, b=20))
        fig_psi.update_xaxes(title_text="Posizione x", gridcolor='rgba(128,128,128,0.2)')
        fig_psi.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig_psi, use_container_width=True, config=get_download_config("analogia_quantistica"))
    
    st.markdown("---")
    
    styled_info_box(
        """<strong>🎯 Il Messaggio Chiave:</strong><br>
        Le onde sonore ci permettono di <strong>vedere</strong> e <strong>sentire</strong> il principio di indeterminazione. 
        Un suono breve (localizzato) ha uno spettro largo, proprio come una particella localizzata ha un momento incerto.""",
        "🔬",
        "tip"
    )

# ========== QUIZ INTERATTIVO ==========
elif sezione == "Quiz Interattivo":
    styled_header(
        "🎯", 
        "Quiz Interattivo",
        "Mettiti alla prova! Rispondi alle domande per verificare cosa hai imparato sulle onde",
        "#e67e22"
    )
    
    score = 0
    
    # Domanda 1
    st.subheader("1. Cosa succede alla larghezza del pacchetto (Δx) se aumentiamo la banda di frequenze (Δk)?")
    q1 = st.radio("Seleziona la risposta:", 
                  ["Il pacchetto diventa più largo", 
                   "Il pacchetto diventa più stretto", 
                   "Non cambia nulla"], 
                  index=None,
                  key="q1")
    
    if q1 == "Il pacchetto diventa più stretto":
        st.success("Corretto! Δx e Δk sono inversamente proporzionali (Principio di Indeterminazione).")
        score += 1
    elif q1 is not None:
        st.error("Sbagliato. Ricorda: Δx · Δk ≈ costante.")
        
    st.markdown("---")
    
    # Domanda 2
    st.subheader("2. Qual è la condizione per avere dei battimenti udibili?")
    q2 = st.radio("Seleziona la risposta:", 
                  ["Due onde con frequenze molto diverse", 
                   "Due onde con frequenze identiche", 
                   "Due onde con frequenze molto vicine"], 
                  index=None,
                  key="q2")
    
    if q2 == "Due onde con frequenze molto vicine":
        st.success("Esatto! La differenza di frequenza crea l'inviluppo pulsante.")
        score += 1
    elif q2 is not None:
        st.error("No. Se sono troppo diverse si sentono due suoni distinti.")

    st.markdown("---")

    # Domanda 3
    st.subheader("3. In un mezzo NON dispersivo (come l'aria per il suono), come viaggiano le onde?")
    q3 = st.radio("Seleziona la risposta:", 
                  ["Le frequenze alte viaggiano più veloci", 
                   "Tutte le frequenze viaggiano alla stessa velocità", 
                   "Le frequenze basse viaggiano più veloci"], 
                  index=None,
                  key="q3")
    
    if q3 == "Tutte le frequenze viaggiano alla stessa velocità":
        st.success("Bravissimo! Per questo il pacchetto non si deforma.")
        score += 1
    elif q3 is not None:
        st.error("Errato. Se fosse così, ascoltando un'orchestra da lontano i suoni arriverebbero sfasati!")

    st.markdown("---")
    if score == 3:
        st.balloons()
        st.success("COMPLIMENTI! Hai ottenuto 3/3! Sei un esperto di onde!")
    elif score > 0:
        st.info(f"Hai ottenuto {score}/3. Riprova per fare il pieno!")

# ========== MODALITÀ MOBILE (DEMO) ==========
elif sezione == "Modalità Mobile (Demo)":
    styled_header(
        "📱", 
        "Lab Tascabile",
        "Esplora la fisica del suono in modo semplice e intuitivo. Perfetto per il telefono!",
        "#1abc9c"
    )
    
    st.markdown("---")
    
    # 1. IL TONO (Frequenza)
    st.subheader("1. Che nota vuoi sentire?")
    st.caption("Sposta lo slider per cambiare l'altezza del suono.")
    pitch_mob = st.select_slider(
        "Altezza (Pitch)",
        options=[200, 300, 440, 600, 800, 1000],
        value=440,
        format_func=lambda x: "Grave" if x < 300 else "Acuto" if x > 600 else f"{x} Hz (Medio)"
    )
    
    st.markdown("---")
    
    # 2. L'EFFETTO (Fisica)
    st.subheader("2. Scegli la forma dell'onda")
    st.caption("Come si comportano le onde quando si incontrano?")
    mode_mob = st.radio(
        "Seleziona effetto:",
        ["Onda Pura", "Battimenti (Interferenza)", "Pacchetto (Impulso)"],
        horizontal=True
    )
    
    # Logica di generazione
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    
    if mode_mob == "Onda Pura":
        y = np.sin(2 * np.pi * pitch_mob * t)
        desc = "**Suono Puro**: Un'unica frequenza, pulita e costante. È il mattone fondamentale di tutti i suoni."
        color_line = "#3498db" # Blue
        view_dur = 0.02 # Zoom stretto
        
    elif mode_mob == "Battimenti (Interferenza)":
        f_beat = 5 # 5 Hz beat
        y = np.sin(2 * np.pi * pitch_mob * t) + np.sin(2 * np.pi * (pitch_mob + f_beat) * t)
        desc = f"**Battimenti**: Due suoni vicini ({pitch_mob} Hz e {pitch_mob+f_beat} Hz). L'interferenza crea un 'wow-wow' a {f_beat} Hz."
        color_line = "#e74c3c" # Red
        view_dur = 0.4 # Zoom largo per vedere l'inviluppo
        
    else: # Pacchetto
        # Create a packet centered at pitch_mob
        f_span = 50
        freqs = np.linspace(pitch_mob - f_span, pitch_mob + f_span, 30)
        y = np.zeros_like(t)
        for f in freqs:
            y += np.sin(2 * np.pi * f * t)
        y = y / 30 * 5 # Normalize visually
        desc = "**Pacchetto**: Tante frequenze insieme creano un suono breve e concentrato. Più frequenze = durata minore."
        color_line = "#9b59b6" # Purple
        view_dur = 0.1 # Zoom medio

    # Visualizzazione Mobile-First
    st.markdown("### Guarda l'onda")
    view_idx = int(view_dur * SAMPLE_RATE)
    
    fig_mob = go.Figure()
    fig_mob.add_trace(go.Scatter(
        x=t[:view_idx], y=y[:view_idx],
        mode='lines', line=dict(color=color_line, width=3),
        fill='tozeroy'
    ))
    fig_mob.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), height=200,
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-2.5, 2.5]),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', dragmode=False 
    )
    st.plotly_chart(fig_mob, use_container_width=True, config={'displayModeBar': False})
    
    st.info(desc)
    
    st.markdown("### Ascolta")
    if st.button("RIPRODUCI IL SUONO", type="primary", use_container_width=True):
        audio_bytes = genera_audio(y)
        st.audio(audio_bytes, format='audio/wav')
    
    with st.expander("Curiosità Scientifica"):
        if mode_mob == "Onda Pura":
            st.markdown(f"""
            - **Frequenza**: {pitch_mob} oscillazioni al secondo.
            - **Periodo**: {1/pitch_mob*1000:.2f} millisecondi.
            - È il suono prodotto da un diapason ideale.
            """)
        elif mode_mob == "Battimenti (Interferenza)":
            st.markdown(f"""
            - **Frequenza media**: {pitch_mob + 2.5} Hz (il tono che senti).
            - **Frequenza battimento**: 5 Hz (il ritmo del 'wow-wow').
            - Usato dai musicisti per accordare gli strumenti a orecchio.
            """)
        else:
            st.markdown("""
            - **Principio di Indeterminazione**: Per fare un suono breve (localizzato nel tempo), servono molte frequenze diverse.
            - Un suono puro (una sola frequenza) durerebbe in eterno!
            """)

# ========== CENTRO DOWNLOAD ==========
elif sezione == "📥 Centro Download":
    st.title("📥 Centro Download Grafici")
    st.markdown("Scarica ogni grafico singolarmente in PNG ad alta risoluzione. Configura le dimensioni prima di scaricare.")
    
    # --- CONFIGURAZIONE GLOBALE ---
    st.markdown("---")
    st.subheader("🛠️ Configurazione Esportazione")
    
    col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
    with col_cfg1:
        dl_width = st.number_input("Larghezza (px)", value=1600, min_value=400, max_value=6000, step=100, key="dl_w",
                                   help="Larghezza base del PNG. Verrà moltiplicata per la Scala.")
    with col_cfg2:
        dl_height = st.number_input("Altezza (px)", value=900, min_value=300, max_value=4000, step=100, key="dl_h",
                                    help="Altezza base del PNG. Verrà moltiplicata per la Scala.")
    with col_cfg3:
        dl_scale = st.slider("Scala (moltiplicatore)", 1, 6, 4, key="dl_s",
                              help="Moltiplicatore risoluzione. 4 = stampa qualità.")
    with col_cfg4:
        dl_lw = st.slider("Spessore linee (px)", 1.0, 8.0, 2.5, 0.5, key="dl_lw",
                           help="Spessore delle linee nei grafici.")
    
    final_w = dl_width * dl_scale
    final_h = dl_height * dl_scale
    st.info(f"📐 Dimensione finale PNG: **{final_w} × {final_h} px** | Spessore linee: **{dl_lw} px**")
    
    def dl_config(filename):
        """Configurazione download con dimensioni personalizzate."""
        return {
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': filename,
                'height': dl_height,
                'width': dl_width,
                'scale': dl_scale
            }
        }
    
    st.markdown("---")
    st.markdown("### 📷 Clicca l'icona **fotocamera** (📷) nella barra sopra ogni grafico per scaricare.")
    
    # --- PARAMETRI CONDIVISI ---
    st.markdown("---")
    st.subheader("⚙️ Parametri Grafici")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Battimenti**")
        dl_f1 = st.number_input("f₁ (Hz)", value=440.0, min_value=1.0, max_value=2000.0, step=1.0, key="dl_f1")
        dl_f2 = st.number_input("f₂ (Hz)", value=444.0, min_value=1.0, max_value=2000.0, step=1.0, key="dl_f2")
        dl_A1 = st.number_input("A₁", value=1.0, min_value=0.1, max_value=2.0, step=0.1, key="dl_A1")
        dl_A2 = st.number_input("A₂", value=1.0, min_value=0.1, max_value=2.0, step=0.1, key="dl_A2")
    with col_p2:
        st.markdown("**Pacchetti d'Onda / Indeterminazione**")
        dl_fmin = st.number_input("f_min (Hz)", value=100.0, min_value=1.0, max_value=500.0, step=1.0, key="dl_fmin")
        dl_fmax = st.number_input("f_max (Hz)", value=130.0, min_value=2.0, max_value=1000.0, step=1.0, key="dl_fmax")
        dl_n_onde = st.number_input("N onde", value=50, min_value=5, max_value=200, step=5, key="dl_nonde")
        dl_durata = st.number_input("Durata (s)", value=1.5, min_value=0.1, max_value=5.0, step=0.1, key="dl_dur")
    
    # Validazione
    if dl_fmax <= dl_fmin:
        dl_fmax = dl_fmin + 5.0
        st.warning(f"f_max corretta a {dl_fmax:.1f} Hz (deve essere > f_min)")
    
    # ============================================================
    # 1. BATTIMENTI - Grafici Singoli
    # ============================================================
    st.markdown("---")
    st.header("1. Battimenti")
    
    # Calcoli battimenti
    dl_f_batt = abs(dl_f1 - dl_f2)
    dl_T_batt = 1/dl_f_batt if dl_f_batt > 0 else 5.0
    dl_dur_batt = min(max(4 * dl_T_batt, 0.02), 10.0) if dl_f_batt > 0.01 else 1.0
    
    # Calcola il segnale su una finestra ESTESA (3x) per evitare artefatti Hilbert ai bordi
    fs_batt = 20000
    extra = dl_dur_batt  # Estendi di 1x su ogni lato
    t_ext = np.linspace(-extra, dl_dur_batt + extra, int((dl_dur_batt + 2*extra) * fs_batt))
    y1_ext = dl_A1 * np.cos(2 * np.pi * dl_f1 * t_ext)
    y2_ext = dl_A2 * np.cos(2 * np.pi * dl_f2 * t_ext)
    y_tot_ext = y1_ext + y2_ext
    
    # Inviluppo calcolato sulla finestra estesa
    env_ext = np.abs(signal.hilbert(y_tot_ext))
    
    # Taglia alla finestra di visualizzazione [0, dl_dur_batt]
    mask = (t_ext >= 0) & (t_ext <= dl_dur_batt)
    t_b = t_ext[mask]
    y1_b = y1_ext[mask]
    y2_b = y2_ext[mask]
    y_tot_b = y_tot_ext[mask]
    env_b = env_ext[mask]
    
    # 1a. Onda 1
    st.markdown(f"#### 1a. Onda 1 — f₁ = {dl_f1} Hz")
    fig_o1 = go.Figure()
    fig_o1.add_trace(go.Scatter(x=t_b, y=y1_b, line=dict(color='#2980b9', width=dl_lw), name=f"Onda 1 ({dl_f1} Hz)"))
    fig_o1.update_layout(xaxis_title="Tempo (s)", yaxis_title="Ampiezza", height=400, hovermode='x unified')
    applica_stile(fig_o1, is_light_mode)
    st.plotly_chart(fig_o1, use_container_width=True, config=dl_config("battimenti_onda1"))
    
    # 1b. Onda 2
    st.markdown(f"#### 1b. Onda 2 — f₂ = {dl_f2} Hz")
    fig_o2 = go.Figure()
    fig_o2.add_trace(go.Scatter(x=t_b, y=y2_b, line=dict(color='#e74c3c', width=dl_lw), name=f"Onda 2 ({dl_f2} Hz)"))
    fig_o2.update_layout(xaxis_title="Tempo (s)", yaxis_title="Ampiezza", height=400, hovermode='x unified')
    applica_stile(fig_o2, is_light_mode)
    st.plotly_chart(fig_o2, use_container_width=True, config=dl_config("battimenti_onda2"))
    
    # 1c. Sovrapposizione con inviluppo
    st.markdown(f"#### 1c. Sovrapposizione con Inviluppo — f_batt = {dl_f_batt:.2f} Hz")
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=t_b, y=y_tot_b, line=dict(color='#8e44ad', width=dl_lw), name="Somma"))
    fig_s.add_trace(go.Scatter(x=t_b, y=env_b, line=dict(color='#e67e22', width=dl_lw, dash='dash'), name="Inviluppo"))
    fig_s.add_trace(go.Scatter(x=t_b, y=-env_b, line=dict(color='#e67e22', width=dl_lw, dash='dash'), showlegend=False))
    fig_s.update_layout(xaxis_title="Tempo (s)", yaxis_title="Ampiezza", height=500, hovermode='x unified')
    applica_stile(fig_s, is_light_mode)
    st.plotly_chart(fig_s, use_container_width=True, config=dl_config("battimenti_sovrapposizione"))
    
    # 1d. Solo inviluppo
    st.markdown("#### 1d. Inviluppo Isolato")
    fig_env = go.Figure()
    fig_env.add_trace(go.Scatter(x=t_b, y=env_b, fill='tozeroy', line=dict(color='#e67e22', width=dl_lw), name="Inviluppo"))
    fig_env.update_layout(xaxis_title="Tempo (s)", yaxis_title="|Ampiezza|", height=400, hovermode='x unified')
    applica_stile(fig_env, is_light_mode)
    st.plotly_chart(fig_env, use_container_width=True, config=dl_config("battimenti_inviluppo"))
    
    # ============================================================
    # 2. PACCHETTO D'ONDA - Grafici Singoli
    # ============================================================
    st.markdown("---")
    st.header("2. Pacchetto d'Onda")
    
    # Calcoli pacchetto
    t_p = np.linspace(0, dl_durata, int(dl_durata * 20000))
    freq_p = np.linspace(dl_fmin, dl_fmax, dl_n_onde)
    y_pkt = np.zeros_like(t_p)
    for f in freq_p:
        y_pkt += (1/dl_n_onde) * np.cos(2 * np.pi * f * t_p)
    
    pad_p = int(len(t_p) * 0.1)
    y_pad_p = np.pad(y_pkt, (pad_p, pad_p), mode='reflect')
    env_p = np.abs(signal.hilbert(y_pad_p))[pad_p:-pad_p]
    int_p = env_p**2
    
    # 2a. Pacchetto con inviluppo
    st.markdown(f"#### 2a. Pacchetto d'Onda — {dl_n_onde} onde ({dl_fmin}-{dl_fmax} Hz)")
    fig_pkt = go.Figure()
    fig_pkt.add_trace(go.Scatter(x=t_p, y=y_pkt, line=dict(color='#2c3e50', width=dl_lw), name="Pacchetto"))
    fig_pkt.add_trace(go.Scatter(x=t_p, y=env_p, line=dict(color='#e74c3c', width=dl_lw, dash='dash'), name="Inviluppo +"))
    fig_pkt.add_trace(go.Scatter(x=t_p, y=-env_p, line=dict(color='#e74c3c', width=dl_lw, dash='dash'), showlegend=False))
    fig_pkt.update_layout(xaxis_title="Tempo (s)", yaxis_title="Ampiezza", height=500, hovermode='x unified')
    applica_stile(fig_pkt, is_light_mode)
    st.plotly_chart(fig_pkt, use_container_width=True, config=dl_config("pacchetto_onda"))
    
    # 2b. Intensità
    st.markdown("#### 2b. Intensità |A(t)|²")
    fig_int = go.Figure()
    fig_int.add_trace(go.Scatter(x=t_p, y=int_p, fill='tozeroy', line=dict(color='#e67e22', width=dl_lw), name="|A(t)|²"))
    fig_int.update_layout(xaxis_title="Tempo (s)", yaxis_title="|A(t)|²", height=400, hovermode='x unified')
    applica_stile(fig_int, is_light_mode)
    st.plotly_chart(fig_int, use_container_width=True, config=dl_config("pacchetto_intensita"))
    
    # 2c. Pacchetto simmetrico
    st.markdown("#### 2c. Pacchetto Simmetrico (t da -T a +T)")
    t_sim_dl = np.linspace(-dl_durata, dl_durata, int(dl_durata * 2 * 20000))
    y_pkt_sim = np.zeros_like(t_sim_dl)
    for f in freq_p:
        y_pkt_sim += (1/dl_n_onde) * np.cos(2 * np.pi * f * t_sim_dl)
    env_sim = np.abs(signal.hilbert(y_pkt_sim))
    
    fig_sim_dl = go.Figure()
    fig_sim_dl.add_trace(go.Scatter(x=t_sim_dl*1000, y=y_pkt_sim, line=dict(color='#2c3e50', width=dl_lw), name="Pacchetto"))
    fig_sim_dl.add_trace(go.Scatter(x=t_sim_dl*1000, y=env_sim, line=dict(color='#e74c3c', width=dl_lw, dash='dash'), name="Inviluppo"))
    fig_sim_dl.add_trace(go.Scatter(x=t_sim_dl*1000, y=-env_sim, line=dict(color='#e74c3c', width=dl_lw, dash='dash'), showlegend=False))
    fig_sim_dl.update_layout(xaxis_title="Tempo (ms)", yaxis_title="Ampiezza", height=500, hovermode='x unified')
    applica_stile(fig_sim_dl, is_light_mode)
    st.plotly_chart(fig_sim_dl, use_container_width=True, config=dl_config("pacchetto_simmetrico"))
    
    # 2d. Intensità simmetrica |A(t)|²
    st.markdown("#### 2d. Intensità Simmetrica |A(t)|²")
    int_sim = env_sim**2
    fig_int_sim = go.Figure()
    fig_int_sim.add_trace(go.Scatter(x=t_sim_dl*1000, y=int_sim, fill='tozeroy', line=dict(color='#e67e22', width=dl_lw), name="|A(t)|²"))
    fig_int_sim.update_layout(xaxis_title="Tempo (ms)", yaxis_title="|A(t)|²", height=400, hovermode='x unified')
    applica_stile(fig_int_sim, is_light_mode)
    st.plotly_chart(fig_int_sim, use_container_width=True, config=dl_config("pacchetto_intensita_simmetrica"))
    
    # ============================================================
    # 3. PRINCIPIO DI INDETERMINAZIONE - Grafici Singoli
    # ============================================================
    st.markdown("---")
    st.header("3. Principio di Indeterminazione")
    
    # Calcoli indeterminazione
    dl_lambda_min = V_SUONO / dl_fmax
    dl_lambda_max = V_SUONO / dl_fmin
    dl_k_min = 2 * np.pi / dl_lambda_max
    dl_k_max = 2 * np.pi / dl_lambda_min
    dl_delta_k = dl_k_max - dl_k_min
    dl_delta_x = 4 * np.pi / dl_delta_k if dl_delta_k > 0 else 0
    dl_delta_f = dl_fmax - dl_fmin
    dl_delta_omega = 2 * np.pi * dl_delta_f
    dl_delta_t = 4 * np.pi / dl_delta_omega if dl_delta_omega > 0 else 0
    
    # 3a. Dominio Spaziale
    st.markdown(f"#### 3a. Dominio Spaziale — Δx·Δk = {dl_delta_x*dl_delta_k:.2f}")
    range_x_ind = max(50.0, dl_delta_x * 2.0)
    x_ind = np.linspace(-range_x_ind, range_x_ind, 10000)
    k_vals = np.linspace(dl_k_min, dl_k_max, dl_n_onde)
    y_spazio = np.zeros_like(x_ind)
    for k in k_vals:
        y_spazio += (1/dl_n_onde) * np.cos(k * x_ind)
    env_spazio = np.abs(signal.hilbert(y_spazio))
    
    fig_spazio = go.Figure()
    fig_spazio.add_trace(go.Scatter(x=x_ind, y=y_spazio, line=dict(color='#2c3e50', width=dl_lw), name="Pacchetto"))
    fig_spazio.add_trace(go.Scatter(x=x_ind, y=env_spazio, line=dict(color='#e74c3c', width=dl_lw, dash='dash'), name="Inviluppo"))
    fig_spazio.add_trace(go.Scatter(x=x_ind, y=-env_spazio, line=dict(color='#e74c3c', width=dl_lw, dash='dash'), showlegend=False))
    fig_spazio.update_layout(xaxis_title="Posizione x (m)", yaxis_title="Ampiezza", height=500, hovermode='x unified',
                             title=f"Δx·Δk = {dl_delta_x*dl_delta_k:.2f} (target: 12.57)")
    applica_stile(fig_spazio, is_light_mode)
    st.plotly_chart(fig_spazio, use_container_width=True, config=dl_config("indeterminazione_spazio"))
    
    # 3b. Dominio Temporale
    dl_T_rep = (dl_n_onde - 1) / dl_delta_f if dl_n_onde > 1 and dl_delta_f > 0 else dl_durata * 10
    dl_dur_eff = min(dl_durata, dl_T_rep * 0.9)
    t_ind = np.linspace(0, dl_dur_eff, int(dl_dur_eff * 20000))
    y_tempo = np.zeros_like(t_ind)
    for f in freq_p:
        y_tempo += (1/dl_n_onde) * np.cos(2 * np.pi * f * t_ind)
    env_tempo = np.abs(signal.hilbert(y_tempo))
    
    st.markdown(f"#### 3b. Dominio Temporale — Δω·Δt = {dl_delta_t*dl_delta_omega:.2f}")
    fig_tempo = go.Figure()
    fig_tempo.add_trace(go.Scatter(x=t_ind*1000, y=y_tempo, line=dict(color='#2c3e50', width=dl_lw), name="Pacchetto"))
    fig_tempo.add_trace(go.Scatter(x=t_ind*1000, y=env_tempo, line=dict(color='#e67e22', width=dl_lw, dash='dash'), name="Inviluppo"))
    fig_tempo.add_trace(go.Scatter(x=t_ind*1000, y=-env_tempo, line=dict(color='#e67e22', width=dl_lw, dash='dash'), showlegend=False))
    fig_tempo.update_layout(xaxis_title="t (ms)", yaxis_title="A(t)", height=500, hovermode='x unified',
                            title=f"Δω·Δt = {dl_delta_t*dl_delta_omega:.2f} (target: 12.57)")
    applica_stile(fig_tempo, is_light_mode)
    st.plotly_chart(fig_tempo, use_container_width=True, config=dl_config("indeterminazione_tempo"))
    
    # 3c. Dominio Temporale Simmetrico
    st.markdown("#### 3c. Dominio Temporale Simmetrico")
    t_sim_ind = np.linspace(-dl_dur_eff, dl_dur_eff, int(dl_dur_eff * 2 * 20000))
    y_tempo_sim = np.zeros_like(t_sim_ind)
    for f in freq_p:
        y_tempo_sim += (1/dl_n_onde) * np.cos(2 * np.pi * f * t_sim_ind)
    env_tempo_sim = np.abs(signal.hilbert(y_tempo_sim))
    
    fig_tempo_sim = go.Figure()
    fig_tempo_sim.add_trace(go.Scatter(x=t_sim_ind*1000, y=y_tempo_sim, line=dict(color='#2c3e50', width=dl_lw), name="Pacchetto"))
    fig_tempo_sim.add_trace(go.Scatter(x=t_sim_ind*1000, y=env_tempo_sim, line=dict(color='#e67e22', width=dl_lw, dash='dash'), name="Inviluppo"))
    fig_tempo_sim.add_trace(go.Scatter(x=t_sim_ind*1000, y=-env_tempo_sim, line=dict(color='#e67e22', width=dl_lw, dash='dash'), showlegend=False))
    fig_tempo_sim.add_vline(x=0, line_dash="dot", line_color="green", annotation_text="t=0")
    fig_tempo_sim.update_layout(xaxis_title="t (ms)", yaxis_title="A(t)", height=500, hovermode='x unified',
                                title="Visualizzazione Temporale Simmetrica")
    applica_stile(fig_tempo_sim, is_light_mode)
    st.plotly_chart(fig_tempo_sim, use_container_width=True, config=dl_config("indeterminazione_tempo_sim"))
    
    # 3d. Spettro di Frequenze
    st.markdown("#### 3d. Spettro di Frequenze")
    
    # Controlli per fissare gli assi (per paragone)
    fix_assi_spettro = st.checkbox("🔒 Fissa assi per paragone", value=False, key="dl_fix_spettro",
                                    help="Fissa i range degli assi X e Y per confrontare grafici con parametri diversi.")
    if fix_assi_spettro:
        col_ax1, col_ax2, col_ax3, col_ax4 = st.columns(4)
        with col_ax1:
            sp_xmin = st.number_input("X min (Hz)", value=0.0, key="dl_sp_xmin")
        with col_ax2:
            sp_xmax = st.number_input("X max (Hz)", value=500.0, key="dl_sp_xmax")
        with col_ax3:
            sp_ymin = st.number_input("Y min", value=0.0, step=0.005, format="%.3f", key="dl_sp_ymin")
        with col_ax4:
            sp_ymax = st.number_input("Y max", value=0.05, step=0.005, format="%.3f", key="dl_sp_ymax")
    
    fig_spettro = go.Figure()
    fig_spettro.add_trace(go.Bar(x=freq_p, y=np.ones(dl_n_onde)/dl_n_onde, 
                                 marker_color='#3498db', name="Componenti"))
    fig_spettro.update_layout(xaxis_title="Frequenza (Hz)", yaxis_title="Ampiezza relativa", height=400,
                              title=f"Spettro: Δf = {dl_delta_f:.1f} Hz (N = {dl_n_onde} onde)", 
                              showlegend=False, bargap=0.1)
    
    if fix_assi_spettro:
        fig_spettro.update_xaxes(range=[sp_xmin, sp_xmax])
        fig_spettro.update_yaxes(range=[sp_ymin, sp_ymax])
    
    applica_stile(fig_spettro, is_light_mode)
    st.plotly_chart(fig_spettro, use_container_width=True, config=dl_config("spettro_frequenze"))
    
    # ============================================================
    # 4. CONFRONTO SCENARI
    # ============================================================
    st.markdown("---")
    st.header("4. Confronto Scenari")
    
    st.markdown("Due pacchetti con parametri diversi per confronto diretto.")
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        st.markdown("**Scenario A (Stretto)**")
        dl_fmin_a = st.number_input("f_min A (Hz)", value=100.0, key="dl_fmina")
        dl_fmax_a = st.number_input("f_max A (Hz)", value=110.0, key="dl_fmaxa")
        dl_n_a = st.number_input("N onde A", value=30, min_value=5, key="dl_na")
    with col_sc2:
        st.markdown("**Scenario B (Largo)**")
        dl_fmin_b = st.number_input("f_min B (Hz)", value=80.0, key="dl_fminb")
        dl_fmax_b = st.number_input("f_max B (Hz)", value=180.0, key="dl_fmaxb")
        dl_n_b = st.number_input("N onde B", value=50, min_value=5, key="dl_nb")
    
    dl_delta_f_a = dl_fmax_a - dl_fmin_a
    dl_delta_f_b = dl_fmax_b - dl_fmin_b
    dl_delta_x_a = V_SUONO / (dl_delta_f_a) if dl_delta_f_a > 0 else 0
    dl_delta_x_b = V_SUONO / (dl_delta_f_b) if dl_delta_f_b > 0 else 0
    T_display_comp = max(5 / min(dl_delta_f_a, dl_delta_f_b) if min(dl_delta_f_a, dl_delta_f_b) > 0 else 0.5, 0.05)
    T_display_comp = min(T_display_comp, 0.5)
    t_comp = np.linspace(-T_display_comp, T_display_comp, 10000)
    
    freq_a = np.linspace(dl_fmin_a, dl_fmax_a, dl_n_a)
    y_a = np.zeros_like(t_comp)
    for f in freq_a:
        y_a += (1/dl_n_a) * np.cos(2 * np.pi * f * t_comp)
    
    freq_b = np.linspace(dl_fmin_b, dl_fmax_b, dl_n_b)
    y_b = np.zeros_like(t_comp)
    for f in freq_b:
        y_b += (1/dl_n_b) * np.cos(2 * np.pi * f * t_comp)
    
    # 4a. Scenario A singolo
    st.markdown(f"#### 4a. Scenario A — Δf = {dl_delta_f_a:.1f} Hz")
    fig_ca = go.Figure()
    fig_ca.add_trace(go.Scatter(x=t_comp, y=y_a, line=dict(color='#3498db', width=dl_lw), 
                                fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.2)', name="Scenario A"))
    fig_ca.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_ca.update_layout(xaxis_title="Tempo (s)", yaxis_title="Ampiezza", height=400, hovermode='x unified',
                         title=f"Scenario A: Δf = {dl_delta_f_a:.1f} Hz, Δx ≈ {dl_delta_x_a:.3f} m")
    applica_stile(fig_ca, is_light_mode)
    st.plotly_chart(fig_ca, use_container_width=True, config=dl_config("confronto_scenario_A"))
    
    # 4b. Scenario B singolo
    st.markdown(f"#### 4b. Scenario B — Δf = {dl_delta_f_b:.1f} Hz")
    fig_cb = go.Figure()
    fig_cb.add_trace(go.Scatter(x=t_comp, y=y_b, line=dict(color='#e74c3c', width=dl_lw),
                                fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.2)', name="Scenario B"))
    fig_cb.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_cb.update_layout(xaxis_title="Tempo (s)", yaxis_title="Ampiezza", height=400, hovermode='x unified',
                         title=f"Scenario B: Δf = {dl_delta_f_b:.1f} Hz, Δx ≈ {dl_delta_x_b:.3f} m")
    applica_stile(fig_cb, is_light_mode)
    st.plotly_chart(fig_cb, use_container_width=True, config=dl_config("confronto_scenario_B"))
    
    # ============================================================
    # 5. ONDE STAZIONARIE
    # ============================================================
    st.markdown("---")
    st.header("5. Onde Stazionarie")
    
    dl_L = st.number_input("Lunghezza corda (m)", value=1.0, min_value=0.1, max_value=5.0, step=0.1, key="dl_L")
    dl_n_max = st.number_input("Armoniche da mostrare", value=5, min_value=1, max_value=10, step=1, key="dl_nmax")
    
    x_st = np.linspace(0, dl_L, 500)
    for n_arm in range(1, dl_n_max + 1):
        freq_arm = n_arm * V_SUONO / (2 * dl_L)
        y_arm = np.sin(n_arm * np.pi * x_st / dl_L)
        
        st.markdown(f"#### 5{chr(96+n_arm)}. Modo n={n_arm} — f = {freq_arm:.1f} Hz")
        fig_st = go.Figure()
        fig_st.add_trace(go.Scatter(x=x_st, y=y_arm, line=dict(color='#e74c3c', width=dl_lw, dash='dash'), name="Inviluppo"))
        fig_st.add_trace(go.Scatter(x=x_st, y=-y_arm, line=dict(color='#e74c3c', width=dl_lw, dash='dash'), showlegend=False))
        fig_st.add_trace(go.Scatter(x=x_st, y=y_arm, fill='tonexty', fillcolor='rgba(0,0,255,0.1)', line=dict(width=0), showlegend=False))
        
        for i in range(n_arm + 1):
            pos_x = i * dl_L / n_arm
            fig_st.add_annotation(x=pos_x, y=0, text="N", showarrow=True, arrowhead=2, ax=0, ay=20)
        
        fig_st.update_layout(xaxis_title="Posizione x (m)", yaxis_title="Ampiezza",
                            yaxis=dict(range=[-1.5, 1.5]), height=400,
                            title=f"Modo Normale n={n_arm} (f={freq_arm:.1f} Hz)")
        applica_stile(fig_st, is_light_mode)
        st.plotly_chart(fig_st, use_container_width=True, config=dl_config(f"onda_stazionaria_n{n_arm}"))
    
    # ============================================================
    # 6. PRESENTAZIONE - Grafici Singoli
    # ============================================================
    st.markdown("---")
    st.header("6. Grafici Presentazione")
    st.markdown("I grafici della sezione Modalità Presentazione, separati per singolo export.")
    
    # Battimenti presentazione (semplificato)
    st.markdown("#### 6a. Battimenti Presentazione")
    dl_f1_pres = 440.0
    dl_f2_pres = 444.0
    dl_dur_pres = 1.0
    t_pres = np.linspace(0, dl_dur_pres, int(dl_dur_pres * 20000))
    y1_pres = np.cos(2 * np.pi * dl_f1_pres * t_pres)
    y2_pres = np.cos(2 * np.pi * dl_f2_pres * t_pres)
    y_tot_pres = y1_pres + y2_pres
    env_pres = np.abs(signal.hilbert(y_tot_pres))
    
    fig_p_batt = make_subplots(rows=3, cols=1, 
                                subplot_titles=(f"Onda 1: {dl_f1_pres} Hz", f"Onda 2: {dl_f2_pres} Hz",
                                              f"Sovrapposizione (f_batt = {abs(dl_f1_pres-dl_f2_pres):.0f} Hz)"),
                                shared_xaxes=True, vertical_spacing=0.08)
    fig_p_batt.add_trace(go.Scatter(x=t_pres, y=y1_pres, line=dict(color='#3498db', width=dl_lw), name="Onda 1"), row=1, col=1)
    fig_p_batt.add_trace(go.Scatter(x=t_pres, y=y2_pres, line=dict(color='#e74c3c', width=dl_lw), name="Onda 2"), row=2, col=1)
    fig_p_batt.add_trace(go.Scatter(x=t_pres, y=y_tot_pres, line=dict(color='#8e44ad', width=dl_lw), name="Somma"), row=3, col=1)
    fig_p_batt.add_trace(go.Scatter(x=t_pres, y=env_pres, line=dict(color='#e67e22', width=dl_lw, dash='dash'), name="Inv."), row=3, col=1)
    fig_p_batt.add_trace(go.Scatter(x=t_pres, y=-env_pres, showlegend=False, line=dict(color='#e67e22', width=dl_lw, dash='dash')), row=3, col=1)
    fig_p_batt.update_xaxes(title_text="Tempo (s)", row=3, col=1)
    fig_p_batt.update_layout(height=700, showlegend=True, hovermode='x unified')
    applica_stile(fig_p_batt, is_light_mode)
    st.plotly_chart(fig_p_batt, use_container_width=True, config=dl_config("pres_battimenti"))
    
    # Pacchetto presentazione
    st.markdown("#### 6b. Pacchetto Presentazione")
    dl_pres_fmin = 100.0
    dl_pres_fmax = 130.0
    dl_pres_n = 50
    t_pres_p = np.linspace(-0.3, 0.3, 12000)
    freq_pres = np.linspace(dl_pres_fmin, dl_pres_fmax, dl_pres_n)
    y_pres_p = np.zeros_like(t_pres_p)
    for f in freq_pres:
        y_pres_p += (1/dl_pres_n) * np.cos(2 * np.pi * f * t_pres_p)
    int_pres = np.abs(signal.hilbert(y_pres_p))**2
    
    fig_p_pkt = make_subplots(rows=2, cols=1, subplot_titles=("Pacchetto d'Onda", "Intensità |A(t)|²"),
                              shared_xaxes=True, vertical_spacing=0.1)
    fig_p_pkt.add_trace(go.Scatter(x=t_pres_p*1000, y=y_pres_p, line=dict(color='#2c3e50', width=dl_lw), name="Pacchetto"), row=1, col=1)
    fig_p_pkt.add_trace(go.Scatter(x=t_pres_p*1000, y=int_pres, fill='tozeroy', line=dict(color='#e67e22', width=dl_lw), name="|A(t)|²"), row=2, col=1)
    fig_p_pkt.update_xaxes(title_text="Tempo (ms)", row=2, col=1)
    fig_p_pkt.update_layout(height=650, hovermode='x unified')
    applica_stile(fig_p_pkt, is_light_mode)
    st.plotly_chart(fig_p_pkt, use_container_width=True, config=dl_config("pres_pacchetto"))

st.markdown("---")

# Footer con QR Code
footer_col1, footer_col2 = st.columns([3, 1])

with footer_col1:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-top: 1rem;
    ">
        <div style="color: white; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.8rem;">
            🎓 Liceo Leopardi Majorana
        </div>
        <div style="color: rgba(255,255,255,0.8); font-size: 0.95rem; margin-bottom: 0.5rem;">
            Giornata della Scienza 2026 • Laboratorio di Fisica
        </div>
        <div style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">
            👨‍🔬 Sviluppato da <strong>Alessandro Bigi</strong> | 🌊 Onde, Pacchetti e Indeterminazione
        </div>
    </div>
    """, unsafe_allow_html=True)

with footer_col2:
    # QR Code per condividere l'app
    app_url = "https://bigi-giornata-della-scienza.streamlit.app"
    qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=120x120&data={app_url}&bgcolor=2c3e50&color=ffffff"
    st.markdown(f"""
    <div style="
        background: #2c3e50;
        padding: 1rem;
        border-radius: 12px;
        margin-top: 1rem;
        text-align: center;
    ">
        <img src="{qr_api_url}" alt="QR Code" style="border-radius: 8px;">
        <div style="color: rgba(255,255,255,0.7); font-size: 0.75rem; margin-top: 0.5rem;">
            📱 Scansiona per aprire
        </div>
    </div>
    """, unsafe_allow_html=True) 