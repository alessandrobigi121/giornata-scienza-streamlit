# Risoluzione Problematiche - Giornata della Scienza

Documento che raccoglie le problematiche riscontrate e le relative soluzioni.

---

## 1. Ripetizione Periodica del Pacchetto d'Onda

### Problema
Nella sezione "Principio di Indeterminazione", i grafici temporali mostravano **picchi ripetuti** a intervalli regolari (es. ogni ~790 ms per il preset "Super-Localizzato").

![Esempio del problema](/Users/ale/.gemini/antigravity/brain/2db07837-688b-44ab-a7f7-1ab361c8ffe4/uploaded_image_1_1768153416717.png)

### Causa Fisica
Quando si sommano **N onde** con frequenze **equispaziate** tra f_min e f_max, il segnale risultante è **periodico** con periodo:

$$T_{ripetizione} = \frac{N - 1}{\Delta f} = \frac{N - 1}{f_{max} - f_{min}}$$

Questo è un fenomeno matematico della serie di Fourier discreta: frequenze commensurabili generano periodicità.

**Esempio (preset Super-Localizzato)**:
- f_min = 100 Hz, f_max = 200 Hz, N = 80
- T_rep = (80-1) / (200-100) = 79/100 = **0.79 s = 790 ms**

### Soluzione Implementata
Limitare automaticamente la **finestra temporale di visualizzazione** a meno dell'80% del periodo di ripetizione:

```python
# Calcolo periodo di ripetizione
delta_f = f_max - f_min
T_ripetizione = (n_onde - 1) / delta_f

# Limita durata visualizzata
durata_effettiva = min(durata, T_ripetizione * 0.8)
```

Inoltre, viene mostrato un avviso all'utente quando la durata viene limitata.

### File Modificato
- `app.py` (righe 1387-1420)

---

*Ultimo aggiornamento: 2026-01-11*
