import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

"""Usé el algoritmo de kleinberg porque el dataset cotiene fechas especificas de cada tweet, 
lo que permite ssaber los momentod de mas actividad de cada tema y saber si hay eventos externos que influencian, 
por ejemplo si el dia de mas actividad de frankenstei fue el dia de su estreno. 
ademas proporciona tendencias que se puede analizar para tomar deciciones"""

try:
    df = pd.read_csv('datasetTexto.csv', encoding='utf-8', on_bad_lines='skip')
except:
    try:
        df = pd.read_csv('datasetTexto.csv', encoding='utf-8', error_bad_lines=False)
    except:
        with open('datasetTexto.csv', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data = []
        for line in lines[1:]:
            parts = line.strip().split(',', 6)
            if len(parts) == 7:
                data.append(parts)
        
        df = pd.DataFrame(data, columns=['ID', 'Categoria', 'Titulo', 'Medio', 'Fecha', 'Resumen', 'Comentario_Reaccion'])

df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
df = df.dropna(subset=['ID'])
df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
df = df.dropna(subset=['Fecha'])

print(f"Dataset cargado con {len(df)} filas válidas")
print(f"Rango temporal: {df['Fecha'].min()} a {df['Fecha'].max()}")


def analisis_basico_series_temporales(df):
    """Análisis básico cuando Prophet no está disponible"""
    # Agregar por día
    daily = df.groupby(df['Fecha'].dt.date).size()
    
    print("\n=== ANÁLISIS BÁSICO SERIES TEMPORALES ===")
    print(f"Media diaria: {daily.mean():.2f} publicaciones/día")
    print(f"Máximo diario: {daily.max()} publicaciones")
    print(f"Día con más actividad: {daily.idxmax()}")
    
    # Calcular tendencia móvil
    daily_series = daily.reset_index()
    daily_series.columns = ['ds', 'y']
    daily_series['moving_avg'] = daily_series['y'].rolling(window=3, center=True).mean()
    
    print(f"Tendencia móvil (3 días): {daily_series['moving_avg'].iloc[-5:].tolist()}")
    
    return daily_series

# ====== DETECCIÓN DE BURSTS - ALGORITMO KLEINBERG ======
def kleinberg_burst_detection(events, s=2, gamma=0.5, smooth_win=3):
    """
    Implementación simplificada del algoritmo de Kleinberg para detección de bursts
    events: lista de conteos por intervalo de tiempo
    s: número de estados
    gamma: factor de costo
    """
    if len(events) < smooth_win:
        return []
    
    # Suavizar datos
    smoothed = pd.Series(events).rolling(window=smooth_win, center=True).mean().fillna(events[0])
    
    # Calcular umbrales para bursts
    mean_activity = np.mean(smoothed)
    std_activity = np.std(smoothed)
    
    bursts = []
    in_burst = False
    burst_start = 0
    
    for i, value in enumerate(smoothed):
        # Umbral para considerar burst (2 desviaciones estándar sobre la media)
        if value > mean_activity + 1.5 * std_activity:
            if not in_burst:
                in_burst = True
                burst_start = i
        else:
            if in_burst:
                in_burst = False
                bursts.append((burst_start, i-1, np.max(smoothed[burst_start:i])))
    
    # Si termina en burst
    if in_burst:
        bursts.append((burst_start, len(smoothed)-1, np.max(smoothed[burst_start:])))
    
    return bursts

# ====== EJECUTAR ANÁLISIS ======
def ejecutar_analisis_completo(df):
    # 1. Preparar datos temporales
    df_sorted = df.sort_values('Fecha')
    
    # Agrupar por día para análisis temporal
    daily_counts = df_sorted.groupby(df_sorted['Fecha'].dt.date).size()
    dates = list(daily_counts.index)
    counts = daily_counts.tolist()
    
    print("=== ANÁLISIS TEMPORAL COMPLETO ===")
    print(f"Total de días analizados: {len(daily_counts)}")
    print(f"Rango: {min(dates)} a {max(dates)}")
    
    # 2. Análisis Prophet
    
    # 3. Detección de Bursts (Kleinberg)
    print("\n=== DETECCIÓN DE BURSTS (Kleinberg) ===")
    bursts = kleinberg_burst_detection(counts)
    
    if bursts:
        print(f"Se detectaron {len(bursts)} períodos de alta actividad:")
        for i, (start, end, max_val) in enumerate(bursts):
            print(f"  Burst {i+1}: Días {start}-{end} (máx: {max_val:.1f} publicaciones)")
            if start < len(dates) and end < len(dates):
                print(f"    Fechas: {dates[start]} a {dates[end]}")
    else:
        print("No se detectaron bursts significativos")
    
    
    # 5. Análisis por categoría
    print("\n=== ANÁLISIS POR CATEGORÍA ===")
    for categoria in df['Categoria'].unique():
        cat_data = df[df['Categoria'] == categoria]
        cat_daily = cat_data.groupby(cat_data['Fecha'].dt.date).size()
        
        if len(cat_daily) > 0:
            cat_bursts = kleinberg_burst_detection(cat_daily.tolist())
            print(f"\n{categoria}:")
            print(f"  Total publicaciones: {len(cat_data)}")
            print(f"  Días con actividad: {len(cat_daily)}")
            print(f"  Bursts detectados: {len(cat_bursts)}")
            
            if cat_bursts:
                for burst in cat_bursts[:2]:  # Mostrar solo los 2 primeros
                    start_idx, end_idx, max_val = burst
                    if start_idx < len(cat_daily.index):
                        start_date = cat_daily.index[start_idx]
                        end_date = cat_daily.index[end_idx] if end_idx < len(cat_daily.index) else cat_daily.index[-1]
                        print(f"    - {start_date} a {end_date} (máx: {max_val:.1f})")

    # 6. Visualización
    print("\n=== VISUALIZACIÓN ===")
    plt.figure(figsize=(12, 8))
    
    # Serie temporal completa
    plt.subplot(2, 1, 1)
    plt.plot(dates, counts, 'b-', alpha=0.7, label='Publicaciones diarias')
    plt.title('Evolución Temporal de Publicaciones')
    plt.ylabel('Número de Publicaciones')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Marcar bursts
    for start, end, max_val in bursts:
        if start < len(dates) and end < len(dates):
            burst_dates = dates[start:end+1]
            burst_counts = counts[start:end+1]
            plt.fill_between(burst_dates, burst_counts, alpha=0.3, color='red', label='Bursts' if start == bursts[0][0] else "")
    
        
    # Análisis por categoría
    plt.subplot(2, 1, 2)
    for categoria in df['Categoria'].unique():
        cat_data = df[df['Categoria'] == categoria]
        cat_daily = cat_data.groupby(cat_data['Fecha'].dt.date).size()
        if len(cat_daily) > 0:
            plt.plot(cat_daily.index, cat_daily.values, marker='o', label=categoria, alpha=0.7)
    
    plt.title('Evolución por Categoría')
    plt.ylabel('Publicaciones')
    plt.xlabel('Fecha')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Ejecutar análisis completo
ejecutar_analisis_completo(df)

# Resumen ejecutivo
print("\n" + "="*50)
print("RESUMEN EJECUTIVO")
print("="*50)

daily_counts = df.groupby(df['Fecha'].dt.date).size()
print(f"• Período analizado: {len(daily_counts)} días")
print(f"• Publicaciones totales: {len(df)}")
print(f"• Tasa promedio: {len(df)/len(daily_counts):.1f} publicaciones/día")
print(f"• Categorías: {', '.join(df['Categoria'].unique())}")

# Recomendaciones basadas en el análisis
bursts = kleinberg_burst_detection(daily_counts.tolist())
if bursts:
    print(f"• Recomendación: Se detectaron {len(bursts)} períodos de alta actividad")
    print("  Considerar aumentar recursos durante estos períodos")
else:
    print("• Recomendación: Actividad estable, mantener monitoreo continuo")