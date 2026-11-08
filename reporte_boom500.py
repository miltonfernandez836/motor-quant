import asyncio
from deriv_datos import descargar_velas
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import requests
import os
import sys

# --- CONFIGURACIÓN V1.3 ---
ACTIVO = 'BOOM500'
INTERVALO = 900  # 15 Minutos
CANTIDAD = 5000 

# ==========================================
# ⚙️ TUS CREDENCIALES DE TELEGRAM
# ==========================================
TOKEN_TELEGRAM = os.getenv('TELEGRAM_TOKEN')
CHAT_ID_TELEGRAM = os.getenv('TELEGRAM_CHAT_ID')
TOKEN_DERIV = os.getenv('DERIV_TOKEN')

async def motor_principal():
    print("🚀 Iniciando Motor V1.3 (Calculando RSI y TODOS los sensores)...")

    if TOKEN_DERIV:
        print("🕵️ DIAGNÓSTICO: ✅ Token de Deriv detectado correctamente en el servidor.")

    try:
        print("📡 Descargando datos...")
        df = await descargar_velas(ACTIVO, INTERVALO, CANTIDAD)
        
        # --- NUEVO SENSOR: RSI (14) ---
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # --- SENSORES DE ESTRUCTURA E IMBALANCES ---
        n = 4  
        df['Pico'] = df['high'] == df['high'].rolling(window=(2*n)+1, center=True).max()
        df['Valle'] = df['low'] == df['low'].rolling(window=(2*n)+1, center=True).min()
        
        df['FVG_Alcista'] = df['low'] > df['high'].shift(2)
        df['FVG_Bajista'] = df['high'] < df['low'].shift(2)
        
        print("✅ MOTOR V1.3 LISTO. Matriz de datos completa.")
        return df

    except Exception as e:
        print(f"❌ Error descargando datos de Deriv: {e!r}")
        return None

# ==========================================
# 🧠 BLOQUE PRINCIPAL DEL PROGRAMA
# ==========================================
async def main():
    datos = await motor_principal()
    
    if datos is None:
        print("Error descargando datos. Saliendo...")
        sys.exit(1)

    print("⚖️ Ejecutando Juez V2.0 (Motor Híbrido SMC + IA 👁️)...")

    ultimo_pico_precio = 0.0
    indice_ultimo_pico = None
    ultimo_valle_precio = 0.0
    indice_ultimo_valle = None

    zonas_alta_probabilidad = []
    dataset_ml = [] 

    # --- Contadores COMPRAS ---
    w_base_c = 0; l_base_c = 0       
    w_fibo_c = 0; l_fibo_c = 0       
    w_rsi_c = 0;  l_rsi_c = 0        
    w_full_c = 0; l_full_c = 0       
    w_ia_c = 0;   l_ia_c = 0 

    # --- Contadores VENTAS ---
    w_base_v = 0; l_base_v = 0       
    w_fibo_v = 0; l_fibo_v = 0       
    w_rsi_v = 0;  l_rsi_v = 0        
    w_full_v = 0; l_full_v = 0       
    w_ia_v = 0;   l_ia_v = 0 

    for fecha, vela in datos.iterrows():
        
        # ==========================================
        # 🟢 LÓGICA DE COMPRAS (LONGS)
        # ==========================================
        if vela['Pico'] == True:
            ultimo_pico_precio = vela['high']
            indice_ultimo_pico = fecha
            
        elif ultimo_pico_precio > 0 and vela['close'] > ultimo_pico_precio:
            rango_impulso = datos.loc[indice_ultimo_pico : fecha]
            fecha_ob = rango_impulso['low'].idxmin()
            vela_ob = datos.loc[fecha_ob]
            
            if datos.loc[fecha_ob : fecha].head(5)['FVG_Alcista'].any():
                nivel_50 = vela_ob['low'] + ((ultimo_pico_precio - vela_ob['low']) * 0.5)
                es_discount = vela_ob['high'] <= nivel_50
                
                valles_previos = datos[datos['Valle'] == True].loc[:fecha_ob].tail(2)
                tiene_divergencia = False
                fecha_previo = None
                rsi_previo = vela_ob['RSI']
                
                if len(valles_previos) > 1:
                    valle_a = valles_previos.iloc[-2] 
                    fecha_previo = valles_previos.index[-2]
                    rsi_previo = valle_a['RSI']
                    if vela_ob['low'] < valle_a['low'] and vela_ob['RSI'] > valle_a['RSI']:
                        tiene_divergencia = True

                velas_impulso = datos.index.get_loc(fecha) - datos.index.get_loc(fecha_ob)
                tamano_ob = vela_ob['high'] - vela_ob['low']
                cuerpo_bos = abs(vela['close'] - vela['open'])
                profundidad = ultimo_pico_precio - vela_ob['low']
                delta_rsi = abs(vela_ob['RSI'] - rsi_previo)

                es_ia_ganadora = False
                if cuerpo_bos <= 252.63:
                    if 156.10 < tamano_ob <= 173.55:
                        es_ia_ganadora = True
                else:
                    if delta_rsi <= 22.24 and profundidad > 502.71:
                        es_ia_ganadora = True

                etiqueta = "⭐"
                if es_discount: etiqueta += "F"
                if tiene_divergencia: etiqueta += "R"
                
                etiqueta_base = etiqueta 
                if es_ia_ganadora:
                    etiqueta += " 👁️"
                
                precio_entrada = vela_ob['high']
                sl = vela_ob['low']
                tp = precio_entrada + ((precio_entrada - sl) * 2)
                
                posicion_rompimiento = datos.index.get_loc(fecha)
                resultado = 'Pendiente'
                fecha_entrada_real = None
                
                if posicion_rompimiento + 1 < len(datos):
                    futuro = datos.iloc[posicion_rompimiento + 1 :]
                    velas_entrada = futuro[futuro['low'] <= precio_entrada]
                    
                    if not velas_entrada.empty:
                        fecha_entrada_real = velas_entrada.index[0]
                        futuro_post = datos.loc[fecha_entrada_real:]
                        for f_s, v_s in futuro_post.iterrows():
                            if v_s['low'] <= sl: 
                                resultado = 'Loss'; break
                            elif v_s['high'] >= tp: 
                                resultado = 'Win'; break
                                
                if resultado == 'Win':
                    if etiqueta_base == "⭐": w_base_c += 1
                    elif etiqueta_base == "⭐F": w_fibo_c += 1
                    elif etiqueta_base == "⭐R": w_rsi_c += 1
                    elif etiqueta_base == "⭐FR": w_full_c += 1
                    if es_ia_ganadora: w_ia_c += 1
                elif resultado == 'Loss':
                    if etiqueta_base == "⭐": l_base_c += 1
                    elif etiqueta_base == "⭐F": l_fibo_c += 1
                    elif etiqueta_base == "⭐R": l_rsi_c += 1
                    elif etiqueta_base == "⭐FR": l_full_c += 1
                    if es_ia_ganadora: l_ia_c += 1

                if resultado in ['Win', 'Loss']:
                    dataset_ml.append({
                        'Tipo': 1, 'Velas_Impulso': velas_impulso, 'Tamano_OB': tamano_ob,
                        'Cuerpo_BOS': cuerpo_bos, 'Profundidad_Pts': profundidad, 'Delta_RSI': delta_rsi,
                        'Target': 1 if resultado == 'Win' else 0
                    })

                zonas_alta_probabilidad.append({
                    'Tipo': 'Compra', 'Fecha_OB': fecha_ob, 'Fecha_Rompimiento': fecha, 'Fecha_Prev': fecha_previo,
                    'OB_High': vela_ob['high'], 'OB_Low': vela_ob['low'], 'RSI_OB': vela_ob['RSI'], 
                    'RSI_Prev': valle_a['RSI'] if tiene_divergencia else None, 'Resultado': resultado, 
                    'Estrellas': etiqueta, 'Divergencia': tiene_divergencia, 'Fecha_Entrada': fecha_entrada_real 
                })
            ultimo_pico_precio = 0.0

        # ==========================================
        # 🔴 LÓGICA DE VENTAS (SHORTS)
        # ==========================================
        if vela['Valle'] == True:
            ultimo_valle_precio = vela['low']
            indice_ultimo_valle = fecha
            
        elif ultimo_valle_precio > 0 and vela['close'] < ultimo_valle_precio:
            rango_impulso = datos.loc[indice_ultimo_valle : fecha]
            fecha_ob = rango_impulso['high'].idxmax()
            vela_ob = datos.loc[fecha_ob]
            
            if datos.loc[fecha_ob : fecha].head(5)['FVG_Bajista'].any():
                nivel_50 = ultimo_valle_precio + ((vela_ob['high'] - ultimo_valle_precio) * 0.5)
                es_premium = vela_ob['low'] >= nivel_50
                
                picos_previos = datos[datos['Pico'] == True].loc[:fecha_ob].tail(2)
                tiene_divergencia = False
                fecha_previo = None
                rsi_previo = vela_ob['RSI'] 
                
                if len(picos_previos) > 1:
                    pico_a = picos_previos.iloc[-2] 
                    fecha_previo = picos_previos.index[-2]
                    rsi_previo = pico_a['RSI']
                    if vela_ob['high'] > pico_a['high'] and vela_ob['RSI'] < pico_a['RSI']:
                        tiene_divergencia = True

                velas_impulso = datos.index.get_loc(fecha) - datos.index.get_loc(fecha_ob)
                tamano_ob = vela_ob['high'] - vela_ob['low']
                cuerpo_bos = abs(vela['close'] - vela['open'])
                profundidad = vela_ob['high'] - ultimo_valle_precio
                delta_rsi = abs(vela_ob['RSI'] - rsi_previo)

                es_ia_ganadora = False
                if profundidad <= 599.56:
                    if velas_impulso <= 3.50 and tamano_ob <= 157.43:
                        es_ia_ganadora = True
                else:
                    if delta_rsi <= 9.87:
                        es_ia_ganadora = True

                etiqueta = "⭐"
                if es_premium: etiqueta += "F"
                if tiene_divergencia: etiqueta += "R"
                
                etiqueta_base = etiqueta
                if es_ia_ganadora:
                    etiqueta += " 👁️"
                
                precio_entrada = vela_ob['low']  
                sl = vela_ob['high']             
                tp = precio_entrada - ((sl - precio_entrada) * 2) 
                
                posicion_rompimiento = datos.index.get_loc(fecha)
                resultado = 'Pendiente'
                fecha_entrada_real = None
                
                if posicion_rompimiento + 1 < len(datos):
                    futuro = datos.iloc[posicion_rompimiento + 1 :]
                    velas_entrada = futuro[futuro['high'] >= precio_entrada]
                    
                    if not velas_entrada.empty:
                        fecha_entrada_real = velas_entrada.index[0]
                        futuro_post = datos.loc[fecha_entrada_real:]
                        for f_s, v_s in futuro_post.iterrows():
                            if v_s['high'] >= sl:  
                                resultado = 'Loss'; break
                            elif v_s['low'] <= tp: 
                                resultado = 'Win'; break
                                
                if resultado == 'Win':
                    if etiqueta_base == "⭐": w_base_v += 1
                    elif etiqueta_base == "⭐F": w_fibo_v += 1
                    elif etiqueta_base == "⭐R": w_rsi_v += 1
                    elif etiqueta_base == "⭐FR": w_full_v += 1
                    if es_ia_ganadora: w_ia_v += 1
                elif resultado == 'Loss':
                    if etiqueta_base == "⭐": l_base_v += 1
                    elif etiqueta_base == "⭐F": l_fibo_v += 1
                    elif etiqueta_base == "⭐R": l_rsi_v += 1
                    elif etiqueta_base == "⭐FR": l_full_v += 1
                    if es_ia_ganadora: l_ia_v += 1

                if resultado in ['Win', 'Loss']:
                    dataset_ml.append({
                        'Tipo': 0, 'Velas_Impulso': velas_impulso, 'Tamano_OB': tamano_ob,
                        'Cuerpo_BOS': cuerpo_bos, 'Profundidad_Pts': profundidad, 'Delta_RSI': delta_rsi,
                        'Target': 1 if resultado == 'Win' else 0
                    })

                zonas_alta_probabilidad.append({
                    'Tipo': 'Venta', 'Fecha_OB': fecha_ob, 'Fecha_Rompimiento': fecha, 'Fecha_Prev': fecha_previo,
                    'OB_High': vela_ob['high'], 'OB_Low': vela_ob['low'], 'RSI_OB': vela_ob['RSI'], 
                    'RSI_Prev': pico_a['RSI'] if tiene_divergencia else None, 'Resultado': resultado, 
                    'Estrellas': etiqueta, 'Divergencia': tiene_divergencia, 'Fecha_Entrada': fecha_entrada_real 
                })
            ultimo_valle_precio = 0.0

    df_zonas_perfectas = pd.DataFrame(zonas_alta_probabilidad)
    df_ml = pd.DataFrame(dataset_ml)
    df_ml.to_csv('dataset_sintetico_ml.csv', index=False)

    # --- CÁLCULOS COMPRAS ---
    t_base_c = w_base_c + l_base_c; wr_base_c = (w_base_c / t_base_c * 100) if t_base_c > 0 else 0
    t_fibo_c = w_fibo_c + l_fibo_c; wr_fibo_c = (w_fibo_c / t_fibo_c * 100) if t_fibo_c > 0 else 0
    t_rsi_c = w_rsi_c + l_rsi_c;    wr_rsi_c = (w_rsi_c / t_rsi_c * 100) if t_rsi_c > 0 else 0
    t_full_c = w_full_c + l_full_c; wr_full_c = (w_full_c / t_full_c * 100) if t_full_c > 0 else 0
    w_tot_c = w_base_c + w_fibo_c + w_rsi_c + w_full_c
    l_tot_c = l_base_c + l_fibo_c + l_rsi_c + l_full_c
    t_tot_c = w_tot_c + l_tot_c
    wr_tot_c = (w_tot_c / t_tot_c * 100) if t_tot_c > 0 else 0
    t_ia_c = w_ia_c + l_ia_c; wr_ia_c = (w_ia_c / t_ia_c * 100) if t_ia_c > 0 else 0

    # --- CÁLCULOS VENTAS ---
    t_base_v = w_base_v + l_base_v; wr_base_v = (w_base_v / t_base_v * 100) if t_base_v > 0 else 0
    t_fibo_v = w_fibo_v + l_fibo_v; wr_fibo_v = (w_fibo_v / t_fibo_v * 100) if t_fibo_v > 0 else 0
    t_rsi_v = w_rsi_v + l_rsi_v;    wr_rsi_v = (w_rsi_v / t_rsi_v * 100) if t_rsi_v > 0 else 0
    t_full_v = w_full_v + l_full_v; wr_full_v = (w_full_v / t_full_v * 100) if t_full_v > 0 else 0
    w_tot_v = w_base_v + w_fibo_v + w_rsi_v + w_full_v
    l_tot_v = l_base_v + l_fibo_v + l_rsi_v + l_full_v
    t_tot_v = w_tot_v + l_tot_v
    wr_tot_v = (w_tot_v / t_tot_v * 100) if t_tot_v > 0 else 0
    t_ia_v = w_ia_v + l_ia_v; wr_ia_v = (w_ia_v / t_ia_v * 100) if t_ia_v > 0 else 0

    # --- CÁLCULOS GLOBALES ---
    w_global = w_tot_c + w_tot_v
    l_global = l_tot_c + l_tot_v
    t_global = w_global + l_global
    wr_global = (w_global / t_global * 100) if t_global > 0 else 0
    t_ia_global = t_ia_c + t_ia_v
    w_ia_global = w_ia_c + w_ia_v
    wr_ia_global = (w_ia_global / t_ia_global * 100) if t_ia_global > 0 else 0

    # ==========================================
    # 📱 PREPARANDO EL REPORTE PARA TELEGRAM
    # ==========================================
    mensaje_telegram = f"""
📈 REPORTE CUANTITATIVO DIARIO - BOOM 500 📉

🟢 COMPRAS (LONGS)
Totales: {t_tot_c} | Efectividad: {wr_tot_c:.2f}% ({w_tot_c}W / {l_tot_c}L)
[⭐] BASE    : {wr_base_c:5.2f}% ({w_base_c}W/{l_base_c}L)
[⭐F] FIBO   : {wr_fibo_c:5.2f}% ({w_fibo_c}W/{l_fibo_c}L)
[⭐R] RSI    : {wr_rsi_c:5.2f}% ({w_rsi_c}W/{l_rsi_c}L)
[⭐FR] FULL  : {wr_full_c:5.2f}% ({w_full_c}W/{l_full_c}L)
👁️ IA       : {wr_ia_c:5.2f}% ({w_ia_c}W/{l_ia_c}L)

🔴 VENTAS (SHORTS)
Totales: {t_tot_v} | Efectividad: {wr_tot_v:.2f}% ({w_tot_v}W / {l_tot_v}L)
[⭐] BASE    : {wr_base_v:5.2f}% ({w_base_v}W/{l_base_v}L)
[⭐F] FIBO   : {wr_fibo_v:5.2f}% ({w_fibo_v}W/{l_fibo_v}L)
[⭐R] RSI    : {wr_rsi_v:5.2f}% ({w_rsi_v}W/{l_rsi_v}L)
[⭐FR] FULL  : {wr_full_v:5.2f}% ({w_full_v}W/{l_full_v}L)
👁️ IA       : {wr_ia_v:5.2f}% ({w_ia_v}W/{l_ia_v}L)

🏆 RESUMEN GLOBAL UNIFICADO: {wr_global:.2f}% | Ops: {t_global}
🤖 RESUMEN GLOBAL IA (OJO) : {wr_ia_global:.2f}% | Ops: {t_ia_global}
"""
    print(mensaje_telegram) # Imprimir en consola local también

    # ==========================================
    # 🎨 GUARDADO DE IMAGEN (PLOTLY)
    # ==========================================
    print("🎨 Generando imagen HMI V2.0...")
    velas_a_graficar = 3000 

    # 1. Copiamos los datos
    df_plot = datos.tail(velas_a_graficar).copy()
    
    # 2. EL PARCHE NUCLEAR: Convertimos todo el índice de fechas a texto
    df_plot.index = df_plot.index.astype(str)
    
    fecha_inicio_plot = datos.index[-velas_a_graficar]
    ultimo_tiempo = datos.index[-1]

    zonas_v20 = df_zonas_perfectas[df_zonas_perfectas['Fecha_OB'] >= fecha_inicio_plot]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.75, 0.25], 
                        subplot_titles=("Gráfico Maestro V2.0 (SMC + Filtro IA 👁️)", "RSI Momentum"))

    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot['open'], high=df_plot['high'],
        low=df_plot['low'], close=df_plot['close'], name='V75'
    ), row=1, col=1)

    for _, zona in zonas_v20.iterrows():
        top_y = zona['OB_High']
        bottom_y = zona['OB_Low']
        
        # --- EL PARCHE: Convertir fechas a texto (String) ---
        start_x = str(zona['Fecha_OB'])
        
        if pd.notna(zona.get('Fecha_Entrada')): 
            end_x = str(zona['Fecha_Entrada'])
        else: 
            end_x = str(ultimo_tiempo)
            
        resultado = zona.get('Resultado', 'Pendiente')
        if resultado == 'Win': color = 'rgba(0, 255, 0, 0.3)'
        elif resultado == 'Loss': color = 'rgba(255, 255, 0, 0.3)'
        else: color = 'rgba(0, 150, 255, 0.4)'
            
        fig.add_shape(
            type="rect", x0=start_x, y0=bottom_y, x1=end_x, y1=top_y,
            fillcolor=color, line=dict(color=color.replace('0.3', '0.8').replace('0.4', '0.8'), width=1),
            layer="below", row=1, col=1
        )
        
        if zona['Divergencia'] == True:
            if zona['Tipo'] == 'Compra':
                y_precio_prev = datos.loc[zona['Fecha_Prev']]['low']
                y_precio_ob = zona['OB_Low']
            else:
                y_precio_prev = datos.loc[zona['Fecha_Prev']]['high']
                y_precio_ob = zona['OB_High']
                
            fecha_prev_str = str(zona['Fecha_Prev'])
            
            fig.add_trace(go.Scatter(
                x=[fecha_prev_str, start_x], y=[y_precio_prev, y_precio_ob],
                mode='lines', line=dict(color='red', width=2, dash='dot'), showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=[fecha_prev_str, start_x], y=[zona['RSI_Prev'], zona['RSI_OB']],
                mode='lines', line=dict(color='lime', width=2, dash='solid'), showlegend=False
            ), row=2, col=1)

        es_ia = "👁️" in zona['Estrellas']
        color_texto = "cyan" if es_ia else "white"
        tamano_texto = 16 if es_ia else 13
        texto_etiqueta = f"{'▲' if zona['Tipo'] == 'Compra' else '▼'} {zona['Estrellas']}"
        
        fig.add_annotation(
            x=start_x, y=zona['OB_High'] if zona['Tipo'] == 'Compra' else zona['OB_Low'],
            text=texto_etiqueta, showarrow=False, 
            yshift=18 if zona['Tipo'] == 'Compra' else -18,
            font=dict(size=tamano_texto, color=color_texto), row=1, col=1
        )

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], line=dict(color='orange', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(
        title='HMI Súper Indicador V2.0', template='plotly_dark', height=900, width=1400,
        xaxis_rangeslider_visible=False, margin=dict(l=50, r=50, b=50, t=80)
    )
    fig.update_xaxes(matches='x')
    
    # GUARDAR LA IMAGEN (Reemplaza fig.show())
    ruta_imagen = 'reporte_boom500.png'
    fig.write_image(ruta_imagen)
    print("✅ Imagen guardada correctamente.")

    # ==========================================
    # 🚀 ENVÍO A TELEGRAM
    # ==========================================
    print("📲 Enviando reporte a Telegram...")
    url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendPhoto"
    
    with open(ruta_imagen, 'rb') as foto:
        payload = {'chat_id': CHAT_ID_TELEGRAM, 'caption': mensaje_telegram}
        archivos = {'photo': foto}
        respuesta = requests.post(url, data=payload, files=archivos)
        
        if respuesta.status_code == 200:
            print("✅ ¡Reporte enviado con éxito a tu celular!")
        else:
            print(f"❌ Error al enviar: {respuesta.text}")
            sys.exit(1)

# ==========================================
# EJECUCIÓN DEL SCRIPT
# ==========================================
if __name__ == '__main__':
    try:
        # Máximo 120 s para todo el proceso (descarga paginada + backtest + gráfico + Telegram)
        asyncio.run(asyncio.wait_for(main(), timeout=120.0))
    except asyncio.TimeoutError:
        print("❌ Error: timeout de 120s (conexión con Deriv colgada).")
        sys.exit(1)
