import os, json, pickle, io
import numpy as np, pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt

from core import rbf_core, metrics as metrics_mod
from utils import file_manager, drive_loader, logger

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RESULTS_DIR = "resultados"
os.makedirs(RESULTS_DIR, exist_ok=True)

class RBFApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RBF App - Modular")
        self.geometry("1100x720")
        # state
        self.df_raw = None
        self.df_numeric = None
        self.X = self.Y = None
        self.X_train = self.X_sim = self.Y_train = self.Y_sim = None
        self.scaler = None
        self.centers = None
        self.weights = None
        self.n_centers = tk.IntVar(value=2)
        self.error_opt = tk.DoubleVar(value=0.1)
        self.seed = 42
        self.meta = {}
        self._build_ui()

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.tab_data = ttk.Frame(nb)
        self.tab_config = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)
        self.tab_sim = ttk.Frame(nb)
        self.tab_plots = ttk.Frame(nb)
        nb.add(self.tab_data, text="Carga de Datos")
        nb.add(self.tab_config, text="Configuración RBF")
        nb.add(self.tab_train, text="Entrenamiento")
        nb.add(self.tab_sim, text="Simulación")
        nb.add(self.tab_plots, text="Gráficas")
        self._build_tab_data()
        self._build_tab_config()
        self._build_tab_train()
        self._build_tab_sim()
        self._build_tab_plots()

    def log(self, text):
        try:
            self.log_text.config(state="normal")
            self.log_text.insert("end", logger.format_msg(text) + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        except Exception:
            print(text)

    def _build_tab_data(self):
        f = self.tab_data
        btn_row = ttk.Frame(f); btn_row.pack(fill="x", padx=6, pady=6)
        ttk.Button(btn_row, text="Limpiar Todo", command=self.reset_all).grid(row=0,column=3,padx=4)
        ttk.Button(btn_row, text="Cargar Dataset (CSV Local)", command=self.load_dataset_local).grid(row=0,column=0,padx=4)
        ttk.Button(btn_row, text="Cargar Dataset (URL/Drive)", command=self.load_dataset_url).grid(row=0,column=1,padx=4)
        ttk.Button(btn_row, text="Agregar datos", command=self.add_data_dialog).grid(row=0,column=2,padx=4)
        self.dataset_status_lbl = ttk.Label(f, text="Dataset: No cargado"); self.dataset_status_lbl.pack(anchor="w", padx=6)
        self.dataset_info_lbl = ttk.Label(f, text="Entradas: 0 | Salidas: 0 | Patrones: 0"); self.dataset_info_lbl.pack(anchor="w", padx=6)
        pv = ttk.Labelframe(f, text="Vista Previa (Train 70% - Sim 30%)"); pv.pack(fill="both", expand=True, padx=6, pady=6)
        self.tv_train = ttk.Treeview(pv, show="headings"); self.tv_sim = ttk.Treeview(pv, show="headings")
        self.tv_train.pack(side="left", fill="both", expand=True); self.tv_sim.pack(side="left", fill="both", expand=True)
        self.log_text = tk.Text(f, height=6, state="disabled"); self.log_text.pack(fill="x", padx=6, pady=6)

    def load_dataset_local(self):
     path = filedialog.askopenfilename(filetypes=[("CSV","*.csv"),("All","*.*")])
     if not path: return
     try:
        df = pd.read_csv(path)
        self.last_dataset_path = path   # ✅ Agregado: recordamos la ruta del dataset original
     except Exception as e:
        messagebox.showerror("Error", f"No se pudo leer CSV: {e}"); return
     self._process_loaded_df(df, path); self.log(f"Dataset cargado: {path}")


    def load_dataset_url(self):
        url = simpledialog.askstring("URL", "Pega la URL pública del CSV (Drive o web):")
        if not url: return
        try:
            df = drive_loader.load_csv_from_url(url)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer desde URL: {e}"); return
        self._process_loaded_df(df); self.log(f"Dataset cargado desde URL")

    def add_data_dialog(self):
        if self.df_raw is None:
            messagebox.showwarning("No dataset","Carga un dataset antes de añadir datos."); return
        top = tk.Toplevel(self); top.title("Agregar Patron (append)")
        tk.Label(top, text="Ingresa valores separados por comas en el mismo orden de columnas:").pack(anchor="w", padx=6, pady=4)
        txt = tk.Entry(top, width=80); txt.pack(padx=6, pady=4)
        def add():
            s = txt.get().strip()
            if not s: messagebox.showwarning("Aviso","No hay datos"); return
            vals = [v.strip() for v in s.split(",")]
            if len(vals) != len(self.df_raw.columns):
                messagebox.showerror("Error","Número de valores no coincide con columnas."); return
            try:
                row = pd.Series(vals, index=self.df_raw.columns)
                self.df_raw = pd.concat([self.df_raw, pd.DataFrame([row])], ignore_index=True)
                self._process_loaded_df(self.df_raw)
                self.log(f"Agregado patrón: {s}")
                top.destroy()
            except Exception as e:
                messagebox.showerror("Error", str(e))
        ttk.Button(top, text="Agregar", command=add).pack(pady=4)

    def _process_loaded_df(self, df, path=None):
     """
     Procesa el DataFrame cargado y registra diagnóstico detallado.
     Intenta convertir cada columna a numérico (limpiando coma decimal y caracteres extra)
     y solo elimina columnas que quedaron totalmente NaN.
     """
     import re, math
     self.df_raw = df.copy()
     if path:
        self.last_dataset_path = path

     # Log: columnas originales y tipos
     cols_orig = list(self.df_raw.columns)
     self.log(f"Columnas originales ({len(cols_orig)}): {cols_orig}")
     #También imprimir a consola para ver rápidamente
     print(f"[DEBUG] Columnas originales: {cols_orig}")

     # -- Intentar limpiar y convertir cada columna a numérico --
     df_work = self.df_raw.copy()
     # Normalizar valores tipo string: reemplazar coma decimal por punto
     df_work = df_work.applymap(lambda v: str(v).strip().replace(",", ".") if not pd.isna(v) else v)

     numeric_df = pd.DataFrame(index=df_work.index)
     convert_report = []
     for col in df_work.columns:
        col_series = df_work[col]
        # eliminar espacios y símbolos excepto numericos, punto, minus, e/E
        cleaned = col_series.astype(str).apply(lambda v: re.sub(r"[^\d\.\-eE]", "", v) if v != "nan" else "")
        # convertir
        num = pd.to_numeric(cleaned.replace("", pd.NA), errors="coerce")
        # contar cuantos valores no-nulos hay
        n_nonnull = int(num.count())
        convert_report.append((col, n_nonnull, col_series.dtype))
        numeric_df[col] = num

     # Registrar reporte de conversión
     for col, n_nonnull, dtype in convert_report:
        self.log(f"Col '{col}': convertidos={n_nonnull}, tipo original={dtype}")
        print(f"[DEBUG] Col '{col}': convertidos={n_nonnull}, tipo original={dtype}")

     # Eliminar columnas totalmente NaN (no convertibles)
     numeric_df = numeric_df.dropna(axis=1, how="all")
     remaining_cols = list(numeric_df.columns)
     self.log(f"Columnas numéricas retenidas ({len(remaining_cols)}): {remaining_cols}")
     print(f"[DEBUG] Columnas numéricas retenidas: {remaining_cols}")

     if len(remaining_cols) < 2:
        messagebox.showerror("Error", "No se detectaron suficientes columnas numéricas (se requieren al menos 2). Revisa el CSV.")
        return

     # llenar df_numeric
     self.df_numeric = numeric_df.copy()

     # rellenar NaN con mediana por columna
     dfn = self.df_numeric.fillna(self.df_numeric.median(numeric_only=True))

     # determinar Y como última columna numérica
     y_cols = [dfn.columns[-1]]
     X_cols = [c for c in dfn.columns if c not in y_cols]

     # Guardar meta para debug
     self.meta = {"X_cols": X_cols, "Y_cols": y_cols}
     self.log(f"Se asumió Y: {y_cols[0]}; X: {X_cols}")

     # construir X, Y
     try:
        X = dfn[X_cols].astype(float).values
        Y = dfn[y_cols].astype(float).values
     except Exception as e:
        self.log(f"Error al convertir X/Y a float: {e}")
        messagebox.showerror("Error", f"No se pudieron convertir X/Y a float: {e}")
        return

     # escalar y dividir
     self.scaler = StandardScaler()
     Xs = self.scaler.fit_transform(X)
     X_train, X_sim, Y_train, Y_sim = train_test_split(Xs, Y, test_size=0.3, random_state=self.seed, shuffle=True)

     # guardar en estado
     self.X = Xs; self.Y = Y
     self.X_train = X_train; self.X_sim = X_sim; self.Y_train = Y_train; self.Y_sim = Y_sim

     # actualizar interfaz
     self._update_info_and_preview()
     self.log(f"Procesamiento finalizado. Entradas detectadas: {len(X_cols)}, Salidas: {len(y_cols)}, Patrones: {X.shape[0]}")
     print(f"[DEBUG] Entradas detectadas: {len(X_cols)}, Salidas: {len(y_cols)}, Patrones: {X.shape[0]}")

    def _prepare_xy(self):
        dfn = self.df_numeric.copy().fillna(self.df_numeric.median())
        y_cols = [dfn.columns[-1]]
        X_cols = [c for c in dfn.columns if c not in y_cols]
        X = dfn[X_cols].astype(float).values; Y = dfn[y_cols].astype(float).values
        self.scaler = StandardScaler(); Xs = self.scaler.fit_transform(X)
        X_train, X_sim, Y_train, Y_sim = train_test_split(Xs, Y, test_size=0.3, random_state=self.seed, shuffle=True)
        self.X = Xs; self.Y = Y; self.X_train=X_train; self.X_sim=X_sim; self.Y_train=Y_train; self.Y_sim=Y_sim
        self.meta = {"X_cols": X_cols, "Y_cols": y_cols}

    def _update_info_and_preview(self):
     """
     Actualiza la información del dataset (entradas, salidas, patrones)
     y muestra una vista previa de los datos de entrenamiento y simulación.
     """
     # Validar que los datos existan
     if self.X is None or self.Y is None:
        self.dataset_status_lbl.config(text="Dataset: No cargado")
        self.dataset_info_lbl.config(text="Entradas: 0 | Salidas: 0 | Patrones: 0")
        return

     # Calcular número de entradas, salidas y patrones
     entries = int(self.X.shape[1])
     outputs = int(self.Y.shape[1]) if getattr(self.Y, "ndim", 1) > 1 else 1
     patterns = int(self.X.shape[0])

     # Mostrar información en etiquetas
     self.dataset_status_lbl.config(text="Dataset: Cargado")
     self.dataset_info_lbl.config(text=f"Entradas: {entries} | Salidas: {outputs} | Patrones: {patterns}")

     # Determinar columnas de la vista previa
     cols = self.meta.get("X_cols", []) + self.meta.get("Y_cols", [])
     if not cols:
        cols = [f"x{i+1}" for i in range(entries)] + [f"y{i+1}" for i in range(outputs)]

     # Limpiar y configurar las tablas de vista previa
     for tv in (self.tv_train, self.tv_sim):
        for k in tv.get_children():
            tv.delete(k)
        tv["columns"] = cols
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=100, anchor="center")

     # Función auxiliar para generar filas de ejemplo
     def rows_from(arrX, arrY):
        rows = []
        nrows = min(arrX.shape[0], 200)
        for i in range(nrows):
            xin = self.scaler.inverse_transform(arrX[i].reshape(1, -1))[0] if self.scaler else arrX[i]
            yd = arrY[i]
            rows.append(list(xin) + list(yd))
        return rows

     # Poblar tablas con una muestra de datos
     for r in rows_from(self.X_train, self.Y_train):
        self.tv_train.insert("", "end", values=r)
     for r in rows_from(self.X_sim, self.Y_sim):
        self.tv_sim.insert("", "end", values=r)

    def _build_tab_config(self):
        f = self.tab_config
        frm = ttk.Frame(f); frm.pack(fill="x", padx=6, pady=6)
        ttk.Label(frm, text="Número de Centros Radiales:").grid(row=0,column=0,sticky="w")
        ttk.Entry(frm, textvariable=self.n_centers, width=8).grid(row=0,column=1,padx=6)
        ttk.Label(frm, text="Error óptimo (0-0.1):").grid(row=1,column=0,sticky="w")
        ttk.Entry(frm, textvariable=self.error_opt, width=8).grid(row=1,column=1,padx=6)
        ttk.Button(frm, text="Inicializar Centros (Aleatorio)", command=lambda:self.init_centers(True)).grid(row=2,column=0,padx=6,pady=6)
        ttk.Button(frm, text="Inicializar Centros (Manual)", command=lambda:self.init_centers(False)).grid(row=2,column=1,padx=6)
        ttk.Button(frm, text="Guardar Modelo Local", command=self.save_model).grid(row=2,column=2,padx=6)
        ttk.Button(frm, text="Cargar Modelo Local", command=self.load_model).grid(row=2,column=3,padx=6)
        self.tv_centers = ttk.Treeview(f, show="headings"); self.tv_centers.pack(fill="both",expand=True,padx=6,pady=6)

    def validate_config(self):
        if self.X_train is None: messagebox.showwarning("Sin dataset","Carga dataset"); return False
        try:
            n = int(self.n_centers.get())
        except Exception:
            messagebox.showerror("Error","ncentros inválido"); return False
        if n < self.X_train.shape[1]:
            messagebox.showerror("Validación",f"n_centros ({n}) debe ser >= n_entradas ({self.X_train.shape[1]})"); return False
        eo=float(self.error_opt.get())
        if not (0.0 <= eo <= 0.1):
            messagebox.showerror("Validación","Error óptimo fuera del rango 0-0.1"); return False
        return True

    def init_centers(self, aleatorio=True):
        if not self.validate_config(): return
        n = int(self.n_centers.get())
        if aleatorio:
            self.centers = rbf_core.initialize_centers_random(self.X_train, n, seed=self.seed)
        else:
            txt = simpledialog.askstring("Manual", f"Ingrese {n} centros, uno por linea, vals separados por comas")
            lines = txt.strip().splitlines() if txt else []
            centers=[]
            for ln in lines:
                parts=[float(p) for p in ln.split(",")]
                centers.append(parts)
           # después de asignar self.centers
        self.centers = np.asarray(self.centers)
        # si la forma no coincide con (n_centers, n_inputs) intentar transponer
        if self.centers.ndim == 2:
           n_centers_expected = int(self.n_centers.get())
        if self.centers.shape[0] != n_centers_expected and self.centers.shape[1] == n_centers_expected:
           self.centers = self.centers.T
        # si quedó 1D y n_centers>1, intentar reshaping
        if self.centers.ndim == 1:
           self.centers = self.centers.reshape((n_centers_expected, -1))
        self._populate_centers_tree() 
        self.log(f"Inicializados {n_centers_expected} centros.")


    def _populate_centers_tree(self):
     """
     Muestra en tv_centers una tabla donde:
      - columnas: 'Feature' + c1..cNcenters
      - filas: cada fila corresponde a una *feature* (entrada), con el valor de cada centro para esa feature.
     Esto asegura que si el usuario pide 4 centros, se vean c1..c4.
     """
     tv = self.tv_centers
     # limpiar
     for r in tv.get_children():
        tv.delete(r)

     if self.centers is None:
        # limpiar columnas si no hay centros
        tv["columns"] = ()
        return

     # Asegurar que centers tenga la forma (n_centers, n_inputs)
     centers = np.asarray(self.centers)
     # si se pasó en forma (n_inputs, n_centers) lo corregimos
     if centers.ndim == 2:
        n0, n1 = centers.shape
        # si primera dimensión coincide con n_inputs (X_train cols) y segunda con n_centers, queremos (n_centers, n_inputs)
        if hasattr(self, "X_train") and self.X_train is not None:
            n_inputs = self.X_train.shape[1]
            if n0 == n_inputs and (n1 == int(self.n_centers.get()) or n1 != n_inputs):
                centers = centers.T
     else:
        centers = centers.reshape((int(self.n_centers.get()), -1))

     # now centers is (n_centers, n_inputs)
     n_centers, n_inputs = centers.shape

     # construir columnas: Feature, c1..cN
     cols = ["Feature"] + [f"c{i+1}" for i in range(n_centers)]
     tv["columns"] = cols
     # configurar headings
     for c in cols:
        tv.heading(c, text=c)
        tv.column(c, width=120, anchor="center")

     # nombres de features
     feature_names = self.meta.get("X_cols", [f"x{i+1}" for i in range(n_inputs)])
     # insertar filas por feature
     for feat_idx in range(n_inputs):
        row_vals = [feature_names[feat_idx]] + [float(centers[c_idx, feat_idx]) for c_idx in range(n_centers)]
        tv.insert("", "end", values=row_vals)

     # guardar de vuelta la versión normalizada (para uso interno)
     self.centers = centers

    def save_model(self):
     if self.centers is None or self.weights is None:
        messagebox.showwarning("Sin modelo", "Entrena un modelo antes de guardarlo.")
        return

     folder = filedialog.askdirectory(title="Selecciona carpeta donde guardar el modelo")
     if not folder:
        messagebox.showinfo("Cancelado", "No se seleccionó carpeta de destino.")
        return

     name = simpledialog.askstring("Nombre del modelo", "Asigna un nombre al modelo (sin espacios):")
     if not name:
        name = pd.Timestamp.now().strftime("modelo_%Y%m%d_%H%M%S")

     safe_name = "".join(c for c in name if c.isalnum() or c in ("_", "-")).strip()
     if not safe_name:
        messagebox.showerror("Error", "Nombre de modelo inválido.")
        return
     try:
        model_dir = os.path.join(folder, safe_name)
        os.makedirs(model_dir, exist_ok=True)
     except Exception as e:
        messagebox.showerror("Error", f"No se pudo crear carpeta:\n{e}")
        return

     try:
        np.save(os.path.join(model_dir, "centros.npy"), self.centers)
        np.save(os.path.join(model_dir, "pesos.npy"), self.weights)

        with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        cfg = {
            "nombre": safe_name,
            "n_centers": int(self.n_centers.get()),
            "error_opt": float(self.error_opt.get()),
            "meta": self.meta,
            "fecha_guardado": pd.Timestamp.now().isoformat()
        }
        with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)

        self.log(f"Modelo guardado correctamente en {model_dir}")
        messagebox.showinfo("Guardado exitoso", f"Modelo guardado en:\n{model_dir}")

     except Exception as e:
        messagebox.showerror("Error", f"No se pudo guardar el modelo:\n{e}")
        self.log(f"Error guardando modelo: {e}")


    def load_model(self):
     path = filedialog.askdirectory(title="Selecciona la carpeta del modelo a cargar")
     if not path:
        return

     try:
        centers_path = os.path.join(path, "centros.npy")
        weights_path = os.path.join(path, "pesos.npy")
        scaler_path = os.path.join(path, "scaler.pkl")
        config_path = os.path.join(path, "config.json")

        if not all(os.path.exists(p) for p in [centers_path, weights_path, scaler_path, config_path]):
            messagebox.showerror("Error", "La carpeta seleccionada no contiene un modelo válido.")
            return

        self.centers = np.load(centers_path)
        self.weights = np.load(weights_path)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            self.n_centers.set(cfg.get("n_centers", self.centers.shape[0]))
            self.error_opt.set(cfg.get("error_opt", 0.1))
            self.meta = cfg.get("meta", {})

        self._populate_centers_tree()
        self.log(f"Modelo cargado correctamente desde {path}")
        messagebox.showinfo("Modelo cargado", f"Modelo cargado desde:\n{path}")

     except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{e}")
        self.log(f"Error cargando modelo: {e}")

    def _build_tab_train(self):
        f=self.tab_train; btnf=ttk.Frame(f); btnf.pack(fill="x")
        ttk.Button(btnf, text="Iniciar Entrenamiento", command=self.start_training).pack(side="left", padx=6, pady=6)
        ttk.Button(btnf, text="Editar centros", command=self.edit_centers).pack(side="left", padx=6)
        self.train_state_lbl = ttk.Label(btnf, text="Estado: No iniciado"); self.train_state_lbl.pack(side="left", padx=12)
        self.tv_train_results = ttk.Treeview(f, columns=("Iter","EG","Ncent"), show="headings"); 
        for c in ("Iter","EG","Ncent"): self.tv_train_results.heading(c,text=c); self.tv_train_results.column(c,width=120)
        self.tv_train_results.pack(fill="x", padx=6, pady=6)
        self.tv_weights = ttk.Treeview(f, show="headings"); self.tv_weights.pack(fill="both", expand=True, padx=6, pady=6)

    def edit_centers(self):
        if self.centers is None: messagebox.showwarning("No centros","Inicializa primero"); return
        txt = simpledialog.askstring("Editar", "Ingresa centros (una linea por centro, valores separados por comas)")
        if not txt: return
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        arr=[] 
        for ln in lines: arr.append([float(p) for p in ln.split(",")])
        arr = np.array(arr)
        if arr.shape[1] != self.X_train.shape[1]: messagebox.showerror("Error","Dimensiones incorrectas"); return
        self.centers=arr; self._populate_centers_tree(); self.log("Centros editados.")

    def start_training(self):
        if self.X_train is None: messagebox.showwarning("No dataset","Carga dataset"); return
        if not self.validate_config(): return
        if self.centers is None: messagebox.showwarning("No centros","Inicializa centros"); return
        self.train_state_lbl.config(text="Estado: En curso"); self.log("Entrenamiento iniciado")
        Phi = rbf_core.compute_phi(self.X_train, self.centers); A = rbf_core.build_A(Phi)
        W = rbf_core.solve_weights(A, self.Y_train); self.weights = W
        Yr = rbf_core.predict(A, W); met = metrics_mod.metrics(self.Y_train, Yr)
        for r in self.tv_train_results.get_children(): self.tv_train_results.delete(r)
        self.tv_train_results.insert("", "end", values=(1, f"{met['EG']:.6f}", int(self.n_centers.get())))
        for r in self.tv_weights.get_children(): self.tv_weights.delete(r)
        self.tv_weights["columns"] = ("Param","Value")
        self.tv_weights.heading("Param",text="Param"); self.tv_weights.heading("Value",text="Value")
        for i in range(self.weights.shape[0]):
            name = "W0" if i==0 else f"W{i}"; val = float(self.weights[i,0]) if self.weights.ndim>1 else float(self.weights[i])
            self.tv_weights.insert("", "end", values=(name, f"{val:.6f}"))
        for idx,c in enumerate(self.centers, start=1):
            self.tv_weights.insert("", "end", values=(f"Center {idx}", ", ".join([f"{float(v):.6f}" for v in c])))
        self.train_state_lbl.config(text="Estado: Finalizado"); self.log(f"Entrenamiento finalizado (EG={met['EG']:.6f})")

    def _build_tab_sim(self):
        f=self.tab_sim; btnf=ttk.Frame(f); btnf.pack(fill="x")
        ttk.Button(btnf, text="Ejecutar Simulación", command=self.run_sim).pack(side="left", padx=6, pady=6)
        ttk.Button(btnf, text="Añadir Patrón de Prueba", command=self.add_sim_pattern).pack(side="left")
        ttk.Button(btnf, text="Cargar datos simulados (local)", command=self.load_simulated_csv).pack(side="left", padx=6)
        ttk.Button(btnf, text="Guardar datos simulados", command=self.save_simulated).pack(side="left", padx=6)
        self.tv_sim_results = ttk.Treeview(f, columns=("Pat","YR","YD","Err"), show="headings")
        for c in ("Pat","YR","YD","Err"): self.tv_sim_results.heading(c,text=c); self.tv_sim_results.column(c,width=120)
        self.tv_sim_results.pack(fill="both", expand=True, padx=6, pady=6)
        self.tv_metrics = ttk.Treeview(f, columns=("Conjunto","EG","MAE","RMSE","Convergencia"), show="headings")
        for c in ("Conjunto","EG","MAE","RMSE","Convergencia"): self.tv_metrics.heading(c,text=c); self.tv_metrics.column(c,width=120)
        self.tv_metrics.pack(fill="x", padx=6, pady=6)

    def add_sim_pattern(self):
        if self.X_train is None:
            messagebox.showwarning("Sin dataset", "Carga un dataset primero")
            s = simpledialog.askstring("Nuevo patrón", "Ingresa valores separados por comas (entradas + salida esperada)")
        if not s:
            return
        parts = [float(x.strip()) for x in s.split(",")]
        n_in = self.X_train.shape[1]
        n_out = self.Y_train.shape[1]
        if len(parts) != n_in + n_out:
            messagebox.showerror("Error", f"Debes ingresar {n_in + n_out} valores (entradas + salida)")
            return
        xin = np.array(parts[:n_in]).reshape(1, -1)
        yd = np.array(parts[n_in:]).reshape(1, -1)
        xin_s = self.scaler.transform(xin)
        self.X_sim = np.vstack([self.X_sim, xin_s]) if self.X_sim is not None else xin_s
        self.Y_sim = np.vstack([self.Y_sim, yd]) if self.Y_sim is not None else yd
        self.log("Nuevo patrón agregado a simulación")

    def run_sim(self):
        if self.centers is None or self.weights is None:
            messagebox.showwarning("Sin modelo", "Primero entrena o carga un modelo guardado.")
            return
        if self.X_sim is None or self.Y_sim is None:
            messagebox.showwarning("Sin datos", "No hay datos de simulación cargados.")
            return
        Phi_sim = rbf_core.compute_phi(self.X_sim, self.centers)
        A_sim = rbf_core.build_A(Phi_sim)
        Yr_sim = rbf_core.predict(A_sim, self.weights)

        met_sim = metrics_mod.metrics(self.Y_sim, Yr_sim)
        Phi_train = rbf_core.compute_phi(self.X_train, self.centers)
        A_train = rbf_core.build_A(Phi_train)
        Yr_train = rbf_core.predict(A_train, self.weights)
        met_train = metrics_mod.metrics(self.Y_train, Yr_train)

        for r in self.tv_sim_results.get_children():
            self.tv_sim_results.delete(r)
        Yd = np.atleast_2d(self.Y_sim)[:, 0]
        Yr = np.atleast_2d(Yr_sim)[:, 0]
        for i in range(len(Yd)):
            err = Yd[i] - Yr[i]
            self.tv_sim_results.insert("", "end", values=(i + 1, f"{Yr[i]:.6f}", f"{Yd[i]:.6f}", f"{err:.6f}"))

        for r in self.tv_metrics.get_children():
            self.tv_metrics.delete(r)
        conv_train = "Sí" if met_train["EG"] <= float(self.error_opt.get()) else "No"
        conv_sim = "Sí" if met_sim["EG"] <= float(self.error_opt.get()) else "No"

        self.tv_metrics.insert("", "end", values=("Entrenamiento", f"{met_train['EG']:.6f}", f"{met_train['MAE']:.6f}", f"{met_train['RMSE']:.6f}", conv_train))
        self.tv_metrics.insert("", "end", values=("Simulación", f"{met_sim['EG']:.6f}", f"{met_sim['MAE']:.6f}", f"{met_sim['RMSE']:.6f}", conv_sim))

        self.log("Simulación ejecutada correctamente.")   

    def load_simulated_csv(self):
     path = filedialog.askopenfilename(title="Selecciona el archivo CSV de simulación", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
     if not path:
        return
     folder = os.path.dirname(path)
     base = os.path.splitext(os.path.basename(path))[0]
     json_path = os.path.join(folder, f"{base}_metrics.json")
     index_path = os.path.join(folder, "sim_index.json")

     try:
        df = pd.read_csv(path)
        expected = {"Patron", "YR", "YD", "Error"}
        if not expected.issubset(set(df.columns)):
            messagebox.showerror("Error", f"El CSV seleccionado no contiene las columnas mínimas esperadas: {expected}")
            return

        for r in self.tv_sim_results.get_children():
            self.tv_sim_results.delete(r)
        for _, row in df.iterrows():
            vals = (int(row["Patron"]), row["YR"], row["YD"], row["Error"])
            self.tv_sim_results.insert("", "end", values=vals)

        for r in self.tv_metrics.get_children():
            self.tv_metrics.delete(r)
        if os.path.exists(json_path):
            import json
            with open(json_path, "r", encoding="utf-8") as jf:
                metrics_rows = json.load(jf)
            for m in metrics_rows:
                if isinstance(m, dict):
                    row = (m.get("Conjunto"), m.get("EG"), m.get("MAE"), m.get("RMSE"), m.get("Convergencia"))
                else:
                    row = tuple(m)
                self.tv_metrics.insert("", "end", values=row)
            self.log(f"Métricas cargadas desde {json_path}")
        else:
            self.log("No se encontró archivo de métricas junto al CSV; sólo se cargaron los datos.")

        if os.path.exists(index_path):
            try:
                import json
                with open(index_path, "r", encoding="utf-8") as idxf:
                    idx = json.load(idxf)
                self.log(f"Encontrado sim_index con {len(idx)} entradas en {folder}")
            except Exception:
                pass

        messagebox.showinfo("Carga completada", f"Datos simulados cargados desde:\n{path}")
     except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el archivo de simulación:\n{e}")
        self.log(f"Error cargando simulación: {e}")



    def save_simulated(self):
     if self.X_sim is None or self.Y_sim is None:
        messagebox.showwarning("Sin datos", "No hay datos de simulación para guardar.")
        return

     rows = []
     for item in self.tv_sim_results.get_children():
        vals = self.tv_sim_results.item(item, "values")
        if len(vals) >= 4:
            rows.append(vals[:4])
     if not rows:
        messagebox.showwarning("Sin resultados", "No hay resultados de simulación para guardar.")
        return
     folder = filedialog.askdirectory(title="Selecciona carpeta donde guardar la simulación")
     if not folder:
        return
     name = simpledialog.askstring("Nombre de la simulación", "Introduce un nombre identificador (sin espacios preferible):")
     if not name:
        from datetime import datetime
        name = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

     safe_name = "".join(c for c in name if c.isalnum() or c in ("_", "-")).strip()
     if safe_name == "":
        safe_name = f"sim_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

     csv_path = os.path.join(folder, f"{safe_name}.csv")
     json_path = os.path.join(folder, f"{safe_name}_metrics.json")
     index_path = os.path.join(folder, "sim_index.json")

     if os.path.exists(csv_path):
        if not messagebox.askyesno("Sobrescribir", f"El archivo {os.path.basename(csv_path)} ya existe. ¿Deseas sobrescribirlo?"):
            messagebox.showinfo("Guardado cancelado", "Elige otro nombre para guardar la simulación.")
            return
     try:
        df_res = pd.DataFrame(rows, columns=["Patron", "YR", "YD", "Error"])
        df_res["Patron"] = df_res["Patron"].astype(int)
        df_res["YR"] = pd.to_numeric(df_res["YR"], errors="coerce")
        df_res["YD"] = pd.to_numeric(df_res["YD"], errors="coerce")
        df_res["Error"] = pd.to_numeric(df_res["Error"], errors="coerce")
        df_res.to_csv(csv_path, index=False)
        metrics_rows = []
        for item in self.tv_metrics.get_children():
            vals = self.tv_metrics.item(item, "values")
            if vals:
                metrics_rows.append({
                    "Conjunto": vals[0],
                    "EG": vals[1],
                    "MAE": vals[2],
                    "RMSE": vals[3],
                    "Convergencia": vals[4]
                })
        if not metrics_rows:
            try:
                if "YR" in df_res.columns and not df_res["YR"].isna().all():
                    Yd = df_res["YD"].values.reshape(-1,1)
                    Yr = df_res["YR"].values.reshape(-1,1)
                    met = metrics_mod.metrics(Yd, Yr)
                    metrics_rows.append({
                        "Conjunto": "Simulación",
                        "EG": f"{met['EG']:.6f}",
                        "MAE": f"{met['MAE']:.6f}",
                        "RMSE": f"{met['RMSE']:.6f}",
                        "Convergencia": "Sí" if met["EG"] <= float(self.error_opt.get()) else "No"
                    })
            except Exception:
                pass
        import json
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(metrics_rows, jf, indent=4, ensure_ascii=False)
        index = []
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as idxf:
                    index = json.load(idxf)
            except Exception:
                index = []
        from datetime import datetime
        entry = {
            "name": safe_name,
            "csv": os.path.basename(csv_path),
            "metrics": os.path.basename(json_path),
            "saved_at": datetime.now().isoformat()
        }
        index.append(entry)
        with open(index_path, "w", encoding="utf-8") as idxf:
            json.dump(index, idxf, indent=4, ensure_ascii=False)

        self.log(f"Simulación guardada: {csv_path} (+ métricas {json_path})")
        messagebox.showinfo("Guardado", f"Simulación guardada en:\n{csv_path}\nMétricas: {json_path}")
     except Exception as e:
        messagebox.showerror("Error", f"No se pudo guardar la simulación:\n{e}")
        self.log(f"Error guardando simulación: {e}")



    def _build_tab_plots(self):
        f = self.tab_plots
        ttk.Button(f, text="Generar Gráficas", command=self.generate_plots).pack(anchor="nw", padx=6, pady=6)
        ttk.Button(f, text="Guardar Gráficas", command=self.save_plots).pack(anchor="nw", padx=6)

    def generate_plots(self):
     if self.weights is None or self.centers is None:
        messagebox.showwarning("Sin modelo", "Entrena o carga un modelo primero.")
        return
     if self.X_train is None or self.Y_train is None:
        messagebox.showwarning("Sin datos", "No hay datos de entrenamiento cargados.")
        return
     Phi_train = rbf_core.compute_phi(self.X_train, self.centers)
     A_train = rbf_core.build_A(Phi_train)
     Yr_train = rbf_core.predict(A_train, self.weights)
     Yd_train = np.atleast_2d(self.Y_train)[:, 0]
     Yr_train_1 = np.atleast_2d(Yr_train)[:, 0]

     met_train = metrics_mod.metrics(self.Y_train, Yr_train)

     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
     plt.subplots_adjust(wspace=0.3)

     axes[0].plot(range(1, len(Yd_train) + 1), Yd_train, marker='o', label='YD (Deseado)')
     axes[0].plot(range(1, len(Yr_train_1) + 1), Yr_train_1, marker='x', linestyle='--', label='YR (Red)')
     axes[0].set_title("YD vs YR (Entrenamiento)")
     axes[0].set_xlabel("Patrón")
     axes[0].set_ylabel("Valor de salida")
     axes[0].legend()
     axes[0].grid(True)

     axes[1].plot([1], [met_train["EG"]], marker='o', markersize=10, label=f"EG = {met_train['EG']:.4f}")
     axes[1].axhline(y=float(self.error_opt.get()), color='r', linestyle='--', label=f"Error óptimo = {float(self.error_opt.get())}")
     axes[1].set_title("EG vs Error óptimo (Entrenamiento)")
     axes[1].set_xticks([1])
     axes[1].set_xticklabels(["Iteración 1"])
     axes[1].set_ylabel("Error")
     axes[1].legend()
     axes[1].grid(True)

     errors = np.abs(Yd_train - Yr_train_1)
     axes[2].bar(range(1, len(errors) + 1), errors)
     axes[2].set_title("Error por Patrón (Entrenamiento)")
     axes[2].set_xlabel("Patrón")
     axes[2].set_ylabel("|YD - YR|")
     axes[2].grid(axis='y')

     plt.suptitle("Gráficas del Entrenamiento RBF", fontsize=14)
     plt.show()


    def save_plots(self):
     if self.weights is None or self.centers is None:
        messagebox.showwarning("Sin modelo", "Entrena o carga un modelo primero.")
        return
     if self.X_train is None or self.Y_train is None:
        messagebox.showwarning("Sin datos", "No hay datos de entrenamiento.")
        return

     folder = filedialog.askdirectory(title="Selecciona carpeta para guardar las gráficas")
     if not folder:
        return

     name = simpledialog.askstring("Nombre base", "Nombre base para las gráficas (sin extensión):")
     if not name:
        name = pd.Timestamp.now().strftime("trainplots_%Y%m%d_%H%M%S")

     Phi_train = rbf_core.compute_phi(self.X_train, self.centers)
     A_train = rbf_core.build_A(Phi_train)
     Yr_train = rbf_core.predict(A_train, self.weights)
     Yd_train = np.atleast_2d(self.Y_train)[:, 0]
     Yr_train_1 = np.atleast_2d(Yr_train)[:, 0]
     met_train = metrics_mod.metrics(self.Y_train, Yr_train)
     errors = np.abs(Yd_train - Yr_train_1)

     fig1 = plt.figure(figsize=(8, 6))
     plt.plot(range(1, len(Yd_train) + 1), Yd_train, marker='o', label='YD (Deseado)')
     plt.plot(range(1, len(Yr_train_1) + 1), Yr_train_1, marker='x', linestyle='--', label='YR (Red)')
     plt.title("YD vs YR (Entrenamiento)")
     plt.xlabel("Patrón")
     plt.ylabel("Valor de salida")
     plt.legend()
     plt.grid(True)
     path1 = os.path.join(folder, f"{name}_Yd_vs_Yr.png")
     plt.savefig(path1, bbox_inches="tight")
     plt.close(fig1)

     fig2 = plt.figure(figsize=(6, 6))
     plt.plot([1], [met_train["EG"]], marker='o', markersize=10, label=f"EG = {met_train['EG']:.4f}")
     plt.axhline(y=float(self.error_opt.get()), color='r', linestyle='--', label=f"Error óptimo = {float(self.error_opt.get())}")
     plt.title("EG vs Error óptimo (Entrenamiento)")
     plt.xticks([1], ["Iteración 1"])
     plt.ylabel("Error")
     plt.legend()
     plt.grid(True)
     path2 = os.path.join(folder, f"{name}_EG_vs_opt.png")
     plt.savefig(path2, bbox_inches="tight")
     plt.close(fig2)

     fig3 = plt.figure(figsize=(8, 6))
     plt.bar(range(1, len(errors) + 1), errors)
     plt.title("Error por Patrón (Entrenamiento)")
     plt.xlabel("Patrón")
     plt.ylabel("|YD - YR|")
     plt.grid(axis='y')
     path3 = os.path.join(folder, f"{name}_Error_por_Patron.png")
     plt.savefig(path3, bbox_inches="tight")
     plt.close(fig3)

     messagebox.showinfo("Guardado", f"Gráficas guardadas en:\n{folder}")
     self.log(f"Gráficas de entrenamiento guardadas como {name}_*.png en {folder}")

    def reset_all(self):
        if messagebox.askyesno("Confirmar", "¿Deseas limpiar todo? Asegúrate de haber guardado tus datos y modelos."):
            self.df_raw = self.df_numeric = None
            self.X = self.Y = self.X_train = self.X_sim = self.Y_train = self.Y_sim = None
            self.scaler = None
            self.centers = None
            self.weights = None
            self.meta = {}
            self.n_centers.set(2)
            self.error_opt.set(0.1)

            for tv in [self.tv_train, self.tv_sim, self.tv_centers, self.tv_train_results, self.tv_weights, self.tv_sim_results, self.tv_metrics]:
                for k in tv.get_children():
                    tv.delete(k)

            self.log_text.config(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.config(state="disabled")

            self.dataset_status_lbl.config(text="Dataset: No cargado")
            self.dataset_info_lbl.config(text="Entradas: 0 | Salidas: 0 | Patrones: 0")
            self.train_state_lbl.config(text="Estado: No iniciado")

            self.log("Estado limpiado. Todo listo para empezar de nuevo.")
            messagebox.showinfo("Listo", "El sistema se limpió correctamente.")

if __name__ == "__main__":
    app = RBFApp()
    app.mainloop()

